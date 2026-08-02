#include "Analysis/Analysis.hpp"
#include "MatlabLikeData.hpp"

#include "Optimization/Filtering.hpp"
#include "ROM/ROM.hpp"
#include "Sensitivity/Sensitivity.hpp"
#include "helper/MyAlgebra.hpp"
#include "helper/PetscEigenBridge.hpp"
#include "helper/PetscEigenHelper.hpp"
#include "JdumasMMASolver.hpp"
#include "MatlabMMA.hpp"
#include "MMASolver.hpp"

#include <petscksp.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <omp.h>

namespace
    {
        int nvar_of(const Eigen::MatrixXd &x)
        {
            return static_cast<int>(x.cols());
        }
        Eigen::MatrixXd unflatten(const std::vector<double> &x, int nelem, int nvar)
        {
            return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                x.data(), nelem, nvar);
        }
        void add_scaled(Eigen::MatrixXd &a, const Eigen::MatrixXd &b, double scale)
        {
            if (b.size() == 0) return;
            if (a.size() == 0) {
                a = scale * b;
                return;
            }
            a.noalias() += scale * b;
        }

        [[maybe_unused]] void element_dofs(const FEM &fem, int elem_id, std::vector<int> &dofs)
        {
            const auto &elem_dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            dofs.resize(elem_dofs.size());
            for (std::size_t i = 0; i < elem_dofs.size(); ++i)
                dofs[i] = static_cast<int>(elem_dofs[i]);
        }

        double ramp_derivative(const std::vector<double> &para, double x, double p)
        {
            const double denom = 1.0 + p * (1.0 - x);
            const double dg = (1.0 + p) / (denom * denom);
            return (para[1] - para[0]) * dg;
        }

        void set_vec_from_column_root(Vec v, const Eigen::MatrixXd &values, int k, double scale = 1.0)
        {
            set_vec_from_eigen_column(v, values, k, scale);
        }

        void set_vec_fixed_from_column_root(Vec v, const std::vector<int> &fixed,
                                            const Eigen::MatrixXd &values, int k, double scale = 1.0)
        {
            set_vec_fixed_from_eigen_column(v, fixed, values, k, scale);
        }

        void zero_fixed(Vec v, const std::vector<int> &fixed)
        {
            if (fixed.empty())
                return;
            PetscMPIInt size = 1;
            MPI_Comm_size(PetscObjectComm((PetscObject)v), &size);
            if (size == 1)
            {
                PetscInt n = 0;
                VecGetLocalSize(v, &n);
                PetscScalar *a = nullptr;
                VecGetArray(v, &a);
                for (int id : fixed)
                {
                    if (id < 0 || id >= n)
                        throw std::runtime_error("zero_fixed: fixed dof out of range");
                    a[id] = 0.0;
                }
                VecRestoreArray(v, &a);
                return;
            }

            PetscMPIInt rank = 0;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            if (rank == 0)
            {
                std::vector<PetscInt> ids(fixed.begin(), fixed.end());
                std::vector<PetscScalar> vals(ids.size(), 0.0);
                VecSetValues(v, static_cast<PetscInt>(ids.size()), ids.data(), vals.data(), INSERT_VALUES);
            }
            VecAssemblyBegin(v);
            VecAssemblyEnd(v);
        }

        void solve_adjoint_full(ROMBase &rom, Mat lhs, Vec rhs, Vec lambda, const Analysis &analysis, int Iter, int time_step)
        {
            ROMContext ctx;
            ctx.analysis_id = analysis.id;
            ctx.opt_iter = Iter;
            ctx.solve_step = time_step + 1;
            ctx.time_step = time_step;
            ctx.time = time_step * analysis.dt;
            ctx.role = "adjoint";
            ctx.system_type = "newmark_adjoint_effective";
            ctx.is_end_of_analysis = time_step == 0;
            rom.set_petsc_system(lhs, rhs, lambda);
            if (!rom.use_mor)
            {
                Eigen::SparseMatrix<double> empty_lhs;
                Eigen::VectorXd empty_rhs;
                Eigen::VectorXd empty_lambda;
                rom.main(empty_lhs, empty_rhs, empty_lambda, ctx);
                return;
            }
            const Eigen::SparseMatrix<double> lhs_e = petsc_mat_to_eigen(lhs);
            const Eigen::VectorXd rhs_e = petsc_vec_to_eigen(rhs);
            Eigen::VectorXd lambda_e = Eigen::VectorXd::Zero(rhs_e.size());
            rom.main(lhs_e, rhs_e, lambda_e, ctx);
            eigen_to_petsc_vec(lambda_e, lambda);
        }

        Mat duplicate_with_dirichlet(Mat A, const std::vector<int> &fixed)
        {
            Mat B = nullptr;
            MatDuplicate(A, MAT_COPY_VALUES, &B);
            MyAlgebra::Petsc::apply_dirichlet_to_matrix(B, fixed, 1.0);
            return B;
        }

        Eigen::MatrixXd explicit_gradient_dynamic(const FEM &fem,
                                                                   const Analysis &analysis,
                                                                   const Eigen::MatrixXd &xPhys,
                                                                   const ObjectiveGroup &object,
                                                                   const Eigen::MatrixXd &lambda_all)
        {
            using Clock = std::chrono::steady_clock;
            const auto timer_begin = Clock::now();
            PetscMPIInt rank = 0;
            PetscMPIInt size = 1;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            MPI_Comm_size(PETSC_COMM_WORLD, &size);

            const int nstep = analysis.Nstep;
            const int nvar = nvar_of(xPhys);
            const int elem_start = (fem.NElem * rank) / size;
            const int elem_end = (fem.NElem * (rank + 1)) / size;
            int sens_element_threads = 1;
            if (const char *env = std::getenv("SENS_ELEMENT_THREADS"))
                sens_element_threads = std::max(1, std::atoi(env));
            if (size != 1)
                sens_element_threads = 1;
            const Region &region = analysis.regions.front();
            const double nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();

            auto t0 = Clock::now();
            std::vector<double> dE(static_cast<std::size_t>(fem.NElem * nvar), 0.0);
            std::vector<double> drho(static_cast<std::size_t>(fem.NElem * nvar), 0.0);
            for (int e = elem_start; e < elem_end; ++e)
            {
                for (int v = 0; v < nvar; ++v)
                {
                    const double x = xPhys(e, v);
                    const std::size_t id = static_cast<std::size_t>(e * nvar + v);
                    dE[id] = ramp_derivative(region.mat_paras.E_inf, x, region.interp.p_E_elastic);
                    drho[id] = ramp_derivative(region.mat_paras.rho, x, region.interp.p_rho);
                }
            }
            auto t1 = Clock::now();
            const double timing_interp_derivative = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            ElementMatrix uniform_em;
            if (fem.uniform_element_matrices)
                uniform_em = fem.compute_element_matrices(analysis.regions.front().elem_ids.front(), nu, fem.Thickness);
            t1 = Clock::now();
            const double timing_element_matrix = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            std::vector<double> df_dx_local(static_cast<std::size_t>(fem.NElem * nvar), 0.0);
            t1 = Clock::now();
            const double timing_alloc_gradient = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            // TEMP THREAD DEBUG: comment out this block after comparing master/worker.
            if (rank == 0)
            {
                int actual_threads = 1;
                #pragma omp parallel num_threads(sens_element_threads) if(sens_element_threads > 1)
                {
                    #pragma omp single
                    actual_threads = omp_get_num_threads();
                }
                std::cout << "  thread_debug explicit_gradient_dynamic: requested_threads="
                          << sens_element_threads
                          << " actual_threads=" << actual_threads
                          << " omp_get_max_threads=" << omp_get_max_threads()
                          << "\n";
            }
            #pragma omp parallel for schedule(static) num_threads(sens_element_threads) if(sens_element_threads > 1)
            for (int e = elem_start; e < elem_end; ++e)
            {
                ElementMatrix local_em;
                const ElementMatrix *em_ptr = &uniform_em;
                if (!fem.uniform_element_matrices)
                {
                    local_em = fem.compute_element_matrices(e, nu, fem.Thickness);
                    em_ptr = &local_em;
                }
                const ElementMatrix &em = *em_ptr;
                if (em.ndof > 24)
                    throw std::runtime_error("explicit_gradient_dynamic: element ndof > 24 is not supported by stack buffer");

                const auto &dofs = fem.ElementDofs.at(static_cast<std::size_t>(e));
                Eigen::Matrix<double, 24, 1> u_elem_buffer;
                Eigen::Matrix<double, 24, 1> acc_elem_buffer;
                Eigen::Matrix<double, 24, 1> lambda_elem_buffer;
                Eigen::Matrix<double, 24, 1> Ku_elem_buffer;
                Eigen::Matrix<double, 24, 1> Ma_elem_buffer;

                for (int k = 0; k <= nstep; ++k)
                {
                    const auto u = analysis.results.U.col(k);
                    const auto acc = analysis.results.A.col(k);
                    const auto lambda = lambda_all.col(k);

                    for (int i = 0; i < em.ndof; ++i)
                    {
                        const int dof_i = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                        u_elem_buffer(i) = u(dof_i);
                        acc_elem_buffer(i) = acc(dof_i);
                        lambda_elem_buffer(i) = lambda(dof_i);
                    }
                    auto u_elem = u_elem_buffer.head(em.ndof);
                    auto acc_elem = acc_elem_buffer.head(em.ndof);
                    auto lambda_elem = lambda_elem_buffer.head(em.ndof);
                    auto Ku_elem = Ku_elem_buffer.head(em.ndof);
                    auto Ma_elem = Ma_elem_buffer.head(em.ndof);

                    Ku_elem.noalias() = em.ke * u_elem;
                    if (k != 0)
                    {
                        Ma_elem.noalias() = em.me * acc_elem;
                    }
                    else
                    {
                        Ma_elem.setZero();
                    }

                    const double lambda_dot_Ku = lambda_elem.dot(Ku_elem);
                    const double lambda_dot_Ma = lambda_elem.dot(Ma_elem);

                    for (int v = 0; v < nvar; ++v)
                    {
                        const std::size_t ev = static_cast<std::size_t>(e * nvar + v);
                        const double dE_ev = dE[ev];
                        const double drho_ev = drho[ev];
                        const double val = dE_ev * lambda_dot_Ku + drho_ev * lambda_dot_Ma;
                        df_dx_local[ev] += val;
                    }
                }
            }
            t1 = Clock::now();
            const double timing_element_loop = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            std::vector<double> global_flat(df_dx_local.size(), 0.0);
            t1 = Clock::now();
            const double timing_flatten = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            MPI_Allreduce(df_dx_local.data(), global_flat.data(), static_cast<int>(df_dx_local.size()), MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
            t1 = Clock::now();
            const double timing_allreduce = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            Eigen::MatrixXd df_dx = unflatten(global_flat, fem.NElem, nvar);
            add_scaled(df_dx, object.d_explicit_df_dx, 1.0);
            t1 = Clock::now();
            const double timing_unflatten_add = std::chrono::duration<double>(t1 - t0).count();

            if (rank == 0)
            {
                const double timing_total = std::chrono::duration<double>(Clock::now() - timer_begin).count();
                std::cout << "  explicit_gradient_dynamic[s]:"
                          << " threads=" << sens_element_threads
                          << " interp_derivative=" << timing_interp_derivative
                          << " element_matrix=" << timing_element_matrix
                          << " alloc_gradient=" << timing_alloc_gradient
                          << " element_loop=" << timing_element_loop
                          << " flatten=" << timing_flatten
                          << " allreduce=" << timing_allreduce
                          << " unflatten_add=" << timing_unflatten_add
                          << " total=" << timing_total
                          << "\n";
            }
            return df_dx;
        }

        Eigen::MatrixXd dynamic_elastic_base_full_impl(const FEM &fem,
                                                       Analysis &analysis,
                                                       const Eigen::MatrixXd &xPhys,
                                                       ObjectiveGroup &object,
                                                       const Filtering &filtering,
                                                       const FilteringState &filtering_state,
                                                       int Iter)
        {
            (void)xPhys;
            const int nstep = analysis.Nstep;
            const double beta = 0.25;
            const double gamma = 0.5;
            const double dt = analysis.dt;
            const double a0c = 1.0 / (beta * dt * dt);
            const double dHdu = a0c;
            const double dLdu = gamma / (beta * dt);
            const double dHda_new = -(0.5 / beta - 1.0);
            const double dHdv_new = -1.0 / (beta * dt);
            const double dHdu_new = -a0c;
            const double dLda_new = (1.0 - 0.5 * gamma / beta) * dt;
            const double dLdv_new = 1.0 - gamma / beta;
            const double dLdu_new = -gamma / (beta * dt);

            if (!object.ROM)
                throw std::runtime_error("PetscSensitivity requires ObjectiveGroup::ROM initialized");
            Mat K0 = duplicate_with_dirichlet(analysis.results.K_inf, analysis.results.SPCdofs);
            Mat Keff = duplicate_with_dirichlet(analysis.results.K_eff, analysis.results.SPCdofs);

            Vec lambda = nullptr;
            Vec lambda_new = nullptr;
            Vec mu = nullptr;
            Vec mu_new = nullptr;
            Vec eta = nullptr;
            Vec eta_new = nullptr;
            Vec rhs = nullptr;
            Vec Mlambda = nullptr;
            MatCreateVecs(Keff, &lambda, &rhs);
            VecDuplicate(lambda, &lambda_new);
            VecDuplicate(lambda, &mu);
            VecDuplicate(lambda, &mu_new);
            VecDuplicate(lambda, &eta);
            VecDuplicate(lambda, &eta_new);
            VecDuplicate(lambda, &Mlambda);

            Eigen::MatrixXd lambda_all(fem.ndof, nstep + 1);

            object.ROM->set_petsc_factorized_solver(Keff, analysis.results.factorized_K_eff);

            int k = nstep;
            set_vec_from_column_root(mu, object.d_explicit_dfda_all, k);
            set_vec_from_column_root(eta, object.d_explicit_dfdv_all, k);
            set_vec_from_column_root(rhs, object.d_explicit_dfdu_all, k, -1.0);
            VecAXPY(rhs, -dHdu, mu);
            VecAXPY(rhs, -dLdu, eta);
            set_vec_fixed_from_column_root(rhs, analysis.results.SPCdofs, object.d_explicit_dfdFext_all, k, -1.0);
            solve_adjoint_full(*object.ROM, Keff, rhs, lambda, analysis, Iter, k);
            lambda_all.col(k) = gather_vec_to_all_local(lambda);

            for (k = nstep - 1; k >= 0; --k)
            {
                VecCopy(lambda, lambda_new);
                VecCopy(mu, mu_new);
                VecCopy(eta, eta_new);
                MatMult(analysis.results.M, lambda_new, Mlambda);

                set_vec_from_column_root(mu, object.d_explicit_dfda_all, k);
                VecAXPY(mu, -(0.5 / beta - 1.0), Mlambda);
                VecAXPY(mu, dHda_new, mu_new);
                VecAXPY(mu, dLda_new, eta_new);
                set_vec_from_column_root(eta, object.d_explicit_dfdv_all, k);
                VecAXPY(eta, -(1.0 / (beta * dt)), Mlambda);
                VecAXPY(eta, dHdv_new, mu_new);
                VecAXPY(eta, dLdv_new, eta_new);
                zero_fixed(mu, analysis.results.SPCdofs);
                zero_fixed(eta, analysis.results.SPCdofs);

                set_vec_from_column_root(rhs, object.d_explicit_dfdu_all, k, -1.0);
                VecAXPY(rhs, a0c, Mlambda);
                VecAXPY(rhs, -dHdu_new, mu_new);
                VecAXPY(rhs, -dLdu_new, eta_new);
                if (k != 0)
                {
                    VecAXPY(rhs, -dHdu, mu);
                    VecAXPY(rhs, -dLdu, eta);
                }
                set_vec_fixed_from_column_root(rhs, analysis.results.SPCdofs, object.d_explicit_dfdFext_all, k, -1.0);
                if (k == 0)
                {
                    object.ROM->set_petsc_factorized_solver(K0, analysis.results.factorized_K_initial);
                }
                solve_adjoint_full(*object.ROM, k == 0 ? K0 : Keff, rhs, lambda, analysis, Iter, k);
                lambda_all.col(k) = gather_vec_to_all_local(lambda);
            }

            Eigen::MatrixXd dfdx_phys =
                explicit_gradient_dynamic(fem, analysis, filtering_state.xPhys, object, lambda_all);

            MatDestroy(&K0);
            MatDestroy(&Keff);
            VecDestroy(&lambda);
            VecDestroy(&lambda_new);
            VecDestroy(&mu);
            VecDestroy(&mu_new);
            VecDestroy(&eta);
            VecDestroy(&eta_new);
            VecDestroy(&rhs);
            VecDestroy(&Mlambda);
            return filtering_sensitivity(dfdx_phys, filtering, filtering_state);
        }
        Eigen::MatrixXd AVM(const FEM &fem,
                            Analysis &analysis,
                            const Eigen::MatrixXd &,
                            const std::vector<int> &x_label,
                            ObjectiveGroup &object,
                            const Filtering &filtering,
                            const FilteringState &filtering_state,
                            int Iter)
        {
            Eigen::MatrixXd out = dynamic_elastic_base_full_impl(fem, analysis, filtering_state.xPhys, object, filtering, filtering_state, Iter);
            for (Eigen::Index e = 0; e < out.rows(); ++e)
            {
                if (x_label[static_cast<std::size_t>(e)] != 0)
                    out.row(e).setZero();
            }
            return out;
        }

    } // namespace

Eigen::MatrixXd PetscSensitivity::objective_sensitivity(const FEM& fem,
                                                        Analysis& analysis,
                                                        const Eigen::MatrixXd& x,
                                                        const std::vector<int>& x_label,
                                                        ObjectiveGroup& object,
                                                        const Filtering& filtering,
                                                        const FilteringState& filtering_state,
                                                        int Iter) {
    return AVM(fem, analysis, x, x_label, object, filtering, filtering_state, Iter);
}
