#include "Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"
#include "ROM/ROM.hpp"
#include "helper/MemoryDebug.hpp"
#include "helper/PetscEigenBridge.hpp"
#include "helper/PetscEigenHelper.hpp"

#include <petscksp.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
}

Eigen::MatrixXd unflatten(const std::vector<double>& x, int nelem, int nvar) {
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
        x.data(), nelem, nvar);
}

void add_scaled(Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double scale) {
    if (b.size() == 0) return;
    if (a.size() == 0) {
        a = scale * b;
        return;
    }
    a.noalias() += scale * b;
}

double ramp_derivative(const std::vector<double>& para, double x, double p) {
    const double denom = 1.0 + p * (1.0 - x);
    const double dg = (1.0 + p) / (denom * denom);
    return (para[1] - para[0]) * dg;
}

void set_vec_from_column_root(Vec v, const Eigen::MatrixXd& values, int k, double scale = 1.0) {
    set_vec_from_eigen_column(v, values, k, scale);
}

void set_vec_fixed_from_column_root(Vec v,
                                    const std::vector<int>& fixed,
                                    const Eigen::MatrixXd& values,
                                    int k,
                                    double scale = 1.0) {
    set_vec_fixed_from_eigen_column(v, fixed, values, k, scale);
}

void zero_fixed(Vec v, const std::vector<int>& fixed) {
    if (fixed.empty()) return;

    PetscMPIInt size = 1;
    MPI_Comm_size(PetscObjectComm(reinterpret_cast<PetscObject>(v)), &size);
    if (size == 1) {
        PetscInt n = 0;
        VecGetLocalSize(v, &n);
        PetscScalar* a = nullptr;
        VecGetArray(v, &a);
        for (int id : fixed) {
            if (id < 0 || id >= n) throw std::runtime_error("zero_fixed: fixed dof out of range");
            a[id] = 0.0;
        }
        VecRestoreArray(v, &a);
        return;
    }

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
        std::vector<PetscInt> ids(fixed.begin(), fixed.end());
        std::vector<PetscScalar> vals(ids.size(), 0.0);
        VecSetValues(v, static_cast<PetscInt>(ids.size()), ids.data(), vals.data(), INSERT_VALUES);
    }
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}

double petsc_wall_time() {
    PetscLogDouble t = 0.0;
    PetscTime(&t);
    return static_cast<double>(t);
}

void solve_adjoint_full(ROMBase& rom,
                        Mat lhs,
                        Vec rhs,
                        Vec lambda,
                        const Analysis& analysis,
                        int Iter,
                        int time_step,
                        Eigen::SparseMatrix<double>* lhs_e_cache = nullptr) {
    const auto& dynamic = dynamic_cast<const DynamicLinearElastic&>(analysis);
    ROMContext ctx;
    ctx.analysis_id = analysis.id;
    ctx.opt_iter = Iter;
    ctx.solve_step = time_step + 1;
    ctx.time_step = time_step;
    ctx.time = time_step * dynamic.dt;
    ctx.role = "adjoint";
    ctx.system_type = "newmark_adjoint_effective";
    ctx.is_end_of_analysis = time_step == 0;

    rom.set_petsc_system(lhs, rhs, lambda);
    if (!rom.use_mor) {
        Eigen::SparseMatrix<double> empty_lhs;
        Eigen::VectorXd empty_rhs;
        Eigen::VectorXd empty_lambda;
        rom.main(empty_lhs, empty_rhs, empty_lambda, ctx);
        return;
    }

    const double tm = petsc_wall_time();
    Eigen::SparseMatrix<double> local_lhs_e;
    const Eigen::SparseMatrix<double>* lhs_e_ptr = nullptr;
    if (lhs_e_cache) {
        if (lhs_e_cache->rows() == 0 || lhs_e_cache->cols() == 0) {
            *lhs_e_cache = petsc_mat_to_eigen(lhs);
        }
        lhs_e_ptr = lhs_e_cache;
    } else {
        local_lhs_e = petsc_mat_to_eigen(lhs);
        lhs_e_ptr = &local_lhs_e;
    }
    const double matrix_convert_time = petsc_wall_time() - tm;

    const double tv = petsc_wall_time();
    const Eigen::VectorXd rhs_e = petsc_vec_to_eigen(rhs);
    const double vec_convert_time = petsc_wall_time() - tv;

    Eigen::VectorXd lambda_e = Eigen::VectorXd::Zero(rhs_e.size());
    rom.main(*lhs_e_ptr, rhs_e, lambda_e, ctx);
    eigen_to_petsc_vec(lambda_e, lambda);

    if (!rom.ROM_info.empty() && !rom.ROM_info.back().empty()) {
        ROMInfo& info = rom.ROM_info.back().back();
        info.timing_petsc_to_eigen_mat += matrix_convert_time;
        info.timing_petsc_to_eigen_vec += vec_convert_time;
    }
}

Eigen::MatrixXd explicit_gradient_dynamic(const FEM& fem,
                                          const Analysis& analysis,
                                          const Eigen::MatrixXd& xPhys,
                                          const DynamicLinearElasticAVM& object,
                                          const Eigen::MatrixXd& lambda_all) {
    using Clock = std::chrono::steady_clock;

    const auto timer_begin = Clock::now();
    const auto elapsed = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const int nstep = analysis.Nstep;
    const int nvar = nvar_of(xPhys);
    const int elem_start = (fem.NElem * rank) / size;
    const int elem_end = (fem.NElem * (rank + 1)) / size;

    if (xPhys.rows() != fem.NElem) {
        throw std::runtime_error("explicit_gradient_dynamic: xPhys row count does not match fem.NElem");
    }
    if (analysis.results.U.cols() < nstep + 1 ||
        analysis.results.A.cols() < nstep + 1 ||
        lambda_all.cols() < nstep + 1) {
        throw std::runtime_error("explicit_gradient_dynamic: insufficient time-step columns");
    }
    if (analysis.results.U.rows() != analysis.results.A.rows() ||
        analysis.results.U.rows() != lambda_all.rows()) {
        throw std::runtime_error("explicit_gradient_dynamic: U, A and lambda_all row mismatch");
    }

    int sens_element_threads = 1;
    if (const char* env = std::getenv("SENS_ELEMENT_THREADS")) {
        sens_element_threads = std::max(1, std::atoi(env));
    }
    if (size != 1) sens_element_threads = 1;

    if (rank == 0) {
        int actual_threads = 1;
        #pragma omp parallel num_threads(sens_element_threads) if(sens_element_threads > 1)
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }
        std::cout << "  thread_debug explicit_gradient_dynamic:"
                  << " requested_threads=" << sens_element_threads
                  << " actual_threads=" << actual_threads
                  << " omp_get_max_threads=" << omp_get_max_threads()
                  << '\n';
    }

    auto t0 = Clock::now();
    std::vector<double> df_dx_local(static_cast<std::size_t>(fem.NElem) * static_cast<std::size_t>(nvar), 0.0);
    auto t1 = Clock::now();
    const double timing_alloc_gradient = elapsed(t0, t1);

    t0 = Clock::now();
    for (const Region& region : analysis.regions) {
        if (region.status != "design_val") continue;
        if (region.elem_ids.empty()) continue;
        if (region.mat_paras.val_ids.empty()) {
            throw std::runtime_error("explicit_gradient_dynamic: design_val region has no val_ids");
        }
        if (region.mat_paras.E_inf.empty()) {
            throw std::runtime_error("explicit_gradient_dynamic: design_val region has no elastic parameters");
        }
        if (region.mat_paras.rho.empty()) {
            throw std::runtime_error("explicit_gradient_dynamic: design_val region has no density parameters");
        }

        for (int val_id : region.mat_paras.val_ids) {
            if (val_id <= 0 || val_id > nvar) {
                throw std::runtime_error("explicit_gradient_dynamic: region val_id is outside xPhys columns");
            }
        }

        for (int elem_id : region.elem_ids) {
            if (elem_id < 0 || elem_id >= fem.NElem) {
                throw std::runtime_error("explicit_gradient_dynamic: invalid element id in region");
            }
            if (elem_id < elem_start || elem_id >= elem_end) continue;
            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            if (dofs.size() > 24) {
                throw std::runtime_error("explicit_gradient_dynamic: element ndof > 24 is not supported");
            }
        }

        const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            uniform_em = fem.compute_element_matrices(region.elem_ids.front(), region_nu, fem.Thickness);
            if (uniform_em.ndof > 24) {
                throw std::runtime_error("explicit_gradient_dynamic: uniform element ndof > 24");
            }
        }

        const int region_nelem = static_cast<int>(region.elem_ids.size());
        #pragma omp parallel for schedule(static) num_threads(sens_element_threads) if(sens_element_threads > 1)
        for (int local_e = 0; local_e < region_nelem; ++local_e) {
            const int elem_id = region.elem_ids[static_cast<std::size_t>(local_e)];
            if (elem_id < elem_start || elem_id >= elem_end) continue;

            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                local_em = fem.compute_element_matrices(elem_id, region_nu, fem.Thickness);
                em_ptr = &local_em;
            }

            const ElementMatrix& em = *em_ptr;
            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));

            Eigen::Matrix<double, 24, 1> u_elem_buffer;
            Eigen::Matrix<double, 24, 1> acc_elem_buffer;
            Eigen::Matrix<double, 24, 1> lambda_elem_buffer;
            Eigen::Matrix<double, 24, 1> Ku_elem_buffer;
            Eigen::Matrix<double, 24, 1> Ma_elem_buffer;

            double sum_lambda_dot_Ku = 0.0;
            double sum_lambda_dot_Ma = 0.0;

            for (int k = 0; k <= nstep; ++k) {
                const auto u = analysis.results.U.col(k);
                const auto acc = analysis.results.A.col(k);
                const auto lambda = lambda_all.col(k);

                for (int i = 0; i < em.ndof; ++i) {
                    const int dof_i = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                    u_elem_buffer(i) = u(dof_i);
                    acc_elem_buffer(i) = acc(dof_i);
                    lambda_elem_buffer(i) = lambda(dof_i);
                }

                const auto u_elem = u_elem_buffer.head(em.ndof);
                const auto acc_elem = acc_elem_buffer.head(em.ndof);
                const auto lambda_elem = lambda_elem_buffer.head(em.ndof);
                auto Ku_elem = Ku_elem_buffer.head(em.ndof);
                auto Ma_elem = Ma_elem_buffer.head(em.ndof);

                Ku_elem.noalias() = em.ke * u_elem;
                if (k == 0) Ma_elem.setZero();
                else Ma_elem.noalias() = em.me * acc_elem;

                sum_lambda_dot_Ku += lambda_elem.dot(Ku_elem);
                sum_lambda_dot_Ma += lambda_elem.dot(Ma_elem);
            }

            for (int val_id : region.mat_paras.val_ids) {
                const double x = xPhys(elem_id, val_id - 1);
                const double dE = ramp_derivative(region.mat_paras.E_inf, x, region.interp.p_E_elastic);
                const double drho = ramp_derivative(region.mat_paras.rho, x, region.interp.p_rho);
                const std::size_t ev =
                    static_cast<std::size_t>(elem_id) * static_cast<std::size_t>(nvar) +
                    static_cast<std::size_t>(val_id - 1);
                df_dx_local[ev] += dE * sum_lambda_dot_Ku + drho * sum_lambda_dot_Ma;
            }
        }
    }
    t1 = Clock::now();
    const double timing_element_loop = elapsed(t0, t1);

    t0 = Clock::now();
    std::vector<double> global_flat(df_dx_local.size(), 0.0);
    t1 = Clock::now();
    const double timing_alloc_global = elapsed(t0, t1);

    if (df_dx_local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("explicit_gradient_dynamic: MPI_Allreduce count exceeds INT_MAX");
    }

    t0 = Clock::now();
    MPI_Allreduce(df_dx_local.data(),
                  global_flat.data(),
                  static_cast<int>(df_dx_local.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  PETSC_COMM_WORLD);
    t1 = Clock::now();
    const double timing_allreduce = elapsed(t0, t1);

    t0 = Clock::now();
    Eigen::MatrixXd df_dx = unflatten(global_flat, fem.NElem, nvar);
    add_scaled(df_dx, object.d_explicit_df_dx, 1.0);
    t1 = Clock::now();
    const double timing_unflatten_add = elapsed(t0, t1);

    if (rank == 0) {
        const double timing_total = elapsed(timer_begin, Clock::now());
        std::cout << "  explicit_gradient_dynamic[s]:"
                  << " ranks=" << size
                  << " threads=" << sens_element_threads
                  << " alloc_gradient=" << timing_alloc_gradient
                  << " element_loop=" << timing_element_loop
                  << " alloc_global=" << timing_alloc_global
                  << " allreduce=" << timing_allreduce
                  << " unflatten_add=" << timing_unflatten_add
                  << " total=" << timing_total
                  << '\n';
    }
    return df_dx;
}

Eigen::MatrixXd dynamic_elastic_base_full_impl(const FEM& fem,
                                               Analysis& analysis,
                                               DynamicLinearElasticAVM& object,
                                               const Filtering& filtering,
                                               const FilteringState& filtering_state,
                                               int Iter) {
    auto& dynamic = dynamic_cast<DynamicLinearElastic&>(analysis);
    const int nstep = analysis.Nstep;
    const double beta = 0.25;
    const double gamma = 0.5;
    const double dt = dynamic.dt;
    const double a0c = 1.0 / (beta * dt * dt);
    const double dHdu = a0c;
    const double dLdu = gamma / (beta * dt);
    const double dHda_new = -(0.5 / beta - 1.0);
    const double dHdv_new = -1.0 / (beta * dt);
    const double dHdu_new = -a0c;
    const double dLda_new = (1.0 - 0.5 * gamma / beta) * dt;
    const double dLdv_new = 1.0 - gamma / beta;
    const double dLdu_new = -gamma / (beta * dt);

    if (!object.ROM) throw std::runtime_error("PetscDynamicLinearElasticAVM requires ROM initialized");
    MemoryDebug::print("petsc_sens_begin");

    Mat K0 = analysis.results.K_inf;
    MemoryDebug::print("petsc_sens_after_get_K0");
    Mat Keff = analysis.results.K_eff;
    MemoryDebug::print("petsc_sens_after_get_Keff");

    Eigen::SparseMatrix<double> K0_eigen_cache;
    Eigen::SparseMatrix<double> Keff_eigen_cache;

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
    MemoryDebug::print("petsc_sens_after_vec_create");

    Eigen::MatrixXd lambda_all(fem.ndof, nstep + 1);
    MemoryDebug::print("petsc_sens_after_lambda_all_alloc");

    object.ROM->set_petsc_factorized_solver(Keff, analysis.results.factorized_K_eff);

    int k = nstep;
    set_vec_from_column_root(mu, object.d_explicit_dfda_all, k);
    set_vec_from_column_root(eta, object.d_explicit_dfdv_all, k);
    set_vec_from_column_root(rhs, object.d_explicit_dfdu_all, k, -1.0);
    VecAXPY(rhs, -dHdu, mu);
    VecAXPY(rhs, -dLdu, eta);
    set_vec_fixed_from_column_root(rhs, analysis.results.SPCdofs, object.d_explicit_dfdFext_all, k, -1.0);
    solve_adjoint_full(*object.ROM, Keff, rhs, lambda, analysis, Iter, k, &Keff_eigen_cache);
    lambda_all.col(k) = gather_vec_to_all_local(lambda);
    MemoryDebug::print("petsc_sens_after_step_end");

    for (k = nstep - 1; k >= 0; --k) {
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
        if (k != 0) {
            VecAXPY(rhs, -dHdu, mu);
            VecAXPY(rhs, -dLdu, eta);
        }
        set_vec_fixed_from_column_root(rhs, analysis.results.SPCdofs, object.d_explicit_dfdFext_all, k, -1.0);

        if (k == 0) {
            object.ROM->set_petsc_factorized_solver(nullptr, nullptr);
            if (analysis.results.factorized_K_eff) KSPDestroy(&analysis.results.factorized_K_eff);

            object.ROM->set_petsc_factorized_solver(K0, analysis.results.factorized_K_initial);
            solve_adjoint_full(*object.ROM, K0, rhs, lambda, analysis, Iter, k, &K0_eigen_cache);

            object.ROM->set_petsc_factorized_solver(nullptr, nullptr);
            if (analysis.results.factorized_K_initial) KSPDestroy(&analysis.results.factorized_K_initial);
        } else {
            solve_adjoint_full(*object.ROM, Keff, rhs, lambda, analysis, Iter, k, &Keff_eigen_cache);
        }

        lambda_all.col(k) = gather_vec_to_all_local(lambda);
        if (k == nstep - 1 || k == nstep / 2 || k == 0) {
            MemoryDebug::print("petsc_sens_after_step_" + std::to_string(k));
        }
    }

    MemoryDebug::print("petsc_sens_before_explicit_gradient");
    Eigen::MatrixXd dfdx_phys = explicit_gradient_dynamic(fem, analysis, filtering_state.xPhys, object, lambda_all);
    MemoryDebug::print("petsc_sens_after_explicit_gradient");

    VecDestroy(&lambda);
    VecDestroy(&lambda_new);
    VecDestroy(&mu);
    VecDestroy(&mu_new);
    VecDestroy(&eta);
    VecDestroy(&eta_new);
    VecDestroy(&rhs);
    VecDestroy(&Mlambda);
    MemoryDebug::print("petsc_sens_after_destroy_temp");

    return filtering.sensitivity(dfdx_phys, filtering_state);
}

} // namespace

Eigen::MatrixXd PetscDynamicLinearElasticAVM::AVM(const FEM& fem,
                                                  Analysis& analysis,
                                                  const Eigen::MatrixXd&,
                                                  const std::vector<int>& x_label,
                                                  const Filtering& filtering,
                                                  const FilteringState& filtering_state,
                                                  int Iter) {
    Eigen::MatrixXd out = dynamic_elastic_base_full_impl(fem, analysis, *this, filtering, filtering_state, Iter);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}
