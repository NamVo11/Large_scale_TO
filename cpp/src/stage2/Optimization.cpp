#include "stage2/Optimization.hpp"

#include "stage2/DynamicElasticLinear.hpp"
#include "stage2/FEMAssembly.hpp"
#include "stage2/ROM.hpp"
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

namespace stage2
{
    namespace
    {

        std::string lower(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        int nvar_of(const std::vector<std::vector<double>> &x)
        {
            return x.empty() ? 0 : static_cast<int>(x.front().size());
        }

        std::vector<double> flatten(const std::vector<std::vector<double>> &x)
        {
            const int nvar = nvar_of(x);
            std::vector<double> out;
            out.reserve(x.size() * static_cast<std::size_t>(nvar));
            for (int j = 0; j < nvar; ++j)
            {
                for (const auto &row : x)
                    out.push_back(row[static_cast<std::size_t>(j)]);
            }
            return out;
        }

        std::vector<std::vector<double>> unflatten(const std::vector<double> &x, int nelem, int nvar)
        {
            std::vector<std::vector<double>> out(static_cast<std::size_t>(nelem), std::vector<double>(static_cast<std::size_t>(nvar), 0.0));
            for (int j = 0; j < nvar; ++j)
            {
                for (int e = 0; e < nelem; ++e)
                    out[static_cast<std::size_t>(e)][static_cast<std::size_t>(j)] = x[static_cast<std::size_t>(j * nelem + e)];
            }
            return out;
        }

        std::vector<std::vector<double>> zeros(int nrow, int ncol)
        {
            return std::vector<std::vector<double>>(static_cast<std::size_t>(nrow), std::vector<double>(static_cast<std::size_t>(ncol), 0.0));
        }

        void add_scaled(std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b, double scale)
        {
            if (a.empty())
            {
                a = b;
                for (auto &row : a)
                {
                    for (double &v : row)
                        v *= scale;
                }
                return;
            }
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                for (std::size_t j = 0; j < a[i].size(); ++j)
                    a[i][j] += scale * b[i][j];
            }
        }

        double dot(const std::vector<double> &a, const std::vector<double> &b)
        {
            double s = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i)
                s += a[i] * b[i];
            return s;
        }

        double sech2(double x)
        {
            const double c = std::cosh(x);
            return 1.0 / (c * c);
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

        std::vector<double> gather_vec_to_all_local(Vec v)
        {
            PetscMPIInt size = 1;
            MPI_Comm_size(PetscObjectComm((PetscObject)v), &size);
            if (size == 1)
            {
                PetscInt n = 0;
                VecGetLocalSize(v, &n);
                std::vector<double> out(static_cast<std::size_t>(n));
                const PetscScalar *a = nullptr;
                VecGetArrayRead(v, &a);
                for (PetscInt i = 0; i < n; ++i)
                    out[static_cast<std::size_t>(i)] = PetscRealPart(a[i]);
                VecRestoreArrayRead(v, &a);
                return out;
            }

            Vec vall = nullptr;
            VecScatter scatter = nullptr;
            VecScatterCreateToAll(v, &scatter, &vall);
            VecScatterBegin(scatter, v, vall, INSERT_VALUES, SCATTER_FORWARD);
            VecScatterEnd(scatter, v, vall, INSERT_VALUES, SCATTER_FORWARD);
            std::vector<double> out;
            PetscInt n = 0;
            VecGetSize(vall, &n);
            out.resize(static_cast<std::size_t>(n));
            const PetscScalar *a = nullptr;
            VecGetArrayRead(vall, &a);
            for (PetscInt i = 0; i < n; ++i)
                out[static_cast<std::size_t>(i)] = PetscRealPart(a[i]);
            VecRestoreArrayRead(vall, &a);
            VecScatterDestroy(&scatter);
            VecDestroy(&vall);
            return out;
        }

        void set_vec_from_column_root(Vec v, const std::vector<std::vector<double>> &values, int k, double scale = 1.0)
        {
            PetscMPIInt size = 1;
            MPI_Comm_size(PetscObjectComm((PetscObject)v), &size);
            if (size == 1)
            {
                if (values.empty())
                {
                    VecSet(v, 0.0);
                    return;
                }
                PetscInt n = 0;
                VecGetLocalSize(v, &n);
                if (static_cast<std::size_t>(n) != values.size())
                    throw std::runtime_error("set_vec_from_column_root: vector size mismatch");
                PetscScalar *a = nullptr;
                VecGetArray(v, &a);
                for (PetscInt i = 0; i < n; ++i)
                    a[i] = scale * values[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
                VecRestoreArray(v, &a);
                return;
            }

            PetscMPIInt rank = 0;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            VecSet(v, 0.0);
            if (rank == 0)
            {
                std::vector<PetscInt> ids(values.size());
                std::vector<PetscScalar> vals(values.size());
                std::iota(ids.begin(), ids.end(), 0);
                for (std::size_t i = 0; i < values.size(); ++i)
                    vals[i] = scale * values[i][static_cast<std::size_t>(k)];
                VecSetValues(v, static_cast<PetscInt>(ids.size()), ids.data(), vals.data(), INSERT_VALUES);
            }
            VecAssemblyBegin(v);
            VecAssemblyEnd(v);
        }

        void set_vec_fixed_from_column_root(Vec v, const std::vector<int> &fixed,
                                            const std::vector<std::vector<double>> &values, int k, double scale = 1.0)
        {
            if (fixed.empty())
                return;
            PetscMPIInt size = 1;
            MPI_Comm_size(PetscObjectComm((PetscObject)v), &size);
            if (size == 1)
            {
                if (values.empty())
                    return;
                PetscInt n = 0;
                VecGetLocalSize(v, &n);
                PetscScalar *a = nullptr;
                VecGetArray(v, &a);
                for (int id : fixed)
                {
                    if (id < 0 || id >= n)
                        throw std::runtime_error("set_vec_fixed_from_column_root: fixed dof out of range");
                    a[id] = scale * values[static_cast<std::size_t>(id)][static_cast<std::size_t>(k)];
                }
                VecRestoreArray(v, &a);
                return;
            }

            PetscMPIInt rank = 0;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            if (rank == 0)
            {
                std::vector<PetscInt> ids(fixed.begin(), fixed.end());
                std::vector<PetscScalar> vals(ids.size());
                for (std::size_t i = 0; i < ids.size(); ++i)
                    vals[i] = scale * values[static_cast<std::size_t>(ids[i])][static_cast<std::size_t>(k)];
                VecSetValues(v, static_cast<PetscInt>(ids.size()), ids.data(), vals.data(), INSERT_VALUES);
            }
            VecAssemblyBegin(v);
            VecAssemblyEnd(v);
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

        IS make_is(const std::vector<int> &ids)
        {
            std::vector<PetscInt> p(ids.begin(), ids.end());
            IS is = nullptr;
            ISCreateGeneral(PETSC_COMM_WORLD, static_cast<PetscInt>(p.size()), p.data(), PETSC_COPY_VALUES, &is);
            return is;
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
            rom.main(lhs, rhs, lambda, ctx);
        }

        Mat duplicate_with_dirichlet(Mat A, const std::vector<int> &fixed)
        {
            Mat B = nullptr;
            MatDuplicate(A, MAT_COPY_VALUES, &B);
            if (!fixed.empty())
            {
                IS is = make_is(fixed);
                MatZeroRowsColumnsIS(B, is, 1.0, nullptr, nullptr);
                ISDestroy(&is);
            }
            return B;
        }

        void L2_compliance_elastic(const FEM &fem, const Analysis &analysis, const std::vector<std::vector<double>> &xPhys, ObjectiveTerm &term)
        {
            const int ndof = fem.ndof;
            const int nstep = analysis.Nstep;
            const int nvar = nvar_of(xPhys);
            term.dfdu_all = zeros(ndof, nstep + 1);
            term.dfdv_all = zeros(ndof, nstep + 1);
            term.dfda_all = zeros(ndof, nstep + 1);
            term.dfdFext_all = zeros(ndof, nstep + 1);
            term.df_dx = zeros(fem.NElem, nvar);
            term.value = 0.0;

            for (int k = 0; k <= nstep; ++k)
            {
                const auto &u = analysis.results.U[static_cast<std::size_t>(k)];
                const auto &f = analysis.results.F_ext[static_cast<std::size_t>(k)];
                const double s = dot(f, u);
                term.value += s * s;
                for (int i = 0; i < ndof; ++i)
                {
                    term.dfdu_all[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] = 2.0 * s * f[static_cast<std::size_t>(i)];
                    term.dfdFext_all[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] = 2.0 * s * u[static_cast<std::size_t>(i)];
                }
                for (int d : analysis.results.SPCdofs)
                {
                    term.dfdu_all[static_cast<std::size_t>(d)][static_cast<std::size_t>(k)] = 0.0;
                    term.dfdv_all[static_cast<std::size_t>(d)][static_cast<std::size_t>(k)] = 0.0;
                    term.dfda_all[static_cast<std::size_t>(d)][static_cast<std::size_t>(k)] = 0.0;
                }
                for (int d : analysis.results.free)
                    term.dfdFext_all[static_cast<std::size_t>(d)][static_cast<std::size_t>(k)] = 0.0;
            }
        }

        void volume_fraction(const FEM &fem, const Analysis &analysis, const std::vector<std::vector<double>> &xPhys, Constraint &c)
        {
            const int ndof = fem.ndof;
            const int nstep = analysis.Nstep;
            const int nvar = nvar_of(xPhys);
            const int val_id = c.mat_id - 2;
            if (val_id < 0 || val_id >= nvar)
                throw std::runtime_error("volume_fraction: invalid mat_id");
            double mean = 0.0;
            for (const auto &row : xPhys)
                mean += row[static_cast<std::size_t>(val_id)];
            mean /= static_cast<double>(fem.NElem);
            c.value = mean;
            c.d_explicit_dfdu_all = zeros(ndof, nstep + 1);
            c.d_explicit_dfdv_all = zeros(ndof, nstep + 1);
            c.d_explicit_dfda_all = zeros(ndof, nstep + 1);
            c.d_explicit_dfdFext_all = zeros(ndof, nstep + 1);
            c.d_explicit_df_dx = zeros(fem.NElem, nvar);
            for (int e = 0; e < fem.NElem; ++e)
                c.d_explicit_df_dx[static_cast<std::size_t>(e)][static_cast<std::size_t>(val_id)] = 1.0 / static_cast<double>(fem.NElem);
        }

        /*
        std::vector<std::vector<double>> explicit_gradient_dynamic(const FEM &fem,
                                                                   const Analysis &analysis,
                                                                   const std::vector<std::vector<double>> &xPhys,
                                                                   const ObjectiveGroup &object,
                                                                   const std::vector<std::vector<double>> &lambda_all)
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

            auto t0 = Clock::now();
            std::vector<std::vector<double>> dE(static_cast<std::size_t>(fem.NElem),
                                                std::vector<double>(static_cast<std::size_t>(nvar), 0.0));
            std::vector<std::vector<double>> drho(static_cast<std::size_t>(fem.NElem),
                                                  std::vector<double>(static_cast<std::size_t>(nvar), 0.0));
            for (int e = elem_start; e < elem_end; ++e)
            {
                const Region &region = analysis.regions.front();
                for (int v = 0; v < nvar; ++v)
                {
                    const double x = xPhys[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)];
                    dE[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] =
                        ramp_derivative(region.mat_paras.E_inf, x, region.interp.p_E_elastic);
                    drho[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] =
                        ramp_derivative(region.mat_paras.rho, x, region.interp.p_rho);
                }
            }
            auto t1 = Clock::now();
            const double timing_interp_derivative = std::chrono::duration<double>(t1 - t0).count();

            const Region &region = analysis.regions.front();
            const double nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
            t0 = Clock::now();
            const ElementMatrix em = compute_element_matrices(fem, 0, nu, fem.Thickness);
            t1 = Clock::now();
            const double timing_element_matrix = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            std::vector<std::vector<double>> df_dx_local = zeros(fem.NElem, nvar);
            t1 = Clock::now();
            const double timing_alloc_gradient = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            std::vector<int> dofs;
            for (int k = 0; k <= nstep; ++k)
            {
                const auto &u = analysis.results.U[static_cast<std::size_t>(k)];
                const auto &acc = analysis.results.A[static_cast<std::size_t>(k)];
                const auto &lambda = lambda_all[static_cast<std::size_t>(k)];
                for (int e = elem_start; e < elem_end; ++e)
                {
                    element_dofs(fem, e, dofs);
                    std::vector<double> row_K(static_cast<std::size_t>(em.ndof), 0.0);
                    std::vector<double> row_M(static_cast<std::size_t>(em.ndof), 0.0);
                    for (int i = 0; i < em.ndof; ++i)
                    {
                        for (int j = 0; j < em.ndof; ++j)
                        {
                            const int dof_j = dofs[static_cast<std::size_t>(j)];
                            row_K[static_cast<std::size_t>(i)] +=
                                em.ke[static_cast<std::size_t>(i * em.ndof + j)] * u[static_cast<std::size_t>(dof_j)];
                            if (k != 0)
                            {
                                row_M[static_cast<std::size_t>(i)] +=
                                    em.me[static_cast<std::size_t>(i * em.ndof + j)] * acc[static_cast<std::size_t>(dof_j)];
                            }
                        }
                    }
                    for (int v = 0; v < nvar; ++v)
                    {
                        double val = 0.0;
                        for (int i = 0; i < em.ndof; ++i)
                        {
                            const int dof_i = dofs[static_cast<std::size_t>(i)];
                            const double derivative_force =
                                row_K[static_cast<std::size_t>(i)] * dE[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] +
                                row_M[static_cast<std::size_t>(i)] * drho[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)];
                            val += lambda[static_cast<std::size_t>(dof_i)] * derivative_force;
                        }
                        df_dx_local[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] += val;
                    }
                }
            }
            t1 = Clock::now();
            const double timing_element_loop = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            std::vector<double> local_flat = flatten(df_dx_local);
            std::vector<double> global_flat(local_flat.size(), 0.0);
            t1 = Clock::now();
            const double timing_flatten = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            MPI_Allreduce(local_flat.data(), global_flat.data(), static_cast<int>(local_flat.size()), MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
            t1 = Clock::now();
            const double timing_allreduce = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            std::vector<std::vector<double>> df_dx = unflatten(global_flat, fem.NElem, nvar);
            add_scaled(df_dx, object.d_explicit_df_dx, 1.0);
            t1 = Clock::now();
            const double timing_unflatten_add = std::chrono::duration<double>(t1 - t0).count();
            if (rank == 0)
            {
                const double timing_total = std::chrono::duration<double>(Clock::now() - timer_begin).count();
                std::cout << "  explicit_gradient_dynamic[s]:"
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
        */

        std::vector<std::vector<double>> explicit_gradient_dynamic(const FEM &fem,
                                                                   const Analysis &analysis,
                                                                   const std::vector<std::vector<double>> &xPhys,
                                                                   const ObjectiveGroup &object,
                                                                   const std::vector<std::vector<double>> &lambda_all)
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
                    const double x = xPhys[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)];
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
                uniform_em = compute_element_matrices(fem, analysis.regions.front().elem_ids.front(), nu, fem.Thickness);
            t1 = Clock::now();
            const double timing_element_matrix = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            std::vector<double> df_dx_local(static_cast<std::size_t>(fem.NElem * nvar), 0.0);
            t1 = Clock::now();
            const double timing_alloc_gradient = std::chrono::duration<double>(t1 - t0).count();

            t0 = Clock::now();
            #pragma omp parallel for schedule(static) num_threads(sens_element_threads) if(sens_element_threads > 1)
            for (int e = elem_start; e < elem_end; ++e)
            {
                ElementMatrix local_em;
                const ElementMatrix *em_ptr = &uniform_em;
                if (!fem.uniform_element_matrices)
                {
                    local_em = compute_element_matrices(fem, e, nu, fem.Thickness);
                    em_ptr = &local_em;
                }
                const ElementMatrix &em = *em_ptr;
                if (em.ndof > 24)
                    throw std::runtime_error("explicit_gradient_dynamic: element ndof > 24 is not supported by stack buffer");

                const auto &dofs = fem.ElementDofs.at(static_cast<std::size_t>(e));
                std::array<double, 24> row_K{};
                std::array<double, 24> row_M{};

                for (int k = 0; k <= nstep; ++k)
                {
                    row_K.fill(0.0);
                    row_M.fill(0.0);
                    const auto &u = analysis.results.U[static_cast<std::size_t>(k)];
                    const auto &acc = analysis.results.A[static_cast<std::size_t>(k)];
                    const auto &lambda = lambda_all[static_cast<std::size_t>(k)];

                    for (int i = 0; i < em.ndof; ++i)
                    {
                        const std::size_t row = static_cast<std::size_t>(i * em.ndof);
                        for (int j = 0; j < em.ndof; ++j)
                        {
                            const int dof_j = static_cast<int>(dofs[static_cast<std::size_t>(j)]);
                            row_K[static_cast<std::size_t>(i)] +=
                                em.ke[row + static_cast<std::size_t>(j)] * u[static_cast<std::size_t>(dof_j)];
                            if (k != 0)
                            {
                                row_M[static_cast<std::size_t>(i)] +=
                                    em.me[row + static_cast<std::size_t>(j)] * acc[static_cast<std::size_t>(dof_j)];
                            }
                        }
                    }

                    for (int v = 0; v < nvar; ++v)
                    {
                        double val = 0.0;
                        const std::size_t ev = static_cast<std::size_t>(e * nvar + v);
                        const double dE_ev = dE[ev];
                        const double drho_ev = drho[ev];
                        for (int i = 0; i < em.ndof; ++i)
                        {
                            const int dof_i = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                            const double derivative_force =
                                row_K[static_cast<std::size_t>(i)] * dE_ev +
                                row_M[static_cast<std::size_t>(i)] * drho_ev;
                            val += lambda[static_cast<std::size_t>(dof_i)] * derivative_force;
                        }
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
            std::vector<std::vector<double>> df_dx = unflatten(global_flat, fem.NElem, nvar);
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

        std::vector<std::vector<double>> dynamic_elastic_base_full_impl(const FEM &fem,
                                                                        Analysis &analysis,
                                                                        const std::vector<std::vector<double>> &xPhys,
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
                object.ROM = make_fom_solver("direct");
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

            std::vector<std::vector<double>> lambda_all(static_cast<std::size_t>(nstep + 1));

            object.ROM->set_factorized_solver(Keff, analysis.results.factorized_K_eff);

            int k = nstep;
            set_vec_from_column_root(mu, object.d_explicit_dfda_all, k);
            set_vec_from_column_root(eta, object.d_explicit_dfdv_all, k);
            set_vec_from_column_root(rhs, object.d_explicit_dfdu_all, k, -1.0);
            VecAXPY(rhs, -dHdu, mu);
            VecAXPY(rhs, -dLdu, eta);
            set_vec_fixed_from_column_root(rhs, analysis.results.SPCdofs, object.d_explicit_dfdFext_all, k, -1.0);
            solve_adjoint_full(*object.ROM, Keff, rhs, lambda, analysis, Iter, k);
            lambda_all[static_cast<std::size_t>(k)] = gather_vec_to_all_local(lambda);

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
                    object.ROM->set_factorized_solver(K0, analysis.results.factorized_K_initial);
                }
                solve_adjoint_full(*object.ROM, k == 0 ? K0 : Keff, rhs, lambda, analysis, Iter, k);
                lambda_all[static_cast<std::size_t>(k)] = gather_vec_to_all_local(lambda);
            }

            std::vector<std::vector<double>> dfdx_phys =
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

        void save_history(const Optimization &Opt, const OutputData &Output)
        {
            PetscMPIInt rank = 0;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            if (rank != 0)
                return;
            std::filesystem::create_directories(Output.outDir_mat);
            std::ofstream out(Output.outDir_mat + "/optimization_history.csv");
            out << "Iter,objective,constraint,change,rms_change,forward_time,objective_time,constraint_time,objective_sensitivity_time,constraint_sensitivity_time,mma_time,iteration_time\n";
            for (std::size_t i = 0; i < Opt.history.Iter.size(); ++i)
            {
                out << Opt.history.Iter[i] << ","
                    << Opt.history.objective[i] << ","
                    << Opt.history.constraint[i] << ","
                    << Opt.history.change[i] << ","
                    << Opt.history.rms_change[i] << ","
                    << Opt.history.forward_time[i] << ","
                    << Opt.history.objective_time[i] << ","
                    << Opt.history.constraint_time[i] << ","
                    << Opt.history.objective_sensitivity_time[i] << ","
                    << Opt.history.constraint_sensitivity_time[i] << ","
                    << Opt.history.mma_time[i] << ","
                    << Opt.history.iteration_time[i] << "\n";
            }
        }

        void write_rom_matrix_csv(const std::string &path,
                                  const std::vector<std::vector<double>> &values)
        {
            std::ofstream out(path);
            if (!out)
                throw std::runtime_error("Cannot write " + path);
            out.precision(17);
            for (const auto &row : values)
            {
                for (std::size_t j = 0; j < row.size(); ++j)
                {
                    if (j)
                        out << ",";
                    out << row[j];
                }
                out << "\n";
            }
        }

        std::vector<std::vector<double>> rom_info_matrix(const std::vector<std::vector<ROMInfo>> &info,
                                                         const std::string &field)
        {
            std::vector<std::vector<double>> values(info.size());
            for (std::size_t i = 0; i < info.size(); ++i)
            {
                values[i].resize(info[i].size(), std::numeric_limits<double>::quiet_NaN());
                for (std::size_t j = 0; j < info[i].size(); ++j)
                {
                    const ROMInfo &r = info[i][j];
                    if (field == "rom_used")
                        values[i][j] = r.used_rom ? 1.0 : 0.0;
                    else if (field == "fom_used")
                        values[i][j] = r.used_rom ? 0.0 : 1.0;
                    else if (field == "accepted")
                        values[i][j] = r.accepted ? 1.0 : 0.0;
                    else if (field == "residual")
                        values[i][j] = r.residual_norm;
                    else if (field == "ksp_iterations")
                        values[i][j] = static_cast<double>(r.ksp_iterations);
                    else if (field == "rank")
                        values[i][j] = static_cast<double>(r.rank);
                    else if (field == "threshold")
                        values[i][j] = r.threshold;
                    else if (field == "error")
                        values[i][j] = r.error;
                    else if (field == "basis_error")
                        values[i][j] = r.basis_error;
                    else if (field == "timing_total")
                        values[i][j] = r.timing_total;
                    else if (field == "timing_petsc_to_eigen_vec")
                        values[i][j] = r.timing_petsc_to_eigen_vec;
                    else if (field == "timing_petsc_to_eigen_mat")
                        values[i][j] = r.timing_petsc_to_eigen_mat;
                    else if (field == "timing_projection")
                        values[i][j] = r.timing_projection;
                    else if (field == "timing_reduced_rhs")
                        values[i][j] = r.timing_reduced_rhs;
                    else if (field == "timing_reduced_solve")
                        values[i][j] = r.timing_reduced_solve;
                    else if (field == "timing_reconstruct")
                        values[i][j] = r.timing_reconstruct;
                    else if (field == "timing_residual")
                        values[i][j] = r.timing_residual;
                    else if (field == "timing_fom_fallback")
                        values[i][j] = r.timing_fom_fallback;
                    else if (field == "timing_fom_factorization")
                        values[i][j] = r.timing_fom_factorization;
                    else if (field == "timing_fom_solve")
                        values[i][j] = r.timing_fom_solve;
                    else if (field == "timing_basis_update")
                        values[i][j] = r.timing_basis_update;
                    else if (field == "timing_basis1_update")
                        values[i][j] = r.timing_basis1_update;
                    else if (field == "timing_online_basis_update")
                        values[i][j] = r.timing_online_basis_update;
                    else if (field == "timing_end_basis_update")
                        values[i][j] = r.timing_end_basis_update;
                    else if (field == "timing_basis_copy_reset")
                        values[i][j] = r.timing_basis_copy_reset;
                    else if (field == "timing_snapshot")
                        values[i][j] = r.timing_snapshot;
                }
            }
            return values;
        }

        void save_rom_info_group(const std::string &out_dir,
                                 const std::string &prefix,
                                 const ROMBase *rom)
        {
            if (!rom)
                return;
            std::vector<std::string> fields = {
                "fom_used", "accepted", "residual", "ksp_iterations",
                "timing_total", "timing_fom_factorization", "timing_fom_solve"};
            if (rom->use_mor)
            {
                fields = {
                    "rom_used", "fom_used", "accepted", "residual",
                    "ksp_iterations", "rank", "threshold", "error", "basis_error",
                    "timing_total", "timing_petsc_to_eigen_vec", "timing_petsc_to_eigen_mat",
                    "timing_projection", "timing_reduced_rhs", "timing_reduced_solve",
                    "timing_reconstruct", "timing_residual", "timing_fom_fallback",
                    "timing_fom_factorization", "timing_fom_solve",
                    "timing_basis_update", "timing_basis1_update", "timing_online_basis_update",
                    "timing_end_basis_update", "timing_basis_copy_reset",
                    "timing_snapshot"};
            }
            for (const std::string &field : fields)
                write_rom_matrix_csv(out_dir + "/" + prefix + "_" + field + ".csv",
                                     rom_info_matrix(rom->ROM_info, field));
        }

        void save_rom_histories(const std::vector<Analysis> &analysis_all,
                                const Optimization &Opt,
                                const OutputData &Output)
        {
            PetscMPIInt rank = 0;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            if (rank != 0)
                return;
            std::filesystem::create_directories(Output.outDir_mat);
            if (!analysis_all.empty() && analysis_all.front().ROM)
                save_rom_info_group(Output.outDir_mat, "forward", analysis_all.front().ROM.get());
            if (!Opt.objective_groups.empty() && Opt.objective_groups.front().ROM)
                save_rom_info_group(Output.outDir_mat, "obj_sens", Opt.objective_groups.front().ROM.get());
        }

        std::vector<double> matlab_like_density_display(const FEM &fem,
                                                        const std::vector<std::vector<double>> &xPhys)
        {
            if (fem.dim != 2 || xPhys.empty())
                return {};

            const int mTopo = static_cast<int>(xPhys.front().size());
            std::vector<double> x_display(static_cast<std::size_t>(fem.nelx * fem.nely), 0.0);
            for (int i = 0; i < fem.nelx; ++i)
            {
                for (int j = 0; j < fem.nely; ++j)
                {
                    const int e = i * fem.nely + j;
                    const auto &X = xPhys[static_cast<std::size_t>(e)];
                    std::vector<double> Qs(static_cast<std::size_t>(mTopo), 0.0);
                    for (int v = 0; v < mTopo; ++v)
                        Qs[static_cast<std::size_t>(v)] = std::max(1e-14, 1.0 - X[static_cast<std::size_t>(v)]);

                    double prod_all = 1.0;
                    for (double q : Qs)
                        prod_all *= q;
                    double multi_topo = 0.0;
                    for (int v = 0; v < mTopo; ++v)
                    {
                        const double prod_except = prod_all / Qs[static_cast<std::size_t>(v)];
                        multi_topo += X[static_cast<std::size_t>(v)] * prod_except * static_cast<double>(v + 1);
                    }
                    x_display[static_cast<std::size_t>(j * fem.nelx + i)] = multi_topo;
                }
            }
            return x_display;
        }

        void save_density_iteration(const FEM &fem,
                                    const std::vector<std::vector<double>> &xPhys,
                                    const OutputData &output,
                                    int Iter)
        {
            PetscMPIInt rank = 0;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            if (rank != 0 || xPhys.empty())
                return;

            const std::filesystem::path dir = output.densityDir.empty()
                                                  ? std::filesystem::path(output.outDir_mat) / "density"
                                                  : std::filesystem::path(output.densityDir);
            std::filesystem::create_directories(dir);

            char name[64] = {0};
            std::snprintf(name, sizeof(name), "density_iter_%04d.csv", Iter);
            std::ofstream csv(dir / name);
            if (!csv)
                throw std::runtime_error("Cannot write density csv");
            csv.precision(17);
            csv << "elem,i,j,k,xPhys\n";
            for (int e = 0; e < fem.NElem; ++e)
            {
                int i = 0;
                int j = 0;
                int k = 0;
                if (fem.dim == 2)
                {
                    i = e / fem.nely;
                    j = e % fem.nely;
                }
                else
                {
                    const int layer = fem.nelx * fem.nely;
                    k = e / layer;
                    const int rem = e % layer;
                    i = rem / fem.nely;
                    j = rem % fem.nely;
                }
                csv << e << "," << i << "," << j << "," << k << "," << xPhys[static_cast<std::size_t>(e)].front() << "\n";
            }

            if (fem.dim != 2)
                return;
            const std::vector<double> x_display = matlab_like_density_display(fem, xPhys);
            std::snprintf(name, sizeof(name), "density_iter_%04d.pgm", Iter);
            std::ofstream pgm(dir / name, std::ios::binary);
            if (!pgm)
                throw std::runtime_error("Cannot write density pgm");
            pgm << "P5\n" << fem.nelx << " " << fem.nely << "\n255\n";
            const double max_val = std::max(1e-14, *std::max_element(x_display.begin(), x_display.end()));
            for (int j = fem.nely - 1; j >= 0; --j)
            {
                for (int i = 0; i < fem.nelx; ++i)
                {
                    const double val = x_display[static_cast<std::size_t>(j * fem.nelx + i)] / max_val;
                    const unsigned char pixel = static_cast<unsigned char>(std::lround(255.0 * std::clamp(val, 0.0, 1.0)));
                    pgm.write(reinterpret_cast<const char *>(&pixel), 1);
                }
            }
        }

        double objective_group_value_for_debug(const FEM &fem,
                                               const Analysis &analysis,
                                               const std::vector<std::vector<double>> &xPhys,
                                               const ObjectiveGroup &object)
        {
            PetscMPIInt rank = 0;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            if (rank != 0)
                return 0.0;

            double value = 0.0;
            for (const auto &source_term : object.terms)
            {
                if (lower(source_term.function_name) != "l2_compliance_elastic")
                    throw std::runtime_error("Unknown objective term");
                ObjectiveTerm term = source_term;
                L2_compliance_elastic(fem, analysis, xPhys, term);
                const double value_initial = source_term.value_initial != 0.0 ? source_term.value_initial : term.value;
                value += term.weight * term.value / value_initial;
            }
            return value;
        }

    } // namespace

    void filtering_initilization(const FEM &fem, Filtering &filtering, FilteringState &state)
    {
        const int r = std::max(0, static_cast<int>(std::ceil(filtering.rmin)) - 1);
        state.neighbors.assign(static_cast<std::size_t>(fem.NElem), {});
        state.weights.assign(static_cast<std::size_t>(fem.NElem), {});
        state.Hs.assign(static_cast<std::size_t>(fem.NElem), 0.0);
        auto eid2 = [&fem](int i, int j)
        { return i * fem.nely + j; };
        auto eid3 = [&fem](int i, int j, int k)
        { return k * fem.nelx * fem.nely + i * fem.nely + j; };
        if (fem.dim == 2)
        {
            for (int i1 = 0; i1 < fem.nelx; ++i1)
            {
                for (int j1 = 0; j1 < fem.nely; ++j1)
                {
                    const int e1 = eid2(i1, j1);
                    for (int i2 = std::max(0, i1 - r); i2 <= std::min(fem.nelx - 1, i1 + r); ++i2)
                    {
                        for (int j2 = std::max(0, j1 - r); j2 <= std::min(fem.nely - 1, j1 + r); ++j2)
                        {
                            const double dist = std::sqrt((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2));
                            const double w = filtering.rmin - dist;
                            if (w > 0.0)
                            {
                                state.neighbors[static_cast<std::size_t>(e1)].push_back(eid2(i2, j2));
                                state.weights[static_cast<std::size_t>(e1)].push_back(w);
                                state.Hs[static_cast<std::size_t>(e1)] += w;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for (int k1 = 0; k1 < fem.nelz; ++k1)
            {
                for (int i1 = 0; i1 < fem.nelx; ++i1)
                {
                    for (int j1 = 0; j1 < fem.nely; ++j1)
                    {
                        const int e1 = eid3(i1, j1, k1);
                        for (int k2 = std::max(0, k1 - r); k2 <= std::min(fem.nelz - 1, k1 + r); ++k2)
                        {
                            for (int i2 = std::max(0, i1 - r); i2 <= std::min(fem.nelx - 1, i1 + r); ++i2)
                            {
                                for (int j2 = std::max(0, j1 - r); j2 <= std::min(fem.nely - 1, j1 + r); ++j2)
                                {
                                    const double dist = std::sqrt((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2) + (k1 - k2) * (k1 - k2));
                                    const double w = filtering.rmin - dist;
                                    if (w > 0.0)
                                    {
                                        state.neighbors[static_cast<std::size_t>(e1)].push_back(eid3(i2, j2, k2));
                                        state.weights[static_cast<std::size_t>(e1)].push_back(w);
                                        state.Hs[static_cast<std::size_t>(e1)] += w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void filtering_calculation(const std::vector<std::vector<double>> &x,
                               const std::vector<int> &x_label,
                               const Filtering &filtering,
                               FilteringState &state)
    {
        const int nelem = static_cast<int>(x.size());
        const int nvar = nvar_of(x);
        state.xTilde = zeros(nelem, nvar);
        state.xPhys = zeros(nelem, nvar);
        const double denom = std::tanh(filtering.beta * filtering.T) + std::tanh(filtering.beta * (1.0 - filtering.T));
        for (int e = 0; e < nelem; ++e)
        {
            for (std::size_t p = 0; p < state.neighbors[static_cast<std::size_t>(e)].size(); ++p)
            {
                const int j = state.neighbors[static_cast<std::size_t>(e)][p];
                const double w = state.weights[static_cast<std::size_t>(e)][p];
                for (int v = 0; v < nvar; ++v)
                    state.xTilde[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] += w * x[static_cast<std::size_t>(j)][static_cast<std::size_t>(v)];
            }
            for (int v = 0; v < nvar; ++v)
            {
                state.xTilde[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] /= state.Hs[static_cast<std::size_t>(e)];
                state.xPhys[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] =
                    (std::tanh(filtering.beta * filtering.T) + std::tanh(filtering.beta * (state.xTilde[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] - filtering.T))) / denom;
            }
            if (x_label[static_cast<std::size_t>(e)] > 0)
                std::fill(state.xPhys[static_cast<std::size_t>(e)].begin(), state.xPhys[static_cast<std::size_t>(e)].end(), 1.0);
            if (x_label[static_cast<std::size_t>(e)] < 0)
                std::fill(state.xPhys[static_cast<std::size_t>(e)].begin(), state.xPhys[static_cast<std::size_t>(e)].end(), 0.0);
        }
    }

    std::vector<std::vector<double>> filtering_sensitivity(const std::vector<std::vector<double>> &df_dx_phys,
                                                           const Filtering &filtering,
                                                           const FilteringState &state)
    {
        const int nelem = static_cast<int>(df_dx_phys.size());
        const int nvar = nvar_of(df_dx_phys);
        std::vector<std::vector<double>> tmp = zeros(nelem, nvar);
        const double denom = std::tanh(filtering.beta * filtering.T) + std::tanh(filtering.beta * (1.0 - filtering.T));
        for (int e = 0; e < nelem; ++e)
        {
            for (int v = 0; v < nvar; ++v)
            {
                const double dproj = filtering.beta * sech2(filtering.beta * (state.xTilde[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] - filtering.T)) / denom;
                tmp[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] = df_dx_phys[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] * dproj / state.Hs[static_cast<std::size_t>(e)];
            }
        }
        std::vector<std::vector<double>> out = zeros(nelem, nvar);
        for (int e = 0; e < nelem; ++e)
        {
            for (std::size_t p = 0; p < state.neighbors[static_cast<std::size_t>(e)].size(); ++p)
            {
                const int j = state.neighbors[static_cast<std::size_t>(e)][p];
                const double w = state.weights[static_cast<std::size_t>(e)][p];
                for (int v = 0; v < nvar; ++v)
                    out[static_cast<std::size_t>(j)][static_cast<std::size_t>(v)] += w * tmp[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)];
            }
        }
        return out;
    }

    void evaluate_objective(const FEM &fem,
                            std::vector<Analysis> &analysis_all,
                            const std::vector<std::vector<double>> &xPhys,
                            std::vector<ObjectiveGroup> &objective_groups,
                            double &obj_value)
    {
        obj_value = 0.0;
        for (auto &group : objective_groups)
        {
            Analysis &analysis = analysis_all[static_cast<std::size_t>(group.analysis_id - 1)];
            group.value = 0.0;
            group.d_explicit_dfdu_all.clear();
            group.d_explicit_dfdv_all.clear();
            group.d_explicit_dfda_all.clear();
            group.d_explicit_dfdFext_all.clear();
            group.d_explicit_df_dx.clear();
            for (auto &term : group.terms)
            {
                if (lower(term.function_name) != "l2_compliance_elastic")
                    throw std::runtime_error("Unknown objective term");
                L2_compliance_elastic(fem, analysis, xPhys, term);
                if (term.value_initial == 0.0)
                    term.value_initial = term.value;
                const double alpha = term.weight / term.value_initial;
                group.value += alpha * term.value;
                add_scaled(group.d_explicit_dfdu_all, term.dfdu_all, alpha);
                add_scaled(group.d_explicit_dfdv_all, term.dfdv_all, alpha);
                add_scaled(group.d_explicit_dfda_all, term.dfda_all, alpha);
                add_scaled(group.d_explicit_dfdFext_all, term.dfdFext_all, alpha);
                add_scaled(group.d_explicit_df_dx, term.df_dx, alpha);
            }
            obj_value += group.value;
        }
    }

    void evaluate_constraint(const FEM &fem,
                             std::vector<Analysis> &analysis_all,
                             const std::vector<std::vector<double>> &xPhys,
                             std::vector<Constraint> &constraints,
                             std::vector<double> &constr_value_vec)
    {
        constr_value_vec.assign(constraints.size(), 0.0);
        for (std::size_t i = 0; i < constraints.size(); ++i)
        {
            Constraint &c = constraints[i];
            Analysis &analysis = analysis_all[static_cast<std::size_t>(c.analysis_id - 1)];
            if (lower(c.function_name) != "volume_fraction")
                throw std::runtime_error("Unknown constraint");
            volume_fraction(fem, analysis, xPhys, c);
            if (c.value_initial == 0.0)
                c.value_initial = c.value;
            constr_value_vec[i] = c.value / c.value_initial - c.limit;
            const double alpha = 1.0 / c.value_initial;
            for (auto &row : c.d_explicit_df_dx)
            {
                for (double &v : row)
                    v *= alpha;
            }
        }
    }

    std::vector<std::vector<double>> AVM(const FEM &fem,
                                         Analysis &analysis,
                                         const std::vector<std::vector<double>> &,
                                         const std::vector<int> &x_label,
                                         ObjectiveGroup &object,
                                         const Filtering &filtering,
                                         const FilteringState &filtering_state,
                                         int Iter)
    {
        std::vector<std::vector<double>> out = dynamic_elastic_base_full_impl(fem, analysis, filtering_state.xPhys, object, filtering, filtering_state, Iter);
        for (std::size_t e = 0; e < out.size(); ++e)
        {
            if (x_label[e] != 0)
                std::fill(out[e].begin(), out[e].end(), 0.0);
        }
        return out;
    }

    std::vector<std::vector<double>> AVM(const FEM &fem,
                                         Analysis &analysis,
                                         const std::vector<std::vector<double>> &,
                                         const std::vector<int> &x_label,
                                         Constraint &object,
                                         const Filtering &filtering,
                                         const FilteringState &filtering_state,
                                         int Iter)
    {
        (void)fem;
        (void)analysis;
        (void)Iter;
        std::vector<std::vector<double>> dfdx_phys = object.d_explicit_df_dx;
        std::vector<std::vector<double>> out = filtering_sensitivity(dfdx_phys, filtering, filtering_state);
        for (std::size_t e = 0; e < out.size(); ++e)
        {
            if (x_label[e] != 0)
                std::fill(out[e].begin(), out[e].end(), 0.0);
        }
        return out;
    }

    std::vector<std::vector<double>> Debug(const FEM &fem,
                                           Analysis &analysis,
                                           const std::vector<std::vector<double>> &x,
                                           const std::vector<int> &x_label,
                                           ObjectiveGroup &object,
                                           const Filtering &filtering,
                                           const FilteringState &filtering_state,
                                           const OutputData &output,
                                           double fd_epsilon,
                                           int max_checks,
                                           int Iter)
    {
        PetscMPIInt rank = 0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        std::vector<std::vector<double>> avm = AVM(fem, analysis, x, x_label, object, filtering, filtering_state, Iter);

        const int nelem = fem.NElem;
        const int nvar = nvar_of(x);
        std::vector<std::pair<int, int>> candidates;
        candidates.reserve(static_cast<std::size_t>(nelem * nvar));
        for (int v = 0; v < nvar; ++v)
        {
            for (int e = 0; e < nelem; ++e)
            {
                if (x_label[static_cast<std::size_t>(e)] == 0)
                    candidates.push_back({e, v});
            }
        }
        if (max_checks <= 0 || max_checks > static_cast<int>(candidates.size()))
            max_checks = static_cast<int>(candidates.size());
        const int stride = std::max(1, static_cast<int>(candidates.size()) / std::max(1, max_checks));

        std::ofstream out;
        if (rank == 0)
        {
            std::filesystem::create_directories(output.outDir_mat);
            out.open(output.outDir_mat + "/sensitivity_debug.csv");
            out.precision(17);
            out << "Iter,elem,var,x,avm,fd,abs_err,rel_err,obj_plus,obj_minus\n";
            std::cout << "Debug sensitivity: finite-difference check on " << max_checks
                      << " design variables, eps=" << fd_epsilon << "\n";
        }

        int checked = 0;
        for (int idx = 0; idx < static_cast<int>(candidates.size()) && checked < max_checks; idx += stride)
        {
            const int e = candidates[static_cast<std::size_t>(idx)].first;
            const int v = candidates[static_cast<std::size_t>(idx)].second;
            const double x0 = x[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)];
            const double h_plus = std::min(fd_epsilon, 1.0 - x0);
            const double h_minus = std::min(fd_epsilon, x0);
            if (h_plus <= 0.0 && h_minus <= 0.0)
                continue;

            double obj_plus = object.value;
            double obj_minus = object.value;
            if (h_plus > 0.0)
            {
                std::vector<std::vector<double>> xp = x;
                FilteringState state_plus = filtering_state;
                xp[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] = x0 + h_plus;
                filtering_calculation(xp, x_label, filtering, state_plus);
                dynamic_elastic_linear(analysis, fem, state_plus.xPhys, Iter);
                obj_plus = objective_group_value_for_debug(fem, analysis, state_plus.xPhys, object);
            }
            if (h_minus > 0.0)
            {
                std::vector<std::vector<double>> xm = x;
                FilteringState state_minus = filtering_state;
                xm[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] = x0 - h_minus;
                filtering_calculation(xm, x_label, filtering, state_minus);
                dynamic_elastic_linear(analysis, fem, state_minus.xPhys, Iter);
                obj_minus = objective_group_value_for_debug(fem, analysis, state_minus.xPhys, object);
            }

            double fd = 0.0;
            if (h_plus > 0.0 && h_minus > 0.0)
            {
                fd = (obj_plus - obj_minus) / (h_plus + h_minus);
            }
            else if (h_plus > 0.0)
            {
                fd = (obj_plus - object.value) / h_plus;
            }
            else
            {
                fd = (object.value - obj_minus) / h_minus;
            }

            if (rank == 0)
            {
                const double g = avm[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)];
                const double abs_err = std::abs(g - fd);
                const double rel_err = abs_err / std::max(1e-14, std::abs(fd));
                out << Iter << "," << e << "," << v << "," << x0 << ","
                    << g << "," << fd << "," << abs_err << "," << rel_err << ","
                    << obj_plus << "," << obj_minus << "\n";
            }
            ++checked;
        }

        dynamic_elastic_linear(analysis, fem, filtering_state.xPhys, Iter);
        return avm;
    }

    void optimization11(const FEM &fem,
                        std::vector<Analysis> &analysis_all,
                        Optimization &Opt,
                        const OutputData &Output)
    {
        PetscMPIInt rank = 0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        FilteringState filtering_state;
        filtering_initilization(fem, Opt.filtering, filtering_state);

        const int nelem = fem.NElem;
        const int nvar = nvar_of(Opt.x);

        // MMA solver selection:
        // Main solver: line-by-line MATLAB mmasub/subsolv port. Use this for MATLAB matching.
        MatlabMMA mma(nelem * nvar, static_cast<int>(Opt.constraints.size()), 1.0, 0.0, 1000.0, 0.0);
        // Alternative 1: jdumas/mma upstream implementation. Not MATLAB-identical.
        // JdumasMMASolver mma(nelem * nvar, static_cast<int>(Opt.constraints.size()), 0.0, 1000.0, 0.0);
        // Alternative 2: older modified MMASolver kept for comparison. Not MATLAB-identical.
        // MMASolver mma(nelem * nvar, static_cast<int>(Opt.constraints.size()), 0.0, 1000.0, 0.0);

        int Iter = 0;
        double Change = 1.0;
        std::vector<double> constr_value_vec;
        std::vector<std::vector<double>> dObj_dx_mat;
        std::vector<double> dObj_dx_vec;
        std::vector<double> dConstr_dx;
        double obj_value = 0.0;
        using Clock = std::chrono::steady_clock;
        auto elapsed = [](Clock::time_point t0, Clock::time_point t1)
        {
            return std::chrono::duration<double>(t1 - t0).count();
        };

        double forward_time = 0.0;
        double objective_time = 0.0;
        double constraint_time = 0.0;
        double objective_sensitivity_time = 0.0;
        double constraint_sensitivity_time = 0.0;
        double mma_time = 0.0;
        int loopbeta = 0;

        while (Iter <= Opt.MaxIter && (Iter == 0 || Change > Opt.Tol))
        {
            const auto iter_t0 = Clock::now();
            if (Iter > 0)
                ++loopbeta;
            filtering_calculation(Opt.x, Opt.x_label, Opt.filtering, filtering_state);
            if (Opt.save_density_each_iter && Iter % Opt.save_density_interval == 0)
            {
                save_density_iteration(fem, filtering_state.xPhys, Output, Iter);
            }
            auto t0 = Clock::now();
            for (auto &A : analysis_all)
                dynamic_elastic_linear(A, fem, filtering_state.xPhys, Iter);
            auto t1 = Clock::now();
            forward_time = elapsed(t0, t1);
            t0 = Clock::now();
            evaluate_objective(fem, analysis_all, filtering_state.xPhys, Opt.objective_groups, obj_value);

            t1 = Clock::now();
            objective_time = elapsed(t0, t1);
            t0 = Clock::now();
            evaluate_constraint(fem, analysis_all, filtering_state.xPhys, Opt.constraints, constr_value_vec);
            t1 = Clock::now();
            constraint_time = elapsed(t0, t1);

            if (rank == 0)
            {
                std::cout << "-----------------------------------------------------" << "\n";
                std::cout << "Iter=" << Iter << " Obj=" << obj_value;
                if (!constr_value_vec.empty())
                    std::cout << " Constr=" << constr_value_vec.front();
                std::cout << " Change=" << Change << "\n";
                std::cout << "  timing[s]: forward=" << forward_time
                          << " obj_sens=" << objective_sensitivity_time
                          << " constr_sens=" << constraint_sensitivity_time << "\n";
            }

            dObj_dx_mat = zeros(nelem, nvar);
            t0 = Clock::now();
            for (auto &group : Opt.objective_groups)
            {
                Analysis &A = analysis_all[static_cast<std::size_t>(group.analysis_id - 1)];
                std::vector<std::vector<double>> dg = lower(group.sens_type) == "debug"
                                                          ? Debug(fem, A, Opt.x, Opt.x_label, group, Opt.filtering, filtering_state, Output, Opt.debug_fd_epsilon, Opt.debug_fd_max_checks, Iter)
                                                          : AVM(fem, A, Opt.x, Opt.x_label, group, Opt.filtering, filtering_state, Iter);
                add_scaled(dObj_dx_mat, dg, 1.0);
            }
            t1 = Clock::now();
            objective_sensitivity_time = elapsed(t0, t1);
            dObj_dx_vec = flatten(dObj_dx_mat);
            dConstr_dx.clear();
            t0 = Clock::now();
            for (auto &c : Opt.constraints)
            {
                Analysis &A = analysis_all[static_cast<std::size_t>(c.analysis_id - 1)];
                std::vector<std::vector<double>> dc = AVM(fem, A, Opt.x, Opt.x_label, c, Opt.filtering, filtering_state, Iter);
                std::vector<double> flat = flatten(dc);
                dConstr_dx.insert(dConstr_dx.end(), flat.begin(), flat.end());
            }
            t1 = Clock::now();
            constraint_sensitivity_time = elapsed(t0, t1);

            Opt.history.Iter.push_back(Iter);
            Opt.history.objective.push_back(obj_value);
            Opt.history.constraint.push_back(constr_value_vec.empty() ? 0.0 : constr_value_vec.front());
            Opt.history.change.push_back(Iter == 0 ? 0.0 : Change);
            Opt.history.rms_change.push_back(0.0);
            Opt.history.forward_time.push_back(forward_time);
            Opt.history.objective_time.push_back(objective_time);
            Opt.history.constraint_time.push_back(constraint_time);
            Opt.history.objective_sensitivity_time.push_back(objective_sensitivity_time);
            Opt.history.constraint_sensitivity_time.push_back(constraint_sensitivity_time);
            Opt.history.mma_time.push_back(0.0);
            Opt.history.iteration_time.push_back(0.0);
            /*
            if (rank == 0) {
                std::cout << "  timing[s]: forward=" << forward_time
                          << " objective=" << objective_time
                          << " constraint=" << constraint_time
                          << " obj_sens=" << objective_sensitivity_time
                          << " constr_sens=" << constraint_sensitivity_time;
            }
            */
            if (Iter >= Opt.MaxIter)
            {
                const double iteration_time = elapsed(iter_t0, Clock::now());
	                Opt.history.iteration_time.back() = iteration_time;
	                if (rank == 0)
	                    std::cout << " mma=0 iter_total=" << iteration_time << "\n";
	                save_history(Opt, Output);
	                save_rom_histories(analysis_all, Opt, Output);
	                break;
	            }

            t0 = Clock::now();
            std::vector<double> xvec = flatten(Opt.x);
            std::vector<double> xmin(xvec.size()), xmax(xvec.size());
            for (std::size_t i = 0; i < xvec.size(); ++i)
            {
                xmin[i] = std::max(0.0, xvec[i] - Opt.algorithm.move);
                xmax[i] = std::min(1.0, xvec[i] + Opt.algorithm.move);
            }
            std::vector<double> xold = xvec;
            mma.Update(xvec.data(), dObj_dx_vec.data(), constr_value_vec.data(), dConstr_dx.data(), xmin.data(), xmax.data());
            Change = 0.0;
            double rms = 0.0;
            for (std::size_t i = 0; i < xvec.size(); ++i)
            {
                xvec[i] = std::clamp(xvec[i], 0.0, 1.0);
                Change = std::max(Change, std::abs(xvec[i] - xold[i]));
                rms += (xvec[i] - xold[i]) * (xvec[i] - xold[i]);
            }
            rms = std::sqrt(rms / static_cast<double>(xvec.size()));
            Opt.history.rms_change.back() = rms;
            Opt.x = unflatten(xvec, nelem, nvar);
            t1 = Clock::now();
            mma_time = elapsed(t0, t1);
            Opt.history.mma_time.back() = mma_time;
            Opt.history.iteration_time.back() = elapsed(iter_t0, Clock::now());
            if (Opt.filtering.beta < Opt.filtering.beta_max && (loopbeta >= Opt.filtering.beta_loop || Change <= Opt.Tol))
            {
                const double beta_step = 2.0;
                Opt.filtering.beta = std::min(Opt.filtering.beta + beta_step, Opt.filtering.beta_max);
                loopbeta = 0;
                Change = 1.0;
                if (rank == 0)
                    std::cout << "Parameter beta increased to " << Opt.filtering.beta << ".\n";
            }
	            if (rank == 0)
	            {
	                std::cout << " mma=" << mma_time
	                          << " iter_total=" << Opt.history.iteration_time.back() << "\n";
	            }
	            save_history(Opt, Output);
	            save_rom_histories(analysis_all, Opt, Output);
	            ++Iter;
	        }
        save_history(Opt, Output);
        save_rom_histories(analysis_all, Opt, Output);
    }

} // namespace stage2

