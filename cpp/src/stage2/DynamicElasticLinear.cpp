#include "stage2/DynamicElasticLinear.hpp"

#include "stage2/FEMAssembly.hpp"
#include "stage2/ROM.hpp"

#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace stage2 {
namespace {

std::vector<double> gather_vec_to_root(Vec v) {
    Vec vall = nullptr;
    VecScatter scatter = nullptr;
    VecScatterCreateToAll(v, &scatter, &vall);
    VecScatterBegin(scatter, v, vall, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatter, v, vall, INSERT_VALUES, SCATTER_FORWARD);

    std::vector<double> out;
    PetscInt n = 0;
    VecGetSize(vall, &n);
    out.resize(static_cast<std::size_t>(n));
    const PetscScalar* a = nullptr;
    VecGetArrayRead(vall, &a);
    for (PetscInt i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = PetscRealPart(a[i]);
    VecRestoreArrayRead(vall, &a);

    VecScatterDestroy(&scatter);
    VecDestroy(&vall);
    return out;
}

std::vector<PetscInt> fixed_dofs(const Analysis& analysis) {
    std::vector<PetscInt> dofs;
    for (const auto& bc : analysis.Dirich) {
        dofs.insert(dofs.end(), bc.dof_ids.begin(), bc.dof_ids.end());
    }
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    return dofs;
}

std::vector<int> to_ints(const std::vector<PetscInt>& in) {
    std::vector<int> out(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) out[i] = static_cast<int>(in[i]);
    return out;
}

std::vector<int> free_dofs(PetscInt ndof, const std::vector<PetscInt>& fixed) {
    std::vector<char> is_fixed(static_cast<std::size_t>(ndof), 0);
    for (PetscInt d : fixed) {
        if (d >= 0 && d < ndof) is_fixed[static_cast<std::size_t>(d)] = 1;
    }
    std::vector<int> free;
    free.reserve(static_cast<std::size_t>(ndof) - fixed.size());
    for (PetscInt d = 0; d < ndof; ++d) {
        if (!is_fixed[static_cast<std::size_t>(d)]) free.push_back(static_cast<int>(d));
    }
    return free;
}

void clear_stored_petsc(AnalysisResult& r) {
    if (r.K_inf) MatDestroy(&r.K_inf);
    if (r.M) MatDestroy(&r.M);
    if (r.K_eff) MatDestroy(&r.K_eff);
}

void set_dirich_zero(Vec rhs, const std::vector<PetscInt>& fixed) {
    if (fixed.empty()) return;
    std::vector<PetscScalar> zeros(fixed.size(), 0.0);
    VecSetValues(rhs, static_cast<PetscInt>(fixed.size()), fixed.data(), zeros.data(), INSERT_VALUES);
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);
}

void set_neum(const Analysis& analysis, Vec f, int istep) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    VecSet(f, 0.0);
    if (rank == 0) {
        for (const auto& bc : analysis.Neum) {
            const PetscScalar value = bc.fun_value ? bc.fun_value(istep) : bc_value(bc, analysis, istep);
            for (int dof0 : bc.dof_ids) {
                PetscInt dof = dof0;
                VecSetValues(f, 1, &dof, &value, ADD_VALUES);
            }
        }
    }
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);
}

ROMContext make_context(const Analysis& analysis,
                        int Iter,
                        int solve_step,
                        int time_step,
                        const std::string& system_type,
                        bool is_end) {
    ROMContext ctx;
    ctx.analysis_id = analysis.id;
    ctx.opt_iter = Iter;
    ctx.solve_step = solve_step;
    ctx.time_step = time_step;
    ctx.time = time_step * analysis.dt;
    ctx.role = "primal";
    ctx.system_type = system_type;
    ctx.is_end_of_analysis = is_end;
    return ctx;
}

double residual_norm_free(Mat A, Vec u, Vec rhs, const std::vector<PetscInt>& fixed) {
    Vec r = nullptr;
    VecDuplicate(rhs, &r);
    MatMult(A, u, r);
    VecAXPY(r, -1.0, rhs);
    set_dirich_zero(r, fixed);
    PetscReal nrm = 0.0;
    VecNorm(r, NORM_2, &nrm);
    VecDestroy(&r);
    return static_cast<double>(nrm);
}

} // namespace

void dynamic_elastic_linear(Analysis& analysis,
                            const FEM& fem,
                            const std::vector<std::vector<double>>& y_elem,
                            int Iter) {
    if (!analysis.ROM) analysis.ROM = make_fom_solver("iterative");
    if (analysis.Nstep < 0 || analysis.dt <= 0.0) throw std::runtime_error("Invalid time data");

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    const PetscInt ndof = fem.ndof;
    clear_stored_petsc(analysis.results);

    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);

    Mat K = nullptr;
    Mat M = nullptr;
    const PetscInt d_nz = fem.dim == 2 ? 64 : 192;
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ndof, ndof,
                 d_nz, nullptr, d_nz, nullptr, &K);
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ndof, ndof,
                 d_nz, nullptr, d_nz, nullptr, &M);
    assemble_K_M(fem, analysis, y_elem, K, M);
    PetscTime(&t1);
    analysis.results.assembly_time = t1 - t0;

    const auto SPCdofs = fixed_dofs(analysis);
    IS fixed_is = nullptr;
    if (!SPCdofs.empty()) {
        ISCreateGeneral(PETSC_COMM_WORLD, static_cast<PetscInt>(SPCdofs.size()), SPCdofs.data(), PETSC_COPY_VALUES, &fixed_is);
    }

    Mat K_initial = nullptr;
    MatDuplicate(K, MAT_COPY_VALUES, &K_initial);
    if (fixed_is) MatZeroRowsColumnsIS(K_initial, fixed_is, 1.0, nullptr, nullptr);

    Vec u = nullptr;
    Vec v = nullptr;
    Vec acc = nullptr;
    Vec u_pred = nullptr;
    Vec v_pred = nullptr;
    Vec rhs = nullptr;
    Vec f_ext = nullptr;
    Vec tmp = nullptr;
    MatCreateVecs(K, &u, &rhs);
    VecDuplicate(u, &v);
    VecDuplicate(u, &acc);
    VecDuplicate(u, &u_pred);
    VecDuplicate(u, &v_pred);
    VecDuplicate(u, &f_ext);
    VecDuplicate(u, &tmp);
    VecSet(u, 0.0);
    VecSet(v, 0.0);
    VecSet(acc, 0.0);

    set_neum(analysis, f_ext, 0);
    VecCopy(f_ext, rhs);
    set_dirich_zero(rhs, SPCdofs);
    analysis.ROM->set_factorization(K_initial);
    analysis.ROM->main(K_initial, rhs, u, make_context(analysis, Iter, 1, 0, "initial_equilibrium", analysis.Nstep == 0));

    const double initial_residual = residual_norm_free(K_initial, u, rhs, SPCdofs);
    if (rank == 0) {
        std::cout << "step   0: ||res_free||_2 = " << initial_residual << "\n";
    }

    const double beta = 0.25;
    const double gamma = 0.5;
    const double a0c = 1.0 / (beta * analysis.dt * analysis.dt);

    Mat K_eff = nullptr;
    MatDuplicate(K, MAT_COPY_VALUES, &K_eff);
    MatAXPY(K_eff, a0c, M, SAME_NONZERO_PATTERN);
    if (fixed_is) MatZeroRowsColumnsIS(K_eff, fixed_is, 1.0, nullptr, nullptr);
    analysis.ROM->set_factorization(K_eff);

    analysis.results.U.clear();
    analysis.results.V.clear();
    analysis.results.A.clear();
    analysis.results.F_ext.clear();
    analysis.results.ksp_iterations.clear();
    analysis.results.ksp_residuals.clear();
    analysis.results.SPCdofs = to_ints(SPCdofs);
    analysis.results.free = free_dofs(ndof, SPCdofs);
    analysis.results.U.push_back(gather_vec_to_root(u));
    analysis.results.V.push_back(gather_vec_to_root(v));
    analysis.results.A.push_back(gather_vec_to_root(acc));
    analysis.results.F_ext.push_back(gather_vec_to_root(f_ext));

    PetscTime(&t0);
    for (int istep = 1; istep <= analysis.Nstep; ++istep) {
        //if (rank == 0) std::cout << "step " << istep << "/" << analysis.Nstep << "\n";

        VecCopy(u, u_pred);
        VecAXPY(u_pred, analysis.dt, v);
        VecAXPY(u_pred, analysis.dt * analysis.dt * (0.5 - beta), acc);

        VecCopy(v, v_pred);
        VecAXPY(v_pred, analysis.dt * (1.0 - gamma), acc);

        set_neum(analysis, f_ext, istep);
        MatMult(M, u_pred, tmp);
        VecCopy(f_ext, rhs);
        VecAXPY(rhs, a0c, tmp);
        set_dirich_zero(rhs, SPCdofs);

        const SolveStats stats = analysis.ROM->main(
            K_eff, rhs, u, make_context(analysis, Iter, istep + 1, istep, "newmark_effective", istep == analysis.Nstep));
        analysis.results.ksp_iterations.push_back(stats.ksp_iterations);
        analysis.results.ksp_residuals.push_back(stats.residual_norm);
        const double step_residual = residual_norm_free(K_eff, u, rhs, SPCdofs);

        VecCopy(u, tmp);
        VecAXPY(tmp, -1.0, u_pred);
        VecCopy(tmp, acc);
        VecScale(acc, a0c);

        VecCopy(v_pred, v);
        VecAXPY(v, gamma * analysis.dt, acc);

        if (rank == 0) {
            std::cout << "step " << istep << ": ||res_free||_2 = "
                      << step_residual
                      << ", ksp_it = " << stats.ksp_iterations
                      << ", ksp_res = " << stats.residual_norm << "\n";
        }

        analysis.results.U.push_back(gather_vec_to_root(u));
        analysis.results.V.push_back(gather_vec_to_root(v));
        analysis.results.A.push_back(gather_vec_to_root(acc));
        analysis.results.F_ext.push_back(gather_vec_to_root(f_ext));
    }
    PetscTime(&t1);
    analysis.results.solve_time = t1 - t0;

    MatDuplicate(K, MAT_COPY_VALUES, &analysis.results.K_inf);
    MatDuplicate(M, MAT_COPY_VALUES, &analysis.results.M);
    MatDuplicate(K_eff, MAT_COPY_VALUES, &analysis.results.K_eff);

    VecDestroy(&u);
    VecDestroy(&v);
    VecDestroy(&acc);
    VecDestroy(&u_pred);
    VecDestroy(&v_pred);
    VecDestroy(&rhs);
    VecDestroy(&f_ext);
    VecDestroy(&tmp);
    MatDestroy(&K_eff);
    MatDestroy(&K_initial);
    MatDestroy(&K);
    MatDestroy(&M);
    if (fixed_is) ISDestroy(&fixed_is);
}

} // namespace stage2
