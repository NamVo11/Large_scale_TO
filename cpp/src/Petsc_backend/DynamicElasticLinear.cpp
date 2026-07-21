#include "Petsc_backend/DynamicElasticLinear.hpp"

#include "FEMAssembly.hpp"
#include "ROM.hpp"
#include "helper/PetscEigenHelper.hpp"

#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace petsc_backend {
namespace {

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
    if (r.factorized_K_initial) KSPDestroy(&r.factorized_K_initial);
    if (r.factorized_K_eff) KSPDestroy(&r.factorized_K_eff);
}

void store_factorized_solver(KSP& target, KSP source) {
     if (!source) return;
    PetscObjectReference((PetscObject)source);
    if (target) KSPDestroy(&target);
    target = source;
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

[[maybe_unused]] double residual_norm_free(Mat A, Vec u, Vec rhs, const std::vector<PetscInt>& fixed) {
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
                            const Eigen::MatrixXd& y_elem,
                            int Iter) {
    if (!analysis.ROM) analysis.ROM = make_fom_solver("iterative");
    if (analysis.Nstep < 0 || analysis.dt <= 0.0) throw std::runtime_error("Invalid time data");

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    const PetscInt ndof = fem.ndof;
    clear_stored_petsc(analysis.results);

    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscLogDouble ts0 = 0.0;
    PetscLogDouble ts1 = 0.0;
    PetscLogDouble rhs_t0 = 0.0;
    PetscLogDouble rhs_t1 = 0.0;
    double timing_bc_and_initial_matrix = 0.0;
    double timing_vec_setup = 0.0;
    double timing_initial_rhs = 0.0;
    double timing_initial_solve = 0.0;
    double timing_keff_prepare = 0.0;
    double timing_predictor = 0.0;
    double timing_rhs_build = 0.0;
    double timing_rhs_set_neum = 0.0;
    double timing_rhs_matmult_mass = 0.0;
    double timing_rhs_vec_ops = 0.0;
    double timing_rhs_dirich = 0.0;
    double timing_solve_wall = 0.0;
    double timing_newmark_update = 0.0;
    double timing_history_store = 0.0;
    double timing_final_matrix_store = 0.0;
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
    if (rank == 0) {
        std::cout << "  assembly[s]: forward K/M = "
                  << analysis.results.assembly_time << "\n";
    }

    PetscTime(&ts0);
    const auto SPCdofs = fixed_dofs(analysis);
    IS fixed_is = nullptr;
    if (!SPCdofs.empty()) {
        ISCreateGeneral(PETSC_COMM_WORLD, static_cast<PetscInt>(SPCdofs.size()), SPCdofs.data(), PETSC_COPY_VALUES, &fixed_is);
    }

    Mat K_initial = nullptr;
    MatDuplicate(K, MAT_COPY_VALUES, &K_initial);
    if (fixed_is) MatZeroRowsColumnsIS(K_initial, fixed_is, 1.0, nullptr, nullptr);
    PetscTime(&ts1);
    timing_bc_and_initial_matrix += static_cast<double>(ts1 - ts0);

    PetscTime(&ts0);
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
    PetscTime(&ts1);
    timing_vec_setup += static_cast<double>(ts1 - ts0);

    PetscTime(&ts0);
    set_neum(analysis, f_ext, 0);
    VecCopy(f_ext, rhs);
    set_dirich_zero(rhs, SPCdofs);
    PetscTime(&ts1);
    timing_initial_rhs += static_cast<double>(ts1 - ts0);

    PetscTime(&ts0);
    const SolveStats initial_stats = analysis.ROM->main(
        K_initial, rhs, u, make_context(analysis, Iter, 1, 0, "initial_equilibrium", analysis.Nstep == 0));
    PetscTime(&ts1);
    timing_initial_solve += static_cast<double>(ts1 - ts0);
    timing_solve_wall += timing_initial_solve;
    store_factorized_solver(analysis.results.factorized_K_initial,
                            analysis.ROM->get_factorized_solver(K_initial));

    // Full residual_norm_free() costs one extra MatMult per step. Keep the
    // function available for debug, but use solver-reported residual in runs.
    const double initial_residual = initial_stats.residual_norm;
    if (rank == 0) {
        std::cout << "step   0: solve_res = " << initial_residual << "\n";
    }

    const double beta = 0.25;
    const double gamma = 0.5;
    const double a0c = 1.0 / (beta * analysis.dt * analysis.dt);

    Mat K_eff = nullptr;
    PetscTime(&ts0);
    MatDuplicate(K, MAT_COPY_VALUES, &K_eff);
    MatAXPY(K_eff, a0c, M, SAME_NONZERO_PATTERN);
    if (fixed_is) MatZeroRowsColumnsIS(K_eff, fixed_is, 1.0, nullptr, nullptr);
    PetscTime(&ts1);
    timing_keff_prepare += static_cast<double>(ts1 - ts0);

    const std::size_t nstore = static_cast<std::size_t>(analysis.Nstep + 1);
    analysis.results.U.resize(ndof, static_cast<int>(nstore));
    analysis.results.V.resize(ndof, static_cast<int>(nstore));
    analysis.results.A.resize(ndof, static_cast<int>(nstore));
    analysis.results.F_ext.resize(ndof, static_cast<int>(nstore));
    analysis.results.ksp_iterations.clear();
    analysis.results.ksp_residuals.clear();
    analysis.results.SPCdofs = to_ints(SPCdofs);
    analysis.results.free = free_dofs(ndof, SPCdofs);
    PetscTime(&ts0);
    gather_vec_to_all_into_column(u, analysis.results.U, 0);
    gather_vec_to_all_into_column(v, analysis.results.V, 0);
    gather_vec_to_all_into_column(acc, analysis.results.A, 0);
    gather_vec_to_all_into_column(f_ext, analysis.results.F_ext, 0);
    PetscTime(&ts1);
    timing_history_store += static_cast<double>(ts1 - ts0);

    PetscTime(&t0);
    for (int istep = 1; istep <= analysis.Nstep; ++istep) {
        //if (rank == 0) std::cout << "step " << istep << "/" << analysis.Nstep << "\n";

        PetscTime(&ts0);
        VecCopy(u, u_pred);
        VecAXPY(u_pred, analysis.dt, v);
        VecAXPY(u_pred, analysis.dt * analysis.dt * (0.5 - beta), acc);

        VecCopy(v, v_pred);
        VecAXPY(v_pred, analysis.dt * (1.0 - gamma), acc);
        PetscTime(&ts1);
        timing_predictor += static_cast<double>(ts1 - ts0);

        PetscTime(&rhs_t0);

        PetscTime(&ts0);
        set_neum(analysis, f_ext, istep);
        PetscTime(&ts1);
        timing_rhs_set_neum += static_cast<double>(ts1 - ts0);

        PetscTime(&ts0);
        MatMult(M, u_pred, tmp);
        PetscTime(&ts1);
        timing_rhs_matmult_mass += static_cast<double>(ts1 - ts0);

        PetscTime(&ts0);
        VecCopy(f_ext, rhs);
        VecAXPY(rhs, a0c, tmp);
        PetscTime(&ts1);
        timing_rhs_vec_ops += static_cast<double>(ts1 - ts0);

        PetscTime(&ts0);
        set_dirich_zero(rhs, SPCdofs);
        PetscTime(&ts1);
        timing_rhs_dirich += static_cast<double>(ts1 - ts0);

        PetscTime(&rhs_t1);
        timing_rhs_build += static_cast<double>(rhs_t1 - rhs_t0);

        PetscTime(&ts0);
        const SolveStats stats = analysis.ROM->main(
            K_eff, rhs, u, make_context(analysis, Iter, istep + 1, istep, "newmark_effective", istep == analysis.Nstep));
        PetscTime(&ts1);
        timing_solve_wall += static_cast<double>(ts1 - ts0);

        store_factorized_solver(analysis.results.factorized_K_eff,
                                    analysis.ROM->get_factorized_solver(K_eff));
      
        analysis.results.ksp_iterations.push_back(stats.ksp_iterations);
        analysis.results.ksp_residuals.push_back(stats.residual_norm);
        const double step_residual = stats.residual_norm;

        PetscTime(&ts0);
        VecCopy(u, tmp);
        VecAXPY(tmp, -1.0, u_pred);
        VecCopy(tmp, acc);
        VecScale(acc, a0c);

        VecCopy(v_pred, v);
        VecAXPY(v, gamma * analysis.dt, acc);
        PetscTime(&ts1);
        timing_newmark_update += static_cast<double>(ts1 - ts0);

        if (rank == 0) {
            std::cout << "step " << istep << ": solve_res = "
                      << step_residual
                      << ", ksp_it = " << stats.ksp_iterations
                      << ", ksp_res = " << stats.residual_norm << "\n";
        }

        const std::size_t store_id = static_cast<std::size_t>(istep);
        PetscTime(&ts0);
        gather_vec_to_all_into_column(u, analysis.results.U, static_cast<int>(store_id));
        gather_vec_to_all_into_column(v, analysis.results.V, static_cast<int>(store_id));
        gather_vec_to_all_into_column(acc, analysis.results.A, static_cast<int>(store_id));
        gather_vec_to_all_into_column(f_ext, analysis.results.F_ext, static_cast<int>(store_id));
        PetscTime(&ts1);
        timing_history_store += static_cast<double>(ts1 - ts0);
    }
    PetscTime(&t1);
    analysis.results.solve_time = t1 - t0;

    PetscTime(&ts0);
    MatDuplicate(K, MAT_COPY_VALUES, &analysis.results.K_inf);
    MatDuplicate(M, MAT_COPY_VALUES, &analysis.results.M);
    MatDuplicate(K_eff, MAT_COPY_VALUES, &analysis.results.K_eff);
    PetscTime(&ts1);
    timing_final_matrix_store += static_cast<double>(ts1 - ts0);

    if (rank == 0) {
        std::cout << "  forward breakdown[s]:"
                  << " bc_initial_matrix=" << timing_bc_and_initial_matrix
                  << " vec_setup=" << timing_vec_setup
                  << " initial_rhs=" << timing_initial_rhs
                  << " initial_solve=" << timing_initial_solve
                  << " keff_prepare=" << timing_keff_prepare
                  << " predictor=" << timing_predictor
                  << " rhs_build=" << timing_rhs_build
                  << " rhs_set_neum=" << timing_rhs_set_neum
                  << " rhs_matmult_mass=" << timing_rhs_matmult_mass
                  << " rhs_vec_ops=" << timing_rhs_vec_ops
                  << " rhs_dirich=" << timing_rhs_dirich
                  << " solve_wall=" << timing_solve_wall
                  << " newmark_update=" << timing_newmark_update
                  << " history_store=" << timing_history_store
                  << " final_matrix_store=" << timing_final_matrix_store
                  << "\n";
    }

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

} // namespace petsc_backend
