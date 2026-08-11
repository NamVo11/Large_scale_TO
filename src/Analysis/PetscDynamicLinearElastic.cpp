#include "Analysis/Analysis.hpp"
#include "MatlabLikeData.hpp"
#include "ROM/ROM.hpp"
#include "helper/MyAlgebra.hpp"
#include "helper/MemoryDebug.hpp"
#include "helper/PetscEigenBridge.hpp"
#include "helper/PetscEigenHelper.hpp"

#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

std::pair<double, double> interp_region(const Region& region, double x) {
    const double E = region.interp.elastic_interp(region.mat_paras.E_inf, x);
    const double rho = region.interp.mass_density_interp(region.mat_paras.rho, x);
    return {E, rho};
}

} // namespace

void PetscDynamicLinearElastic::assemble_K_M(const FEM& fem,
                                             const Eigen::MatrixXd& y_elem,
                                             Mat K,
                                             Mat M) const {
    const Analysis& analysis = *this;
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    const int elem_start = (fem.NElem * rank) / size;
    const int elem_end = (fem.NElem * (rank + 1)) / size;

    for (const Region& region : analysis.regions) {
        const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            if (region.elem_ids.empty()) throw std::runtime_error("assemble_K_M: region has no elements");
            uniform_em = fem.compute_element_matrices(region.elem_ids.front(), region_nu, fem.Thickness);
        }

        for (int elem_id : region.elem_ids) {
            if (elem_id < elem_start || elem_id >= elem_end) continue;
            const double x = y_elem(elem_id, 0);
            const auto [Eeff, rhoeff] = interp_region(region, x);
            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                local_em = fem.compute_element_matrices(elem_id, region_nu, fem.Thickness);
                em_ptr = &local_em;
            }
            const ElementMatrix& em = *em_ptr;
            const auto& rows = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            const LocalDenseMatrix ke = Eeff * em.ke;
            const LocalDenseMatrix me = rhoeff * em.me;
            MatSetValues(K, em.ndof, rows.data(), em.ndof, rows.data(), ke.data(), ADD_VALUES);
            MatSetValues(M, em.ndof, rows.data(), em.ndof, rows.data(), me.data(), ADD_VALUES);
        }
    }
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

namespace {

void clear_stored_petsc(Analysis& analysis) {
    if (analysis.ROM) analysis.ROM->set_petsc_factorized_solver(nullptr, nullptr);
    AnalysisResult& r = analysis.results;
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

void zero_spc_entries_for_residual(Vec rhs, const std::vector<int>& fixed) {
    if (fixed.empty()) return;
    std::vector<PetscInt> ids;
    ids.reserve(fixed.size());
    for (int d : fixed) ids.push_back(static_cast<PetscInt>(d));
    std::vector<PetscScalar> zeros(fixed.size(), 0.0);
    VecSetValues(rhs, static_cast<PetscInt>(ids.size()), ids.data(), zeros.data(), INSERT_VALUES);
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);
}

double petsc_wall_time() {
    PetscLogDouble t = 0.0;
    PetscTime(&t);
    return static_cast<double>(t);
}

void set_dirichlet_rhs(const Analysis& analysis, Vec rhs, int istep) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
        for (const auto& bc : analysis.Dirich) {
            const PetscScalar value = bc.fun_value ? bc.fun_value(istep) : bc_value(bc, analysis, istep);
            for (int dof0 : bc.dof_ids) {
                const PetscInt dof = static_cast<PetscInt>(dof0);
                VecSetValues(rhs, 1, &dof, &value, INSERT_VALUES);
            }
        }
    }
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

[[maybe_unused]] double residual_norm_free(Mat A, Vec u, Vec rhs, const std::vector<int>& fixed) {
    Vec r = nullptr;
    VecDuplicate(rhs, &r);
    MatMult(A, u, r);
    VecAXPY(r, -1.0, rhs);
    zero_spc_entries_for_residual(r, fixed);
    PetscReal nrm = 0.0;
    VecNorm(r, NORM_2, &nrm);
    VecDestroy(&r);
    return static_cast<double>(nrm);
}

SolveStats solve_with_common_rom(ROMBase& rom,
                                 Mat lhs,
                                 Vec rhs,
                                 Vec u,
                                 const ROMContext& context,
                                 Eigen::SparseMatrix<double>* lhs_e_cache = nullptr) {
    rom.set_petsc_system(lhs, rhs, u);
    if (!rom.use_mor) {
        Eigen::SparseMatrix<double> empty_lhs;
        Eigen::VectorXd empty_rhs;
        Eigen::VectorXd empty_u;
        return rom.main(empty_lhs, empty_rhs, empty_u, context);
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
    Eigen::VectorXd u_e = Eigen::VectorXd::Zero(rhs_e.size());
    const SolveStats stats = rom.main(*lhs_e_ptr, rhs_e, u_e, context);
    eigen_to_petsc_vec(u_e, u);
    if (!rom.ROM_info.empty() && !rom.ROM_info.back().empty()) {
        ROMInfo& info = rom.ROM_info.back().back();
        info.timing_petsc_to_eigen_mat += matrix_convert_time;
        info.timing_petsc_to_eigen_vec += vec_convert_time;
    }
    return stats;
}

} // namespace

void PetscDynamicLinearElastic::solver(const FEM& fem, const Eigen::MatrixXd& y_elem,int Iter)
{
    Analysis& analysis = *this;
    if (!analysis.ROM) throw std::runtime_error("PetscDynamicLinearElastic::solver requires analysis.ROM initialized");
    if (analysis.Nstep < 0 || analysis.dt <= 0.0) throw std::runtime_error("Invalid time data");

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    const PetscInt ndof = fem.ndof;
    clear_stored_petsc(analysis);

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
    Eigen::SparseMatrix<double> K_initial_eigen_cache;
    Eigen::SparseMatrix<double> K_eff_eigen_cache;
    PetscTime(&t0);
    MemoryDebug::print("forward_begin");

    const PetscInt d_nz = fem.dim == 2 ? 64 : 192; // approximate nonzeros per row
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ndof, ndof, d_nz, nullptr, d_nz, nullptr, &analysis.results.K_inf);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ndof, ndof,d_nz, nullptr, d_nz, nullptr, &analysis.results.M);

    // Memory-saving flow: results.K_inf is assembled as raw K first. After
    // results.K_eff is formed, K_inf is converted in-place to K_initial with SPC.
    Mat K = analysis.results.K_inf;
    Mat M = analysis.results.M;
    assemble_K_M(fem, y_elem, K, M);
    MemoryDebug::print("forward_after_assemble_K_M");
    PetscTime(&t1);
    analysis.results.assembly_time = t1 - t0;
    if (rank == 0) {
        std::cout << "  assembly[s]: forward K/M = "
                  << analysis.results.assembly_time << "\n";
    }

    PetscTime(&ts0);
    const auto SPCdofs = analysis.SPC();

    const double beta = 0.25;
    const double gamma = 0.5;
    const double a0c = 1.0 / (beta * analysis.dt * analysis.dt);

    PetscTime(&ts0);
    MatDuplicate(K, MAT_COPY_VALUES, &analysis.results.K_eff);
    MatAXPY(analysis.results.K_eff, a0c, M, SAME_NONZERO_PATTERN);
    Mat K_eff = analysis.results.K_eff;
    MemoryDebug::print("forward_after_prepare_Keff");
    PetscTime(&ts1);
    timing_keff_prepare += static_cast<double>(ts1 - ts0);

    //MyAlgebra::Petsc::check_matrix_diagonal(K,"K");

    PetscTime(&ts0);
    // From here, K is K_initial with SPC, not raw K.
    MyAlgebra::Petsc::apply_dirichlet_to_matrix(K, SPCdofs, 1.0);
    MyAlgebra::Petsc::apply_dirichlet_to_matrix(K_eff, SPCdofs, 1.0);
    MemoryDebug::print("forward_after_dirichlet_K_initial");
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
    MemoryDebug::print("forward_after_vec_create");
    VecSet(u, 0.0);
    VecSet(v, 0.0);
    VecSet(acc, 0.0);
    PetscTime(&ts1);
    timing_vec_setup += static_cast<double>(ts1 - ts0);

    PetscTime(&ts0);
    set_neum(analysis, f_ext, 0);
    VecCopy(f_ext, rhs);
    set_dirichlet_rhs(analysis, rhs, 0);
    PetscTime(&ts1);
    timing_initial_rhs += static_cast<double>(ts1 - ts0);

    PetscTime(&ts0);
    const SolveStats initial_stats = solve_with_common_rom(
        *analysis.ROM, K, rhs, u,
        make_context(analysis, Iter, 1, 0, "initial_equilibrium", analysis.Nstep == 0),
        &K_initial_eigen_cache);
    PetscTime(&ts1);
    timing_initial_solve += static_cast<double>(ts1 - ts0);
    timing_solve_wall += timing_initial_solve;

    if (analysis.save_factorized_K_initial) {
        store_factorized_solver(analysis.results.factorized_K_initial, analysis.ROM->get_petsc_factorized_solver(K));
    } else {
        analysis.ROM->set_petsc_factorized_solver(nullptr, nullptr);
        if (analysis.results.factorized_K_initial) KSPDestroy(&analysis.results.factorized_K_initial);
    }

    MemoryDebug::print("forward_after_initial_solve_store_factor");

    // Full residual_norm_free() costs one extra MatMult per step. Keep the
    // function available for debug, but use solver-reported residual in runs.
    const double initial_residual = initial_stats.residual_norm;
    if (rank == 0) {
        std::cout << "step   0: solve_res = " << initial_residual << "\n";
    }

    const std::size_t nstore = static_cast<std::size_t>(analysis.Nstep + 1);
    analysis.results.U.resize(ndof, static_cast<int>(nstore));
    analysis.results.V.resize(0, 0);
    analysis.results.A.resize(ndof, static_cast<int>(nstore));
    analysis.results.F_ext.resize(ndof, static_cast<int>(nstore));
    MemoryDebug::print("forward_after_history_alloc");
    analysis.results.ksp_iterations.clear();
    analysis.results.ksp_residuals.clear();
    analysis.results.SPCdofs = SPCdofs;
    analysis.results.free = analysis.free_dofs(static_cast<int>(ndof));
    PetscTime(&ts0);
    gather_vec_to_all_into_column(u, analysis.results.U, 0);
    // Velocity history is not used by the current objective/sensitivity flow.
    // Keep results.V empty in this memory-saving version.
    // gather_vec_to_all_into_column(v, analysis.results.V, 0);
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
        set_dirichlet_rhs(analysis, rhs, istep);
        PetscTime(&ts1);
        timing_rhs_dirich += static_cast<double>(ts1 - ts0);

        PetscTime(&rhs_t1);
        timing_rhs_build += static_cast<double>(rhs_t1 - rhs_t0);

        PetscTime(&ts0);
        const SolveStats stats = solve_with_common_rom( *analysis.ROM, K_eff, rhs, u,
                                            make_context(analysis, Iter, istep + 1, istep, "newmark_effective", istep == analysis.Nstep),
                                            &K_eff_eigen_cache);
                                            
        PetscTime(&ts1);
        timing_solve_wall += static_cast<double>(ts1 - ts0);

        if (!analysis.results.factorized_K_eff) {
            store_factorized_solver(analysis.results.factorized_K_eff,
                                    analysis.ROM->get_petsc_factorized_solver(K_eff));
        }

        if (istep == 1 || istep == analysis.Nstep) {
            MemoryDebug::print("forward_after_step_" + std::to_string(istep));
        }
      
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
        // Velocity history is intentionally not stored in this memory-saving flow.
        // gather_vec_to_all_into_column(v, analysis.results.V, static_cast<int>(store_id));
        gather_vec_to_all_into_column(acc, analysis.results.A, static_cast<int>(store_id));
        gather_vec_to_all_into_column(f_ext, analysis.results.F_ext, static_cast<int>(store_id));
        PetscTime(&ts1);
        timing_history_store += static_cast<double>(ts1 - ts0);
    }
    PetscTime(&t1);
    analysis.results.solve_time = t1 - t0;

    MemoryDebug::print("forward_after_store_matrices");

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
    // K, M and K_eff are owned by analysis.results and reused by sensitivity.
    MemoryDebug::print("forward_after_destroy_temp");
}
