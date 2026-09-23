#include "ROM/ROM_method/MyStrategy.hpp"

#include "ROM/Reduced_Basis_Algorithm/UpdateBase.hpp"

#include <stdexcept>
#include <utility>

SolveStats MyStrategy::main(const Eigen::SparseMatrix<double>& lhs,
                            const Eigen::VectorXd& rhs,
                            Eigen::VectorXd& u,
                            const ROMContext& context) {
    check_strategy_ready("MyStrategy");

    const double total_t0 = now_time();
    ROMInfo info;
    info.name = name;
    SolveStats stats;

    const auto& basis_runtime = *basis_list[1];
    const Eigen::MatrixXd& V = basis_runtime.getBasis(context);
    const int rank = static_cast<int>(V.cols());
    info.rank = rank;

    bool use_fom = !use_mor || rank == 0;
    Eigen::VectorXd u_rom;

    if (!use_fom) {
        const bool rebuild_projection = fom_flag || A_r_lhs_cache_ != &lhs || A_r_rank_cache_ != rank;
        if (rebuild_projection) {
            rebuild_reduced_cache(lhs, V, AV_cache_, A_r_lhs_cache_, A_r_rank_cache_, info);
        }

        const double tb = now_time();
        const Eigen::VectorXd b_r = V.transpose() * rhs;
        info.timing_reduced_rhs += now_time() - tb;

        const double ts = now_time();
        Eigen::VectorXd coeff;
        if (A_r_cache.rows() == 0 || A_r_cache.cols() == 0) {
            use_fom = true;
            fom_flag = 1;
        } else {
            coeff = factorized_A_r.solve(b_r);
            if (!coeff.allFinite()) {
                use_fom = true;
                fom_flag = 1;
            }
        }
        info.timing_reduced_solve += now_time() - ts;

        if (!use_fom) {
            const double tr = now_time();
            u_rom = V * coeff;
            info.timing_reconstruct += now_time() - tr;

            const double te = now_time();
            const ErrorAssessment assessment = error_estimator->assess(lhs, rhs, u_rom, context, mkl_lhs_cache_);
            info.timing_residual += now_time() - te;
            info.rank = rank;
            info.rank_true = rank;
            info.error_indicator = assessment.error_indicator;
            info.residual = assessment.residual;
            info.residual_norm = assessment.residual;
            info.threshold = assessment.threshold;
            info.residual_tol_eff = assessment.residual_tol_eff;
            info.accepted = assessment.accepted;
            append_decision_debug(context, rank, assessment.residual, assessment.threshold, assessment.accepted);
            stats.residual_norm = assessment.residual;
            use_fom = !info.accepted;
            if (use_fom) fom_flag = 1;
        }
    }

    Eigen::VectorXd solution_e;
    if (use_fom) {
        stats = fom_solver->FOM_main(lhs, rhs, u, context);
        solution_e = u;
        info.fom_used = true;
        info.ksp_iterations = stats.ksp_iterations;
        info.timing_fom_factorization += stats.factorization_time;
        info.timing_fom_solve += stats.solve_time;
        if (u_rom.size() == solution_e.size()) {
            info.error = error_estimator->relative_solution_error(solution_e, u_rom);
            info.basis_error = error_estimator->basis_error(V, solution_e);
        }
        if (online) {
            const double tu = now_time();
            basis_list[1]->update(solution_e, context);
            append_basis_debug("online_update_basis2", context, 2, *basis_list[1]);
            const double dt = now_time() - tu;
            info.timing_online_basis_update += dt;
            info.timing_basis_update += dt;
        }
        fom_flag = online ? (basis_list[1]->update_flag ? 1 : 0) : 0;
    } else {
        Eigen::VectorXd fom_reference;
        const SolveStats fom_stats = fom_solver->FOM_main(lhs, rhs, fom_reference, context);
        info.timing_fom_factorization += fom_stats.factorization_time;
        info.timing_fom_solve += fom_stats.solve_time;
        if (u_rom.size() == fom_reference.size()) {
            info.error = error_estimator->relative_solution_error(fom_reference, u_rom);
            info.basis_error = error_estimator->basis_error(V, fom_reference);
        }

        u = u_rom;
        solution_e = std::move(u_rom);
        info.rom_used = true;
        info.used_rom = true;
        info.accepted = true;
        fom_flag = 0;
    }

    const double tss = now_time();
    basis_list[0]->addSnapshot(solution_e, context);
    info.timing_snapshot += now_time() - tss;

    if (context.is_end_of_analysis) {
        if (basis_list[0]->max_rank != 0) {
            const double tu = now_time();
            basis_list[0]->update(solution_e, context);
            append_basis_debug("offline_update_basis1", context, 1, *basis_list[0]);
            const double dt = now_time() - tu;
            info.timing_basis1_update += dt;
            info.timing_end_basis_update += dt;
            info.timing_basis_update += dt;

            const double tc = now_time();
            basis_list[1]->setBasis(basis_list[0]->V, basis_list[0]->s);
            append_basis_debug("copy_basis1_to_basis2", context, 2, *basis_list[1]);
            info.timing_basis_copy_reset += now_time() - tc;
        } else {
            const double tc = now_time();
            basis_list[1]->reset();
            append_basis_debug("reset_basis2", context, 2, *basis_list[1]);
            info.timing_basis_copy_reset += now_time() - tc;
        }
        const double tr = now_time();
        basis_list[1]->clearSnapshots();
        basis_list[0]->reset();
        fom_flag = 1;
        clear_reduced_cache(AV_cache_, A_r_lhs_cache_, A_r_rank_cache_);
        info.timing_basis_copy_reset += now_time() - tr;
    }

    info.timing_total = now_time() - total_t0;
    store_info(info, context);
    return stats;
}
