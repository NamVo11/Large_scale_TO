#include "ROM/ROM_method/MyStrategy.hpp"

#include "ROM/Error_Estimator/ErrorEstimatorBase.hpp"
#include "ROM/Reduced_Basis_Algorithm/UpdateBase.hpp"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace {

bool env_flag(const char* name) {
    const char* value = std::getenv(name);
    return value && *value && std::atoi(value) != 0;
}

} // namespace

SolveStats MyStrategy::main(const Eigen::SparseMatrix<double>& lhs,
                            const Eigen::VectorXd& rhs,
                            Eigen::VectorXd& u) {
    check_strategy_ready("MyStrategy");
    const ROMContext& context = ctx;

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
        const bool needs_projection = fom_flag ||
                                      A_r_lhs_cache_ != &lhs ||
                                      A_r_rank_cache_ != rank ||
                                      A_r_cache.rows() != rank ||
                                      A_r_cache.cols() != rank ||
                                      AV_cache_.cols() != rank;
        if (needs_projection) {
            bool cache_ok = false;
            const bool force_full_projection = env_flag("STAGE2_ROM_FULL_PROJECTION") ||
                                               env_flag("STAGE2_MYSTRATEGY2_FULL_PROJECTION");
            if (incremental_projection && !force_full_projection && A_r_lhs_cache_ == &lhs) {
                cache_ok = basis_runtime.update_reduced_cache_incremental(lhs,
                                                                          V,
                                                                          A_r_cache,
                                                                          factorized_A_r,
                                                                          AV_cache_,
                                                                          A_r_rank_cache_,
                                                                          info);
            }
            if (cache_ok) {
                A_r_lhs_cache_ = &lhs;
                fom_flag = 0;
            } else {
                rebuild_reduced_cache(lhs, V, AV_cache_, A_r_lhs_cache_, A_r_rank_cache_, info);
            }
        }

        const double tb = now_time();
        const Eigen::VectorXd b_r = V.transpose() * rhs;
        info.timing_reduced_rhs += now_time() - tb;

        const double ts = now_time();
        Eigen::VectorXd coeff;
        if (A_r_cache.rows() == 0 || A_r_cache.cols() == 0) {
            use_fom = true;
        } else {
            coeff = factorized_A_r.solve(b_r);
            if (!coeff.allFinite()) {
                use_fom = true;
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
        }
    }

    Eigen::VectorXd solution_e;
    if (use_fom) {
        fom_solver->ctx = context;
        stats = fom_solver->FOM_main(lhs, rhs, u);
        solution_e = u;
        info.fom_used = true;
        info.ksp_iterations = stats.ksp_iterations;
        info.timing_fom_factorization += stats.factorization_time;
        info.timing_fom_solve += stats.solve_time;
        if (u_rom.size() == solution_e.size()) {
            //info.error = error_estimator->relative_solution_error(solution_e, u_rom);
            //info.basis_error = error_estimator->basis_error(V, solution_e);
        }
        if (online_update) {
            const double tu = now_time();
            basis_list[1]->update(solution_e, context);
            append_basis_debug("online_update_basis2", context, 2, *basis_list[1]);
            const double dt = now_time() - tu;
            info.timing_online_basis_update += dt;
            info.timing_basis_update += dt;
        }
        fom_flag = online_update ? (basis_list[1]->update_flag ? 1 : 0) : 0;
    } else {
        u = u_rom;
        solution_e = std::move(u_rom);
        info.rom_used = true;
        info.used_rom = true;
        info.accepted = true;
        fom_flag = 0;
    }

    if (offline_update) {
        const double tss = now_time();
        basis_list[0]->addSnapshot(solution_e, context);
        info.timing_snapshot += now_time() - tss;
    }

    if (context.is_end_of_analysis) {
        if (offline_update) {
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
        }
        const double tr = now_time();
        basis_list[1]->clearSnapshots();
        if (offline_update) basis_list[0]->reset();
        fom_flag = 1;
        clear_reduced_cache(AV_cache_, A_r_lhs_cache_, A_r_rank_cache_);
        info.timing_basis_copy_reset += now_time() - tr;
    }

    info.timing_total = now_time() - total_t0;
    store_info(info, context);
    return stats;
}
