#include "ROM/ROM_method/MyStrategy2.hpp"

#include "ROM/Reduced_Basis_Algorithm/MyIncrementalQR1.hpp"
#include "ROM/Reduced_Basis_Algorithm/UpdateBase.hpp"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace {

int env_int_or_default(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    return std::atoi(value);
}

} // namespace

bool MyStrategy2::update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                                   const Eigen::MatrixXd& V,
                                                   const UpdateBase& basis_runtime,
                                                   ROMInfo& info) {
    const int rank = static_cast<int>(V.cols());
    const int old_rank = A_r_rank_cache_;
    if (rank <= 0 || A_r_cache.rows() != old_rank || AV_cache_.cols() != old_rank) return false;

    const auto* qr_basis = dynamic_cast<const MyIncrementalQR1*>(&basis_runtime);
    const int drop_index = qr_basis ? qr_basis->base_rank : -1;
    const double t0 = now_time();

    auto append_column = [&](const Eigen::MatrixXd& A_r_base,
                             const Eigen::MatrixXd& AV_base,
                             const Eigen::VectorXd& v_new,
                             int new_rank) {
        const Eigen::VectorXd Av_new = A * v_new;
        Eigen::MatrixXd A_r_new(new_rank, new_rank);
        if (new_rank > 1) {
            A_r_new.topLeftCorner(new_rank - 1, new_rank - 1) = A_r_base;
            A_r_new.topRightCorner(new_rank - 1, 1) = V.leftCols(new_rank - 1).transpose() * Av_new;
            A_r_new.bottomLeftCorner(1, new_rank - 1) = v_new.transpose() * AV_base;
        }
        A_r_new(new_rank - 1, new_rank - 1) = v_new.dot(Av_new);

        Eigen::MatrixXd AV_new(AV_base.rows(), new_rank);
        if (new_rank > 1) AV_new.leftCols(new_rank - 1) = AV_base;
        AV_new.col(new_rank - 1) = Av_new;
        A_r_cache = std::move(A_r_new);
        factorized_A_r.compute(A_r_cache);
        AV_cache_ = std::move(AV_new);
    };

    if (rank == old_rank + 1) {
        append_column(A_r_cache, AV_cache_, V.col(rank - 1), rank);
        info.timing_projection += now_time() - t0;
        A_r_rank_cache_ = rank;
        fom_flag = 0;
        return true;
    }

    if (qr_basis && drop_index >= 0 && drop_index < old_rank &&
        (rank == old_rank || rank == old_rank - 1)) {
        const int kept_rank = old_rank - 1;
        Eigen::MatrixXd A_r_kept(kept_rank, kept_rank);
        Eigen::MatrixXd AV_kept(AV_cache_.rows(), kept_rank);
        int ii = 0;
        for (int i = 0; i < old_rank; ++i) {
            if (i == drop_index) continue;
            AV_kept.col(ii) = AV_cache_.col(i);
            int jj = 0;
            for (int j = 0; j < old_rank; ++j) {
                if (j == drop_index) continue;
                A_r_kept(ii, jj) = A_r_cache(i, j);
                ++jj;
            }
            ++ii;
        }

        if (rank == kept_rank) {
            A_r_cache = std::move(A_r_kept);
            factorized_A_r.compute(A_r_cache);
            AV_cache_ = std::move(AV_kept);
        } else {
            append_column(A_r_kept, AV_kept, V.col(rank - 1), rank);
        }
        info.timing_projection += now_time() - t0;
        A_r_rank_cache_ = rank;
        fom_flag = 0;
        return true;
    }

    return false;
}

SolveStats MyStrategy2::main(const Eigen::SparseMatrix<double>& lhs,
                             const Eigen::VectorXd& rhs,
                             Eigen::VectorXd& u,
                             const ROMContext& context) {
    check_strategy_ready("MyStrategy2");

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
        const bool needs_projection = fom_flag || A_r_lhs_cache_ != &lhs ||
                                      A_r_rank_cache_ != rank || A_r_cache.rows() != rank;
        if (needs_projection) {
            bool cache_ok = false;
            const bool force_full_projection = env_int_or_default("STAGE2_MYSTRATEGY2_FULL_PROJECTION", 0) != 0;
            if (!force_full_projection && A_r_lhs_cache_ == &lhs) {
                cache_ok = update_reduced_cache_incremental(lhs, V, basis_runtime, info);
            }
            if (!cache_ok) {
                rebuild_reduced_cache(lhs, V, AV_cache_, A_r_lhs_cache_,
                                      A_r_rank_cache_, info);
            }
            else A_r_lhs_cache_ = &lhs;
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
            if (!coeff.allFinite()) use_fom = true;
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
        stats = fom_solver->FOM_main(lhs, rhs, u, context);
        solution_e = u;
        info.fom_used = true;
        info.ksp_iterations = stats.ksp_iterations;
        info.timing_fom_factorization += stats.factorization_time;
        info.timing_fom_solve += stats.solve_time;
        if (u_rom.size() == solution_e.size()) {
            //info.error = error_estimator->relative_solution_error(solution_e, u_rom);
            //info.basis_error = error_estimator->basis_error(V, solution_e);
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
        clear_reduced_cache(AV_cache_, A_r_lhs_cache_, A_r_rank_cache_);
        info.timing_basis_copy_reset += now_time() - tr;
    }

    info.timing_total = now_time() - total_t0;
    store_info(info, context);
    return stats;
}
