#include "Eigen_backend/ROM/ROM_method/MyStrategy.hpp"

#include "Eigen_backend/ROM/Utils/rom_eigen_utils.hpp"
#include "ROM/Common/Reduced_Basis_Agorithm/MyIncrementalQR1.hpp"
#include "ROM/Common/Reduced_Basis_Agorithm/UpdateBase.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <iostream>

namespace eigen_backend {
namespace {

double now_time() {
    using Clock = std::chrono::steady_clock;
    static const auto t0 = Clock::now();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int env_int_or_default(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    return std::atoi(value);
}

void append_basis_debug(const char* phase,
                        const ROMContext& context,
                        int basis_id,
                        const UpdateBase& basis) {
    const char* dir = std::getenv("STAGE2_ROM_DEBUG_DIR");
    if (dir == nullptr || *dir == '\0') return;

    std::ostringstream path;
    path << dir << "/cpp_rom_basis_debug.csv";
    std::ifstream existing(path.str());
    const bool needs_header = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
    existing.close();

    std::ofstream out(path.str(), std::ios::app);
    if (!out) return;
    if (needs_header) out << "phase,role,iter,step,basis_id,rank,singular_values\n";

    out << phase << ',' << context.role << ',' << context.opt_iter << ','
        << context.time_step << ',' << basis_id << ',' << basis.V.cols() << ',';
    out << std::setprecision(17);
    for (Eigen::Index i = 0; i < basis.s.size(); ++i) {
        if (i > 0) out << ';';
        out << basis.s[i];
    }
    out << '\n';
}

void append_decision_debug(const ROMContext& context,
                           int rank,
                           double residual,
                           double threshold,
                           bool accepted) {
    const char* dir = std::getenv("STAGE2_ROM_DECISION_DEBUG_DIR");
    if (dir == nullptr || *dir == '\0') return;

    std::ostringstream path;
    path << dir << "/cpp_rom_decision_debug.csv";
    std::ifstream existing(path.str());
    const bool needs_header = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
    existing.close();

    std::ofstream out(path.str(), std::ios::app);
    if (!out) return;
    if (needs_header) out << "role,iter,step,rank,residual,threshold,accepted\n";
    out << std::setprecision(17)
        << context.role << ',' << context.opt_iter << ',' << context.time_step << ','
        << rank << ',' << residual << ',' << threshold << ',' << (accepted ? 1 : 0) << '\n';
}

} // namespace

MyStrategy::MyStrategy(std::vector<std::shared_ptr<UpdateBase>> basis_list_in,
                       double residual_tol_in,
                       int n_p_in,
                       std::shared_ptr<ROMBase> fom_solver_in,
                       std::shared_ptr<ErrorEstimatorBase> error_estimator_in,
                       bool online_in)
    : ROMBase(std::move(basis_list_in)),
      residual_tol(residual_tol_in),
      n_p(n_p_in),
      fom_solver(std::move(fom_solver_in)),
      error_estimator(std::move(error_estimator_in)),
      online(online_in) {
    name = "MyStrategy";
}

std::shared_ptr<FactorizedSolver> MyStrategy::get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const {
    if (!fom_solver) return nullptr;
    return fom_solver->get_factorized_solver(lhs);
}

void MyStrategy::set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                                       std::shared_ptr<FactorizedSolver> solver) {
    if (!fom_solver) throw std::runtime_error("MyStrategy needs fom_solver");
    fom_solver->set_factorized_solver(lhs, std::move(solver));
}

void MyStrategy::clear_reduced_cache() {
    A_r_cache.resize(0, 0);
    AV_cache_.resize(0, 0);
    A_r_lhs_cache_ = nullptr;
    A_r_rank_cache_ = 0;
}

void MyStrategy::rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                                       const Eigen::MatrixXd& V,
                                       ROMInfo& info) {
    const double t0 = now_time();
    AV_cache_.noalias() = A * V;
    A_r_cache.noalias() = V.transpose() * AV_cache_;
    factorized_A_r.compute(A_r_cache);

    info.timing_projection += now_time() - t0;
    A_r_lhs_cache_ = &A;
    A_r_rank_cache_ = static_cast<int>(V.cols());
    fom_flag = 0;
}

SolveStats MyStrategy::main(const Eigen::SparseMatrix<double>& lhs,
                            const Eigen::VectorXd& rhs,
                            Eigen::VectorXd& u,
                            const ROMContext& context) {
    if (!fom_solver || !error_estimator) throw std::runtime_error("MyStrategy is not fully initialized");
    if (basis_list.size() < 2) throw std::runtime_error("MyStrategy expects basis_list{1} and basis_list{2}");

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
        if (rebuild_projection) rebuild_reduced_cache(lhs, V, info);

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
            const ErrorAssessment assessment = error_estimator->assess(lhs, rhs, u_rom, context);
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
        info.residual_norm = stats.residual_norm;
        info.timing_fom_factorization += stats.factorization_time;
        info.timing_fom_solve += stats.solve_time;
        if (u_rom.size() == solution_e.size()) {
            info.error = eigen_relative_solution_error(solution_e, u_rom);
            info.basis_error = eigen_basis_error(V, solution_e);
        }
        if (online) {
            const double tu = now_time();
            basis_list[1]->update(solution_e, context);
            append_basis_debug("online_update_basis2", context, 2, *basis_list[1]);
            const double dt = now_time() - tu;
            info.timing_online_basis_update += dt;
            info.timing_basis_update += dt;
        }
        fom_flag = 1;
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
        fom_flag = 1;
        clear_reduced_cache();
        info.timing_basis_copy_reset += now_time() - tr;
    }

    info.timing_total = now_time() - total_t0;
    store_info(info, context);
    return stats;
}

MyStrategy2::MyStrategy2(std::vector<std::shared_ptr<UpdateBase>> basis_list_in,
                         double residual_tol_in,
                         int n_p_in,
                         std::shared_ptr<ROMBase> fom_solver_in,
                         std::shared_ptr<ErrorEstimatorBase> error_estimator_in,
                         bool online_in)
    : MyStrategy(std::move(basis_list_in), residual_tol_in, n_p_in,
                 std::move(fom_solver_in), std::move(error_estimator_in), online_in) {
    name = "MyStrategy2";
}

std::shared_ptr<FactorizedSolver> MyStrategy2::get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const {
    if (!fom_solver) return nullptr;
    return fom_solver->get_factorized_solver(lhs);
}

void MyStrategy2::set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                                        std::shared_ptr<FactorizedSolver> solver) {
    if (!fom_solver) throw std::runtime_error("MyStrategy2 needs fom_solver");
    fom_solver->set_factorized_solver(lhs, std::move(solver));
}

void MyStrategy2::clear_reduced_cache2() {
    A_r_cache.resize(0, 0);
    AV_cache2_.resize(0, 0);
    A_r_lhs_cache2_ = nullptr;
    A_r_rank_cache2_ = 0;
}

void MyStrategy2::rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                                        const Eigen::MatrixXd& V,
                                        ROMInfo& info) {
    const double t0 = now_time();
    AV_cache2_.noalias() = A * V;
    A_r_cache.noalias() = V.transpose() * AV_cache2_;
    factorized_A_r.compute(A_r_cache);
    info.timing_projection += now_time() - t0;
    A_r_lhs_cache2_ = &A;
    A_r_rank_cache2_ = static_cast<int>(V.cols());
    fom_flag = 0;
}

bool MyStrategy2::update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                                   const Eigen::MatrixXd& V,
                                                   const UpdateBase& basis_runtime,
                                                   ROMInfo& info) {
    const int rank = static_cast<int>(V.cols());
    const int old_rank = A_r_rank_cache2_;
    if (rank <= 0 || A_r_cache.rows() != old_rank || AV_cache2_.cols() != old_rank) return false;

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
        AV_cache2_ = std::move(AV_new);
    };

    if (rank == old_rank + 1) {
        append_column(A_r_cache, AV_cache2_, V.col(rank - 1), rank);
        info.timing_projection += now_time() - t0;
        A_r_rank_cache2_ = rank;
        fom_flag = 0;
        return true;
    }

    if (qr_basis && drop_index >= 0 && drop_index < old_rank &&
        (rank == old_rank || rank == old_rank - 1)) {
        const int kept_rank = old_rank - 1;
        Eigen::MatrixXd A_r_kept(kept_rank, kept_rank);
        Eigen::MatrixXd AV_kept(AV_cache2_.rows(), kept_rank);
        int ii = 0;
        for (int i = 0; i < old_rank; ++i) {
            if (i == drop_index) continue;
            AV_kept.col(ii) = AV_cache2_.col(i);
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
            AV_cache2_ = std::move(AV_kept);
        } else {
            append_column(A_r_kept, AV_kept, V.col(rank - 1), rank);
        }
        info.timing_projection += now_time() - t0;
        A_r_rank_cache2_ = rank;
        fom_flag = 0;
        return true;
    }

    return false;
}

SolveStats MyStrategy2::main(const Eigen::SparseMatrix<double>& lhs,
                             const Eigen::VectorXd& rhs,
                             Eigen::VectorXd& u,
                             const ROMContext& context) {
    if (!fom_solver || !error_estimator) throw std::runtime_error("MyStrategy2 is not fully initialized");
    if (basis_list.size() < 2) throw std::runtime_error("MyStrategy2 expects basis_list{1} and basis_list{2}");

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
        const bool needs_projection = fom_flag || A_r_lhs_cache2_ != &lhs ||
                                      A_r_rank_cache2_ != rank || A_r_cache.rows() != rank;
        if (needs_projection) {
            bool cache_ok = false;
            const bool force_full_projection = env_int_or_default("STAGE2_MYSTRATEGY2_FULL_PROJECTION", 0) != 0;
            if (!force_full_projection && A_r_lhs_cache2_ == &lhs) {
                cache_ok = update_reduced_cache_incremental(lhs, V, basis_runtime, info);
            }
            if (!cache_ok) rebuild_reduced_cache(lhs, V, info);
            else A_r_lhs_cache2_ = &lhs;
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
            const ErrorAssessment assessment = error_estimator->assess(lhs, rhs, u_rom, context);
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
        info.residual_norm = stats.residual_norm;
        info.timing_fom_factorization += stats.factorization_time;
        info.timing_fom_solve += stats.solve_time;
        if (u_rom.size() == solution_e.size()) {
            info.error = eigen_relative_solution_error(solution_e, u_rom);
            info.basis_error = eigen_basis_error(V, solution_e);
        }
        if (online) {
            const double tu = now_time();
            basis_list[1]->update(solution_e, context);
            append_basis_debug("online_update_basis2", context, 2, *basis_list[1]);
            const double dt = now_time() - tu;
            info.timing_online_basis_update += dt;
            info.timing_basis_update += dt;
        }
        fom_flag = 1;
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
        clear_reduced_cache2();
        info.timing_basis_copy_reset += now_time() - tr;
    }

    info.timing_total = now_time() - total_t0;
    store_info(info, context);
    return stats;
}

} // namespace eigen_backend