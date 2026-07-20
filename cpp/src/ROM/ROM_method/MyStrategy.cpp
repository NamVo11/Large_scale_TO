#include "ROM/ROM_method/MyStrategy.hpp"

#include "ROM/Reduced_Basis_Agorithm/UpdateBase.hpp"
#include "ROM/Reduced_Basis_Agorithm/MyIncrementalQR1.hpp"
#include "helper/PetscEigenBridge.hpp"

#include <petscsys.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

double now_time() {
    PetscLogDouble t = 0.0;
    PetscTime(&t);
    return static_cast<double>(t);
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
    if (needs_header) {
        out << "phase,role,iter,step,basis_id,rank,singular_values\n";
    }

    out << phase << ','
        << context.role << ','
        << context.opt_iter << ','
        << context.time_step << ','
        << basis_id << ','
        << basis.V.cols() << ',';

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
    if (needs_header) {
        out << "role,iter,step,rank,residual,threshold,accepted\n";
    }

    out << std::setprecision(17)
        << context.role << ','
        << context.opt_iter << ','
        << context.time_step << ','
        << rank << ','
        << residual << ','
        << threshold << ','
        << (accepted ? 1 : 0) << '\n';
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

KSP MyStrategy::get_factorized_solver(Mat lhs) const {
    if (!fom_solver) return nullptr;
    return fom_solver->get_factorized_solver(lhs);
}

void MyStrategy::set_factorized_solver(Mat lhs, KSP ksp) {
    if (!fom_solver) throw std::runtime_error("MyStrategy needs fom_solver");
    fom_solver->set_factorized_solver(lhs, ksp);
}

void MyStrategy::clear_reduced_cache() {
    A_r_cache.resize(0, 0);
    A_r_lhs_cache_ = nullptr;
    A_r_rank_cache_ = 0;
}

const Eigen::SparseMatrix<double>& MyStrategy::eigen_lhs(Mat lhs, ROMInfo& info) {
    if (eigen_lhs_cache_ != lhs) {
        const double t0 = now_time();
        eigen_lhs_cache_matrix_ = petsc_mat_to_eigen(lhs);
        info.timing_petsc_to_eigen_mat += now_time() - t0;
        eigen_lhs_cache_ = lhs;
        clear_reduced_cache();
    }
    return eigen_lhs_cache_matrix_;
}

SolveStats MyStrategy::main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) {
    if (!fom_solver || !error_estimator) throw std::runtime_error("MyStrategy is not fully initialized");
    if (basis_list.size() < 2) throw std::runtime_error("MyStrategy expects basis_list{1} and basis_list{2}");

    const double total_t0 = now_time();
    ROMInfo info;
    info.name = name;

    const auto& basis_runtime = *basis_list[1];
    const Eigen::MatrixXd& V = basis_runtime.getBasis(context);
    const int rank = static_cast<int>(V.cols());
    info.rank = rank;

    const double vec_t0 = now_time();
    Eigen::VectorXd rhs_e = petsc_vec_to_eigen(rhs);
    info.timing_petsc_to_eigen_vec += now_time() - vec_t0;

    const Eigen::SparseMatrix<double>& A_e = eigen_lhs(lhs, info);

    bool use_fom = !use_mor || rank == 0;
    Eigen::VectorXd u_rom;
    SolveStats stats;

    if (!use_fom) {
        const bool rebuild_projection = fom_flag ||
                                        A_r_lhs_cache_ != lhs ||
                                        A_r_rank_cache_ != rank;
        if (rebuild_projection) {
            const double t0 = now_time();
            const Eigen::MatrixXd AV = A_e * V;
            A_r_cache = V.transpose() * AV;
            factorized_A_r.compute(A_r_cache);
            info.timing_projection += now_time() - t0;
            A_r_lhs_cache_ = lhs;
            A_r_rank_cache_ = rank;
            fom_flag = 0;
        }

        Eigen::VectorXd b_r;
        {
            const double t0 = now_time();
            b_r = V.transpose() * rhs_e;
            info.timing_reduced_rhs += now_time() - t0;
        }

        Eigen::VectorXd coeff;
        {
            const double t0 = now_time();
            /*
            Eigen::PartialPivLU<Eigen::MatrixXd> lu(A_r_cache);
            if (A_r_cache.rows() == 0 || A_r_cache.cols() == 0) {
                use_fom = true;
                fom_flag = 1;
            } else {
                coeff = lu.solve(b_r);
                if (!coeff.allFinite()) {
                    use_fom = true;
                    fom_flag = 1;
                }
            }
            */
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
            info.timing_reduced_solve += now_time() - t0;
        }

        if (!use_fom) {
            const double t0 = now_time();
            u_rom = V * coeff;
            info.timing_reconstruct += now_time() - t0;

            const double tr = now_time();
            const double rel_res = eigen_relative_residual(A_e, rhs_e, u_rom);
            info.timing_residual += now_time() - tr;
            info.rank = rank;
            info.rank_true = rank;
            info.error_indicator = rel_res;
            info.residual = rel_res;
            info.residual_norm = rel_res;
            info.threshold = residual_tol;
            info.residual_tol_eff = residual_tol;
            info.accepted = rel_res <= residual_tol;
            append_decision_debug(context, rank, rel_res, residual_tol, info.accepted);
            use_fom = !info.accepted;
            stats.ksp_iterations = 0;
            stats.residual_norm = rel_res;
            if (use_fom) fom_flag = 1;
        }
    }

    Eigen::VectorXd solution_e;
    if (use_fom) {
        fom_flag = 1;
        // const double tf = now_time();
        stats = fom_solver->FOM_main(lhs, rhs, u, context);
        //info.timing_fom_fallback += now_time() - tf;
        info.timing_fom_factorization += stats.factorization_time;
        info.timing_fom_solve += stats.solve_time;
        info.fom_used = true;
        info.ksp_iterations = stats.ksp_iterations;
        info.residual_norm = stats.residual_norm;

        const double tv = now_time();
        solution_e = petsc_vec_to_eigen(u);
        info.timing_petsc_to_eigen_vec += now_time() - tv;

        if (u_rom.size() == solution_e.size()) {
            info.error = eigen_relative_solution_error(solution_e, u_rom);
            info.basis_error = eigen_basis_error(V, solution_e);
        }
        if (online) {
            const double tb = now_time();
            basis_list[1]->update(solution_e, context);
            append_basis_debug("online_update_basis2", context, 2, *basis_list[1]);
            const double dt = now_time() - tb;
            info.timing_online_basis_update += dt;
            info.timing_basis_update += dt;
        }
        fom_flag = 1;
    } else {
        const double tw = now_time();
        eigen_to_petsc_vec(u_rom, u);
        info.timing_reconstruct += now_time() - tw;
        solution_e = std::move(u_rom);
        info.rom_used = true;
        info.used_rom = true;
        info.accepted = true;
        fom_flag = 0;
    }

    {
        const double ts = now_time();
        basis_list[0]->addSnapshot(solution_e, context);
        info.timing_snapshot += now_time() - ts;
    }

    if (context.is_end_of_analysis) {
        if (basis_list[0]->max_rank != 0) {
            const double tu = now_time();
            basis_list[0]->update(solution_e, context);
            append_basis_debug("offline_update_basis1", context, 1, *basis_list[0]);
            const double update_dt = now_time() - tu;
            info.timing_basis1_update += update_dt;
            info.timing_end_basis_update += update_dt;
            info.timing_basis_update += update_dt;

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
    : MyStrategy(std::move(basis_list_in),
                 residual_tol_in,
                 n_p_in,
                 std::move(fom_solver_in),
                 std::move(error_estimator_in),
                 online_in) {
    name = "MyStrategy2";
}

KSP MyStrategy2::get_factorized_solver(Mat lhs) const {
    if (!fom_solver) return nullptr;
    return fom_solver->get_factorized_solver(lhs);
}

void MyStrategy2::set_factorized_solver(Mat lhs, KSP ksp) {
    if (!fom_solver) throw std::runtime_error("MyStrategy2 needs fom_solver");
    fom_solver->set_factorized_solver(lhs, ksp);
}

void MyStrategy2::clear_reduced_cache2() {
    A_r_cache.resize(0, 0);
    AV_cache2_.resize(0, 0);
    A_r_lhs_cache2_ = nullptr;
    A_r_rank_cache2_ = 0;
}

const Eigen::SparseMatrix<double>& MyStrategy2::eigen_lhs2(Mat lhs, ROMInfo& info) {
    if (eigen_lhs_cache2_ != lhs) {
        const double t0 = now_time();
        eigen_lhs_cache_matrix2_ = petsc_mat_to_eigen(lhs);
        info.timing_petsc_to_eigen_mat += now_time() - t0;
        eigen_lhs_cache2_ = lhs;
        clear_reduced_cache2();
    }
    return eigen_lhs_cache_matrix2_;
}

void MyStrategy2::rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A_e,
                                        const Eigen::MatrixXd& V,
                                        Mat lhs,
                                        ROMInfo& info) {
    const double t0 = now_time();
    AV_cache2_ = A_e * V;
    A_r_cache = V.transpose() * AV_cache2_;
    factorized_A_r.compute(A_r_cache);
    info.timing_projection += now_time() - t0;
    A_r_lhs_cache2_ = lhs;
    A_r_rank_cache2_ = static_cast<int>(V.cols());
    fom_flag = 0;
}

bool MyStrategy2::update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A_e,
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
        const Eigen::VectorXd Av_new = A_e * v_new;
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

SolveStats MyStrategy2::main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) {
    if (!fom_solver || !error_estimator) throw std::runtime_error("MyStrategy2 is not fully initialized");
    if (basis_list.size() < 2) throw std::runtime_error("MyStrategy2 expects basis_list{1} and basis_list{2}");

    const double total_t0 = now_time();
    ROMInfo info;
    info.name = name;

    const auto& basis_runtime = *basis_list[1];
    const Eigen::MatrixXd& V = basis_runtime.getBasis(context);
    const int rank = static_cast<int>(V.cols());
    info.rank = rank;

    const double vec_t0 = now_time();
    Eigen::VectorXd rhs_e = petsc_vec_to_eigen(rhs);
    info.timing_petsc_to_eigen_vec += now_time() - vec_t0;

    const Eigen::SparseMatrix<double>& A_e = eigen_lhs2(lhs, info);

    bool use_fom = !use_mor || rank == 0;
    Eigen::VectorXd u_rom;
    SolveStats stats;

    if (!use_fom) {
        const bool needs_projection = fom_flag ||
                                      A_r_lhs_cache2_ != lhs ||
                                      A_r_rank_cache2_ != rank ||
                                      A_r_cache.rows() != rank;
        if (needs_projection) {
            bool cache_ok = false;
            const bool force_full_projection =
                env_int_or_default("STAGE2_MYSTRATEGY2_FULL_PROJECTION", 0) != 0;
            if (!force_full_projection && A_r_lhs_cache2_ == lhs) {
                cache_ok = update_reduced_cache_incremental(A_e, V, basis_runtime, info);
            }
            if (!cache_ok) {
                rebuild_reduced_cache(A_e, V, lhs, info);
            } else {
                A_r_lhs_cache2_ = lhs;
            }
        }

        Eigen::VectorXd b_r;
        {
            const double t0 = now_time();
            b_r = V.transpose() * rhs_e;
            info.timing_reduced_rhs += now_time() - t0;
        }

        Eigen::VectorXd coeff;
        {
            const double t0 = now_time();
            /*
            Eigen::PartialPivLU<Eigen::MatrixXd> lu(A_r_cache);
            if (A_r_cache.rows() == 0 || A_r_cache.cols() == 0) {
                use_fom = true;
            } else {
                coeff = lu.solve(b_r);
                if (!coeff.allFinite()) use_fom = true;
            }
            */
            if (A_r_cache.rows() == 0 || A_r_cache.cols() == 0) {
                use_fom = true;
            } else {
                coeff = factorized_A_r.solve(b_r);
                if (!coeff.allFinite()) use_fom = true;
            }
            info.timing_reduced_solve += now_time() - t0;
        }

        if (!use_fom) {
            const double t0 = now_time();
            u_rom = V * coeff;
            info.timing_reconstruct += now_time() - t0;

            const double tr = now_time();
            const double rel_res = eigen_relative_residual(A_e, rhs_e, u_rom);
            info.timing_residual += now_time() - tr;
            info.rank = rank;
            info.rank_true = rank;
            info.error_indicator = rel_res;
            info.residual = rel_res;
            info.residual_norm = rel_res;
            info.threshold = residual_tol;
            info.residual_tol_eff = residual_tol;
            info.accepted = rel_res <= residual_tol;
            append_decision_debug(context, rank, rel_res, residual_tol, info.accepted);
            use_fom = !info.accepted;
            stats.ksp_iterations = 0;
            stats.residual_norm = rel_res;
        }
    }

    Eigen::VectorXd solution_e;
    if (use_fom) {
        stats = fom_solver->FOM_main(lhs, rhs, u, context);
        info.timing_fom_factorization += stats.factorization_time;
        info.timing_fom_solve += stats.solve_time;
        info.fom_used = true;
        info.ksp_iterations = stats.ksp_iterations;
        info.residual_norm = stats.residual_norm;

        const double tv = now_time();
        solution_e = petsc_vec_to_eigen(u);
        info.timing_petsc_to_eigen_vec += now_time() - tv;

        if (u_rom.size() == solution_e.size()) {
            info.error = eigen_relative_solution_error(solution_e, u_rom);
            info.basis_error = eigen_basis_error(V, solution_e);
        }
        if (online) {
            const double tb = now_time();
            basis_list[1]->update(solution_e, context);
            append_basis_debug("online_update_basis2", context, 2, *basis_list[1]);
            const double dt = now_time() - tb;
            info.timing_online_basis_update += dt;
            info.timing_basis_update += dt;
        }
        fom_flag = 1;
    } else {
        const double tw = now_time();
        eigen_to_petsc_vec(u_rom, u);
        info.timing_reconstruct += now_time() - tw;
        solution_e = std::move(u_rom);
        info.rom_used = true;
        info.used_rom = true;
        info.accepted = true;
        fom_flag = 0;
    }

    {
        const double ts = now_time();
        basis_list[0]->addSnapshot(solution_e, context);
        info.timing_snapshot += now_time() - ts;
    }

    if (context.is_end_of_analysis) {
        if (basis_list[0]->max_rank != 0) {
            const double tu = now_time();
            basis_list[0]->update(solution_e, context);
            append_basis_debug("offline_update_basis1", context, 1, *basis_list[0]);
            const double update_dt = now_time() - tu;
            info.timing_basis1_update += update_dt;
            info.timing_end_basis_update += update_dt;
            info.timing_basis_update += update_dt;

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

