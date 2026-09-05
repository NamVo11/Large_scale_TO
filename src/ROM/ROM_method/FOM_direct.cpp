#include "ROM/ROM_method/FOM_direct.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#if defined(__has_include)
#if __has_include(<mkl_service.h>)
#include <mkl_service.h>
#define STAGE2_EIGEN_HAS_MKL_SERVICE 1
#endif
#endif

namespace {

double now_time() {
    using Clock = std::chrono::steady_clock;
    static const auto t0 = Clock::now();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

double relative_residual(const Eigen::SparseMatrix<double>& A,
                         const Eigen::VectorXd& rhs,
                         const Eigen::VectorXd& u) {
    const double denom = std::max(1e-30, rhs.norm());
    return (A * u - rhs).norm() / denom;
}

int env_int_or_default(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    return std::max(1, std::atoi(value));
}

bool env_flag_or_default(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    return std::atoi(value) != 0;
}

class MKLThreadGuard {
public:
    MKLThreadGuard(int factor_threads, int runtime_threads)
        : runtime_threads_(std::max(1, runtime_threads)) {
#if defined(STAGE2_EIGEN_HAS_MKL_SERVICE)
        mkl_set_num_threads_local(std::max(1, factor_threads));
#else
        (void)factor_threads;
#endif
    }

    ~MKLThreadGuard() {
#if defined(STAGE2_EIGEN_HAS_MKL_SERVICE)
        mkl_set_num_threads_local(runtime_threads_);
#endif
    }

private:
    int runtime_threads_ = 1;
};

} // namespace

FOM_direct::FOM_direct() {
    name = "FOM_direct";
    use_mor = false;
}

SolveStats FOM_direct::main(const Eigen::SparseMatrix<double>& lhs,
                            const Eigen::VectorXd& rhs,
                            Eigen::VectorXd& u,
                            const ROMContext& context) {
    const double t0 = now_time();
    const SolveStats stats = FOM_main(lhs, rhs, u, context);
    ROMInfo info;
    info.name = name;
    info.fom_used = true;
    info.accepted = true;
    info.ksp_iterations = stats.ksp_iterations;
    info.residual_norm = stats.residual_norm;
    info.timing_fom_factorization = stats.factorization_time;
    info.timing_fom_solve = stats.solve_time;
    info.timing_total = now_time() - t0;
    store_info(info, context);
    return stats;
}

void FOM_direct::set_factorization(const Eigen::SparseMatrix<double>& lhs) {
    const double tf = now_time();
    const int requested_factor_threads = env_int_or_default("FACTOR_NUM_THREADS", env_int_or_default("MKL_NUM_THREADS", 1));
    const int requested_runtime_threads = env_int_or_default("MKL_NUM_THREADS", 1);
    MKLThreadGuard mkl_threads(requested_factor_threads, requested_runtime_threads);
#if defined(STAGE2_EIGEN_HAS_MKL_SERVICE)
    std::cout << "  thread_debug factorization: requested_factor_threads="
              << requested_factor_threads
              << " requested_runtime_threads=" << requested_runtime_threads
              << " mkl_get_max_threads=" << mkl_get_max_threads()
              << "\n";
#else
    std::cout << "  thread_debug factorization: MKL service header not available"
              << " requested_factor_threads=" << requested_factor_threads
              << " requested_runtime_threads=" << requested_runtime_threads
              << "\n";
#endif
    solver_ = std::make_shared<FactorizedSolver>();
#ifdef EIGEN_USE_LAPACKE
    solver_->solver.pardisoParameterArray()[7] = 1;
#endif
    solver_->solver.compute(lhs);
    last_factorization_time_ = now_time() - tf;
    if (solver_->solver.info() != Eigen::Success) {
        solver_.reset();
        factor_ = nullptr;
        throw std::runtime_error("Eigen FOM_direct factorization failed");
    }
    factor_ = &lhs;
}

std::shared_ptr<FactorizedSolver> FOM_direct::get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const {
    if (factor_ == &lhs && solver_) return solver_;
    return nullptr;
}

void FOM_direct::set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                                       std::shared_ptr<FactorizedSolver> solver) {
    solver_ = std::move(solver);
    factor_ = solver_ ? &lhs : nullptr;
}

SolveStats FOM_direct::FOM_main(const Eigen::SparseMatrix<double>& lhs,
                                const Eigen::VectorXd& rhs,
                                Eigen::VectorXd& u,
                                const ROMContext& context) {
    SolveStats stats;
    if (rhs.norm() < 1e-30) {
        u = Eigen::VectorXd::Zero(rhs.size());
        stats.residual_norm = 0.0;
        return stats;
    }

    if (!solver_ || factor_ != &lhs) {
        set_factorization(lhs);
        stats.factorization_time = last_factorization_time_;
        std::cout << "  factorization[s]: " << context.role
                  << " " << context.system_type << " = "
                  << stats.factorization_time << "\n";
    }

    const double ts = now_time();
    u = solver_->solver.solve(rhs);
    stats.solve_time = now_time() - ts;
    if (solver_->solver.info() != Eigen::Success) throw std::runtime_error("Eigen FOM_direct solve failed");

    stats.ksp_iterations = 1;
    if (env_flag_or_default("STAGE2_EIGEN_CHECK_FOM_RESIDUAL", false)) {
        const double tr = now_time();
        stats.residual_norm = relative_residual(lhs, rhs, u);
        std::cout << "  eigen_fom_residual_check[s]: "
                  << now_time() - tr << "\n";
    } else {
        stats.residual_norm = 0.0;
    }
    return stats;
}

