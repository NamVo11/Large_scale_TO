#include "ROM/ROM_method/FOM_iterative.hpp"

#include <Eigen/IterativeLinearSolvers>

#include <algorithm>
#include <chrono>
#include <utility>

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

} // namespace

FOM_iterative::FOM_iterative(std::string method_in) : method(std::move(method_in)) {
    name = "FOM_iterative";
    use_mor = false;
}

SolveStats FOM_iterative::main(const Eigen::SparseMatrix<double>& lhs,
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

SolveStats FOM_iterative::FOM_main(const Eigen::SparseMatrix<double>& lhs,
                                   const Eigen::VectorXd& rhs,
                                   Eigen::VectorXd& u,
                                   const ROMContext&) {
    SolveStats stats;
    if (rhs.norm() < 1e-30) {
        u = Eigen::VectorXd::Zero(rhs.size());
        stats.residual_norm = 0.0;
        return stats;
    }

    const double tf = now_time();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
    solver.setTolerance(pcg_tol);
    solver.setMaxIterations(pcg_maxit);
    solver.compute(lhs);
    stats.factorization_time = now_time() - tf;

    const double ts = now_time();
    u = solver.solve(rhs);
    stats.solve_time = now_time() - ts;

    stats.ksp_iterations = static_cast<int>(solver.iterations());
    stats.residual_norm = relative_residual(lhs, rhs, u);
    return stats;
}

