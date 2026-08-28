#include "ROM/ROM_method/FOM.hpp"

#include <Eigen/IterativeLinearSolvers>

#include <algorithm>
#include <chrono>
#include <utility>

namespace {

double wall_time() {
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

EigenFOM_iterative::EigenFOM_iterative(std::string method_in) : method(std::move(method_in)) {
    name = "EigenFOM_iterative";
    use_mor = false;
}

SolveStats EigenFOM_iterative::main(const Eigen::SparseMatrix<double>& lhs,
                                    const Eigen::VectorXd& rhs,
                                    Eigen::VectorXd& u) {
    const ROMContext& context = ctx;
    const double t0 = wall_time();
    const SolveStats stats = FOM_main(lhs, rhs, u);
    ROMInfo info;
    info.name = name;
    info.fom_used = true;
    info.accepted = true;
    info.ksp_iterations = stats.ksp_iterations;
    info.residual_norm = stats.residual_norm;
    info.timing_fom_factorization = stats.factorization_time;
    info.timing_fom_solve = stats.solve_time;
    info.timing_total = wall_time() - t0;
    store_info(info, context);
    return stats;
}

SolveStats EigenFOM_iterative::FOM_main(const Eigen::SparseMatrix<double>& lhs,
                                        const Eigen::VectorXd& rhs,
                                        Eigen::VectorXd& u) {
    SolveStats stats;
    if (rhs.norm() < 1e-30) {
        u = Eigen::VectorXd::Zero(rhs.size());
        stats.residual_norm = 0.0;
        return stats;
    }

    const double tf = wall_time();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
    solver.setTolerance(pcg_tol);
    solver.setMaxIterations(pcg_maxit);
    solver.compute(lhs);
    stats.factorization_time = wall_time() - tf;

    const double ts = wall_time();
    u = solver.solve(rhs);
    stats.solve_time = wall_time() - ts;

    stats.ksp_iterations = static_cast<int>(solver.iterations());
    stats.residual_norm = relative_residual(lhs, rhs, u);
    return stats;
}
