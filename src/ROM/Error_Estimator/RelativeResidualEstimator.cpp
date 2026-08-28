#include "ROM/Error_Estimator/RelativeResidualEstimator.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

#ifdef EIGEN_USE_LAPACKE
#include <mkl_cblas.h>
#include <mkl_spblas.h>
#endif

namespace {

double wall_seconds() {
    using Clock = std::chrono::steady_clock;
    static const auto t0 = Clock::now();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

} // namespace

ErrorAssessment RelativeResidualEstimator::assess(const Eigen::SparseMatrix<double>& lhs,
                                                  const Eigen::VectorXd& rhs,
                                                  const Eigen::VectorXd& solution,
                                                  const ROMBase::ROMContext&,
                                                  MklSparseMatrixHandle lhs_mkl) {
    const double rel_res = relative_residual(lhs, rhs, solution, lhs_mkl);
    ErrorAssessment assessment;
    assessment.estimator_name = name;
    assessment.error_indicator = rel_res;
    assessment.residual = rel_res;
    assessment.threshold = residual_tol;
    assessment.residual_tol_eff = residual_tol;
    assessment.accepted = rel_res <= residual_tol;
    return assessment;
}

double RelativeResidualEstimator::relative_residual(const Eigen::SparseMatrix<double>& lhs,
                                                   const Eigen::VectorXd& rhs,
                                                   const Eigen::VectorXd& solution,
                                                   MklSparseMatrixHandle lhs_mkl) const {
    const double t_mkl0 = wall_seconds();
    const double rel_res_mkl = mkl_relative_residual(lhs, rhs, solution, lhs_mkl);
    const double mkl_time = wall_seconds() - t_mkl0;
    const double rel_res = rel_res_mkl;

    // Compare with Eigen implementation for validation
    /* 
    const double t0 = wall_seconds();
    const double rel_res_eigen = eigen_relative_residual(lhs, rhs, solution);
    const double eigen_time = wall_seconds() - t0;

    std::cout << "  ROM residual compare[s]: eigen=" << eigen_time
              << " mkl=" << mkl_time
              << " eigen_residual=" << rel_res_eigen
              << " mkl_residual=" << rel_res_mkl
              << " diff=" << std::abs(rel_res_eigen - rel_res_mkl) << "\n";
    */
    (void)mkl_time;
    return rel_res;
}

double RelativeResidualEstimator::eigen_relative_residual(const Eigen::SparseMatrix<double>& lhs,
                                                         const Eigen::VectorXd& rhs,
                                                         const Eigen::VectorXd& solution) const {
    const double denom = std::max(rhs.norm(), 1e-30);
    return (lhs * solution - rhs).norm() / denom;
}

double RelativeResidualEstimator::relative_solution_error(const Eigen::VectorXd& fom_solution,
                                                         const Eigen::VectorXd& rom_solution) const {
    const double denom = std::max(fom_solution.norm(), 1e-30);
    return (fom_solution - rom_solution).norm() / denom;
}

double RelativeResidualEstimator::basis_error(const Eigen::MatrixXd& basis,
                                             const Eigen::VectorXd& solution) const {
    if (basis.cols() == 0) return 1.0;
    const Eigen::VectorXd projection = basis * (basis.transpose() * solution);
    return relative_solution_error(solution, projection);
}

double RelativeResidualEstimator::mkl_relative_residual(const Eigen::SparseMatrix<double>& lhs,
                                                       const Eigen::VectorXd& rhs,
                                                       const Eigen::VectorXd& solution,
                                                       MklSparseMatrixHandle lhs_mkl) const {
#ifdef EIGEN_USE_LAPACKE
    if (!lhs_mkl) return eigen_relative_residual(lhs, rhs, solution);

    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    Eigen::VectorXd y(rhs.size());
    const sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                                                   1.0,
                                                   lhs_mkl,
                                                   descr,
                                                   solution.data(),
                                                   0.0,
                                                   y.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("RelativeResidualEstimator: mkl_sparse_d_mv failed");
    }

    const MKL_INT n = static_cast<MKL_INT>(rhs.size());
    cblas_daxpy(n, -1.0, rhs.data(), 1, y.data(), 1);
    const double denom = std::max(cblas_dnrm2(n, rhs.data(), 1), 1e-30);
    return cblas_dnrm2(n, y.data(), 1) / denom;
#else
    (void)lhs_mkl;
    return eigen_relative_residual(lhs, rhs, solution);
#endif
}
