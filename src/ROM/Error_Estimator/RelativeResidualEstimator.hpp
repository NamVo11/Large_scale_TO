#pragma once

#include "ROM/Error_Estimator/ErrorEstimatorBase.hpp"

class RelativeResidualEstimator final : public ErrorEstimatorBase {
public:
    double residual_tol = 1e-3;

    explicit RelativeResidualEstimator(double residual_tol_in) {
        name = "RelativeResidualEstimator";
        residual_tol = residual_tol_in;
    }

    ~RelativeResidualEstimator() override = default;

    ErrorAssessment assess(const Eigen::SparseMatrix<double>& lhs,
                           const Eigen::VectorXd& rhs,
                           const Eigen::VectorXd& solution,
                           const ROMContext& context,
                           MklSparseMatrixHandle lhs_mkl = nullptr) override;

    double relative_residual(const Eigen::SparseMatrix<double>& lhs,
                             const Eigen::VectorXd& rhs,
                             const Eigen::VectorXd& solution,
                             MklSparseMatrixHandle lhs_mkl = nullptr) const override;
    double eigen_relative_residual(const Eigen::SparseMatrix<double>& lhs,
                                   const Eigen::VectorXd& rhs,
                                   const Eigen::VectorXd& solution) const override;
    double mkl_relative_residual(const Eigen::SparseMatrix<double>& lhs,
                                 const Eigen::VectorXd& rhs,
                                 const Eigen::VectorXd& solution,
                                 MklSparseMatrixHandle lhs_mkl) const override;
    double relative_solution_error(const Eigen::VectorXd& fom_solution,
                                   const Eigen::VectorXd& rom_solution) const override;
    double basis_error(const Eigen::MatrixXd& basis,
                       const Eigen::VectorXd& solution) const override;
};
