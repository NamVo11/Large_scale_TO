#pragma once

#include "Eigen_backend/ROM/Error_Estimator/ErrorEstimatorBase.hpp"

namespace eigen_backend {

class RelativeResidualEstimator final : public ErrorEstimatorBase {
public:
    double residual_tol = 1e-3;

    explicit RelativeResidualEstimator(double residual_tol_in);
    ErrorAssessment assess(const Eigen::SparseMatrix<double>& lhs,
                           const Eigen::VectorXd& rhs,
                           const Eigen::VectorXd& solution,
                           const ROMContext& context) override;
};

} // namespace eigen_backend
