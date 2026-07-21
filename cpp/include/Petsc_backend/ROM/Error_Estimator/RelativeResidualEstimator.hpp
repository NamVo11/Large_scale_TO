#pragma once

#include "Petsc_backend/ROM/Error_Estimator/ErrorEstimatorBase.hpp"

namespace petsc_backend {

class RelativeResidualEstimator final : public ErrorEstimatorBase {
public:
    double residual_tol = 1e-3;

    explicit RelativeResidualEstimator(double residual_tol_in);
    ErrorAssessment assess(Mat lhs, Vec rhs, Vec solution, const ROMContext& context) override;
};

} // namespace petsc_backend
