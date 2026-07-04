#pragma once

#include "ROM/Error_Estimator/ErrorEstimatorBase.hpp"

namespace stage2 {

class RelativeResidualEstimator final : public ErrorEstimatorBase {
public:
    double residual_tol = 1e-3;

    explicit RelativeResidualEstimator(double residual_tol_in);
    ErrorAssessment assess(Mat lhs, Vec rhs, Vec solution, const ROMContext& context) override;
};

} // namespace stage2

