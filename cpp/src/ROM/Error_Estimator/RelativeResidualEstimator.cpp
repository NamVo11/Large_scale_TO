#include "ROM/Error_Estimator/RelativeResidualEstimator.hpp"

#include "ROM/Utils/rom_petsc_utils.hpp"

namespace stage2 {

RelativeResidualEstimator::RelativeResidualEstimator(double residual_tol_in) {
    name = "RelativeResidualEstimator";
    residual_tol = residual_tol_in;
}

ErrorAssessment RelativeResidualEstimator::assess(Mat lhs, Vec rhs, Vec solution, const ROMContext&) {
    const double rel_res = compute_relative_residual(lhs, rhs, solution);
    ErrorAssessment assessment;
    assessment.estimator_name = name;
    assessment.error_indicator = rel_res;
    assessment.residual = rel_res;
    assessment.threshold = residual_tol;
    assessment.residual_tol_eff = residual_tol;
    assessment.accepted = rel_res <= residual_tol;
    return assessment;
}

} // namespace stage2

