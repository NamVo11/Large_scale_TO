#include "ROM/Error_Estimator/RelativeResidualEstimator.hpp"

#include "ROM/Utils/rom_eigen_utils.hpp"


RelativeResidualEstimator::RelativeResidualEstimator(double residual_tol_in) {
    name = "RelativeResidualEstimator";
    residual_tol = residual_tol_in;
}

ErrorAssessment RelativeResidualEstimator::assess(const Eigen::SparseMatrix<double>& lhs,
                                                  const Eigen::VectorXd& rhs,
                                                  const Eigen::VectorXd& solution,
                                                  const ROMContext&) {
    const double rel_res = eigen_relative_residual(lhs, rhs, solution);
    ErrorAssessment assessment;
    assessment.estimator_name = name;
    assessment.error_indicator = rel_res;
    assessment.residual = rel_res;
    assessment.threshold = residual_tol;
    assessment.residual_tol_eff = residual_tol;
    assessment.accepted = rel_res <= residual_tol;
    return assessment;
}

