#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

namespace stage2 {

struct ErrorAssessment {
    std::string estimator_name;
    double error_indicator = 0.0;
    double residual = 0.0;
    double threshold = 0.0;
    double residual_tol_eff = 0.0;
    bool accepted = false;
};

class ErrorEstimatorBase {
public:
    std::string name = "ErrorEstimatorBase";
    virtual ~ErrorEstimatorBase() = default;
    virtual ErrorAssessment assess(Mat lhs, Vec rhs, Vec solution, const ROMContext& context) = 0;
};

} // namespace stage2

