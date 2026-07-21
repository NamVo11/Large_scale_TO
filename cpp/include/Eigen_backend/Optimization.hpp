#pragma once

#include "MatlabLikeData.hpp"

#include <vector>

namespace eigen_backend {

void optimization11(const FEM& fem,
                    std::vector<Analysis>& analysis_all,
                    Optimization& Opt,
                    const OutputData& Output);

} // namespace eigen_backend
