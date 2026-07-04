#pragma once

#include "stage2/MatlabLikeData.hpp"

namespace stage2 {

void dynamic_elastic_linear(Analysis& analysis,
                            const FEM& fem,
                            const std::vector<std::vector<double>>& y_elem,
                            int Iter = 0);

} // namespace stage2
