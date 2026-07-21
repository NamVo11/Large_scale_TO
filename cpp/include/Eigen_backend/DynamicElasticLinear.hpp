#pragma once

#include "MatlabLikeData.hpp"

namespace eigen_backend {

void dynamic_elastic_linear(Analysis& analysis,
                            const FEM& fem,
                            const Eigen::MatrixXd& y_elem,
                            int Iter = 0);

} // namespace eigen_backend
