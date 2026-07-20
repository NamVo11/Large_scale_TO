#pragma once

#include "MatlabLikeData.hpp"

void dynamic_elastic_linear(Analysis& analysis,
                            const FEM& fem,
                            const Eigen::MatrixXd& y_elem,
                            int Iter = 0);
