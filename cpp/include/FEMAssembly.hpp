#pragma once

#include "MatlabLikeData.hpp"

#include <Eigen/Dense>
#include <petscmat.h>

using LocalDenseMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct ElementMatrix {
    int ndof = 0;
    LocalDenseMatrix ke;
    LocalDenseMatrix me;
};

ElementMatrix compute_element_matrices(const FEM& fem, int elem_id, double nu, double thickness);

void assemble_K_M(const FEM& fem,
                  const Analysis& analysis,
                  const Eigen::MatrixXd& y_elem,
                  Mat K,
                  Mat M);
