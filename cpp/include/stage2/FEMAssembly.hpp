#pragma once

#include "stage2/MatlabLikeData.hpp"

#include <petscmat.h>

#include <vector>

namespace stage2 {

struct ElementMatrix {
    int ndof = 0;
    std::vector<double> ke;
    std::vector<double> me;
};

ElementMatrix compute_element_matrices(const FEM& fem, int elem_id, double nu, double thickness);

void assemble_K_M(const FEM& fem,
                  const Analysis& analysis,
                  const std::vector<std::vector<double>>& y_elem,
                  Mat K,
                  Mat M);

} // namespace stage2
