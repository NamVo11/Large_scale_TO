#include "Analysis/Analysis.hpp"

#include "MatlabLikeData.hpp"

#include <stdexcept>

void Analysis::solver(const FEM&, const Eigen::MatrixXd&, int) {
    throw std::runtime_error("Analysis::solver is not implemented for base Analysis");
}
