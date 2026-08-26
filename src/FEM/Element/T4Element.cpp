#include "FEM/Element/ElementBase.hpp"

#include <cmath>
#include <stdexcept>

Eigen::MatrixXd T4Element::shape_function_matrix(const Eigen::Vector3d& natural_coord) const {
    const double r = natural_coord(0);
    const double s = natural_coord(1);
    const double t = natural_coord(2);
    Eigen::Matrix<double, 1, 4> N;
    N << 1.0 - r - s - t, r, s, t;
    return N;
}

double T4Element::volume(const FEM& fem) const {
    if (volume_is_cached) return volume_value;
    if (node_ids.size() != 4) return 0.0;
    volume_value = tetra_volume(node_vec(fem, node_ids[0]), node_vec(fem, node_ids[1]),
                                node_vec(fem, node_ids[2]), node_vec(fem, node_ids[3]));
    volume_is_cached = true;
    return volume_value;
}

double T4Element::mesh_size(const FEM& fem) const {
    return std::cbrt(volume(fem));
}

LocalDenseMatrix T4Element::element_stiffness(const FEM&, double, double) const {
    throw std::runtime_error("T4 element_stiffness is not implemented yet");
}

LocalDenseMatrix T4Element::element_mass(const FEM&, double) const {
    throw std::runtime_error("T4 element_mass is not implemented yet");
}

ElementMatrix T4Element::compute_element_matrices(const FEM&, double, double) const {
    throw std::runtime_error("T4 compute_element_matrices is not implemented yet");
}
