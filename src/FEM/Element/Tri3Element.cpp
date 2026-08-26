#include "FEM/Element/ElementBase.hpp"

#include <cmath>
#include <stdexcept>

Eigen::MatrixXd Tri3Element::shape_function_matrix(const Eigen::Vector3d& natural_coord) const {
    const double r = natural_coord(0);
    const double s = natural_coord(1);
    Eigen::Matrix<double, 1, 3> N;
    N << 1.0 - r - s, r, s;
    return N;
}

double Tri3Element::volume(const FEM& fem) const {
    if (volume_is_cached) return volume_value;
    if (node_ids.size() != 3) return 0.0;
    volume_value = triangle_area(node_vec(fem, node_ids[0]), node_vec(fem, node_ids[1]), node_vec(fem, node_ids[2]));
    volume_is_cached = true;
    return volume_value;
}

double Tri3Element::mesh_size(const FEM& fem) const {
    return std::sqrt(volume(fem));
}

LocalDenseMatrix Tri3Element::element_stiffness(const FEM&, double, double) const {
    throw std::runtime_error("TRI3 element_stiffness is not implemented yet");
}

LocalDenseMatrix Tri3Element::element_mass(const FEM&, double) const {
    throw std::runtime_error("TRI3 element_mass is not implemented yet");
}

ElementMatrix Tri3Element::compute_element_matrices(const FEM&, double, double) const {
    throw std::runtime_error("TRI3 compute_element_matrices is not implemented yet");
}
