#include "FEM/Element/ElementBase.hpp"

#include <cmath>
#include <stdexcept>

Eigen::MatrixXd P6Element::shape_function_matrix(const Eigen::Vector3d& natural_coord) const {
    const double r = natural_coord(0);
    const double s = natural_coord(1);
    const double t = natural_coord(2);
    Eigen::Matrix<double, 1, 6> N;
    N << 0.5 * (1.0 - r - s) * (1.0 - t),
         0.5 * r * (1.0 - t),
         0.5 * s * (1.0 - t),
         0.5 * (1.0 - r - s) * (1.0 + t),
         0.5 * r * (1.0 + t),
         0.5 * s * (1.0 + t);
    return N;
}

double P6Element::volume(const FEM& fem) const {
    if (volume_is_cached) return volume_value;
    if (node_ids.size() != 6) return 0.0;
    constexpr double t_gp[2] = {-0.57735026918962576451, 0.57735026918962576451};
    volume_value = 0.0;
    for (double t : t_gp) {
        const double r = 1.0 / 3.0;
        const double s = 1.0 / 3.0;
        const double dN[18] = {
            -0.5 * (1.0 - t), -0.5 * (1.0 - t), -0.5 * (1.0 - r - s),
             0.5 * (1.0 - t),  0.0,             -0.5 * r,
             0.0,              0.5 * (1.0 - t), -0.5 * s,
            -0.5 * (1.0 + t), -0.5 * (1.0 + t),  0.5 * (1.0 - r - s),
             0.5 * (1.0 + t),  0.0,              0.5 * r,
             0.0,              0.5 * (1.0 + t),  0.5 * s};
        std::array<double, 9> J{};
        for (int a = 0; a < 6; ++a) {
            const Eigen::Vector3d x = node_vec(fem, node_ids[static_cast<std::size_t>(a)]);
            J[0] += x(0) * dN[3 * a + 0];   J[1] += x(0) * dN[3 * a + 1];   J[2] += x(0) * dN[3 * a + 2];
            J[3] += x(1) * dN[3 * a + 0];   J[4] += x(1) * dN[3 * a + 1];   J[5] += x(1) * dN[3 * a + 2];
            J[6] += x(2) * dN[3 * a + 0];   J[7] += x(2) * dN[3 * a + 1];   J[8] += x(2) * dN[3 * a + 2];
        }
        volume_value += 0.5 * std::abs(det3(J));
    }
    volume_is_cached = true;
    return volume_value;
}

double P6Element::mesh_size(const FEM& fem) const {
    return std::cbrt(volume(fem));
}

LocalDenseMatrix P6Element::element_stiffness(const FEM&, double, double) const {
    throw std::runtime_error("P6 element_stiffness is not implemented yet");
}

LocalDenseMatrix P6Element::element_mass(const FEM&, double) const {
    throw std::runtime_error("P6 element_mass is not implemented yet");
}

ElementMatrix P6Element::compute_element_matrices(const FEM&, double, double) const {
    throw std::runtime_error("P6 compute_element_matrices is not implemented yet");
}
