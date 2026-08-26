#include "FEM/Element/ElementBase.hpp"

#include <cmath>
#include <stdexcept>

Eigen::MatrixXd Q4Element::shape_function_matrix(const Eigen::Vector3d& natural_coord) const {
    const double xi = natural_coord(0);
    const double eta = natural_coord(1);
    Eigen::Matrix<double, 1, 4> N;
    N << 0.25 * (1.0 - xi) * (1.0 - eta),
         0.25 * (1.0 + xi) * (1.0 - eta),
         0.25 * (1.0 + xi) * (1.0 + eta),
         0.25 * (1.0 - xi) * (1.0 + eta);
    return N;
}

double Q4Element::volume(const FEM& fem) const {
    if (volume_is_cached) return volume_value;
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};
    volume_value = 0.0;
    for (double xi : gp) {
        for (double eta : gp) {
            const double dNdxi[8] = {
                -0.25 * (1.0 - eta), -0.25 * (1.0 - xi),
                 0.25 * (1.0 - eta), -0.25 * (1.0 + xi),
                 0.25 * (1.0 + eta),  0.25 * (1.0 + xi),
                -0.25 * (1.0 + eta),  0.25 * (1.0 - xi)};
            double J[4] = {0.0, 0.0, 0.0, 0.0};
            for (int a = 0; a < 4; ++a) {
                const auto& p = fem.Node.at(static_cast<std::size_t>(node_ids[a]));
                J[0] += p[0] * dNdxi[2 * a + 0];
                J[1] += p[0] * dNdxi[2 * a + 1];
                J[2] += p[1] * dNdxi[2 * a + 0];
                J[3] += p[1] * dNdxi[2 * a + 1];
            }
            const double signed_detJ = J[0] * J[3] - J[1] * J[2];
            if (!(signed_detJ > 0.0)) {
                detJ_is_positive = false;
            }
            volume_value += std::abs(signed_detJ);
        }
    }
    volume_is_cached = true;
    return volume_value;
}

double Q4Element::mesh_size(const FEM& fem) const {
    return std::sqrt(volume(fem));
}

LocalDenseMatrix Q4Element::element_stiffness(const FEM& fem, double nu, double thickness) const {
    constexpr int ndof = 8;
    LocalDenseMatrix ke = LocalDenseMatrix::Zero(ndof, ndof);
    const double c = 1.0 / (1.0 - nu * nu);
    const std::vector<double> D = {
        c, c * nu, 0.0,
        c * nu, c, 0.0,
        0.0, 0.0, c * (1.0 - nu) / 2.0};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};

    for (double xi : gp) {
        for (double eta : gp) {
            const double dNdxi[8] = {
                -0.25 * (1.0 - eta), -0.25 * (1.0 - xi),
                 0.25 * (1.0 - eta), -0.25 * (1.0 + xi),
                 0.25 * (1.0 + eta),  0.25 * (1.0 + xi),
                -0.25 * (1.0 + eta),  0.25 * (1.0 - xi)};
            double J[4] = {0.0, 0.0, 0.0, 0.0};
            for (int a = 0; a < 4; ++a) {
                const auto& p = fem.Node.at(static_cast<std::size_t>(node_ids[a]));
                J[0] += p[0] * dNdxi[2 * a + 0];
                J[1] += p[0] * dNdxi[2 * a + 1];
                J[2] += p[1] * dNdxi[2 * a + 0];
                J[3] += p[1] * dNdxi[2 * a + 1];
            }
            const double detJ = J[0] * J[3] - J[1] * J[2];
            if (std::abs(detJ) < 1e-30) throw std::runtime_error("Singular Q4 Jacobian");
            const double invJ[4] = {J[3] / detJ, -J[1] / detJ, -J[2] / detJ, J[0] / detJ};
            std::vector<double> B(3 * ndof, 0.0);
            for (int a = 0; a < 4; ++a) {
                const double dx = invJ[0] * dNdxi[2 * a + 0] + invJ[1] * dNdxi[2 * a + 1];
                const double dy = invJ[2] * dNdxi[2 * a + 0] + invJ[3] * dNdxi[2 * a + 1];
                B[0 * ndof + 2 * a + 0] = dx;
                B[1 * ndof + 2 * a + 1] = dy;
                B[2 * ndof + 2 * a + 0] = dy;
                B[2 * ndof + 2 * a + 1] = dx;
            }
            if (!(detJ > 0.0)) {
                throw std::runtime_error("Q4Element::element_stiffness: non-positive detJ for element local_id=" +
                                         std::to_string(local_id) +
                                         ", raw_id=" + std::to_string(raw_id));
            }
            add_btdb(ke, B, D, 3, ndof, detJ * thickness);
        }
    }
    return ke;
}

LocalDenseMatrix Q4Element::element_mass(const FEM& fem, double thickness) const {
    constexpr int ndof = 8;
    LocalDenseMatrix me = LocalDenseMatrix::Zero(ndof, ndof);
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};
    for (double xi : gp) {
        for (double eta : gp) {
            const Eigen::Vector3d natural_coord(xi, eta, 0.0);
            const Eigen::MatrixXd N = shape_function_matrix(natural_coord);
            const double dNdxi[8] = {
                -0.25 * (1.0 - eta), -0.25 * (1.0 - xi),
                 0.25 * (1.0 - eta), -0.25 * (1.0 + xi),
                 0.25 * (1.0 + eta),  0.25 * (1.0 + xi),
                -0.25 * (1.0 + eta),  0.25 * (1.0 - xi)};
            double J[4] = {0.0, 0.0, 0.0, 0.0};
            for (int a = 0; a < 4; ++a) {
                const auto& p = fem.Node.at(static_cast<std::size_t>(node_ids[a]));
                J[0] += p[0] * dNdxi[2 * a + 0];
                J[1] += p[0] * dNdxi[2 * a + 1];
                J[2] += p[1] * dNdxi[2 * a + 0];
                J[3] += p[1] * dNdxi[2 * a + 1];
            }
            const double detJ = J[0] * J[3] - J[1] * J[2];
            if (!(detJ > 0.0)) {
                throw std::runtime_error("Q4Element::element_mass: non-positive detJ for element local_id=" +
                                         std::to_string(local_id) +
                                         ", raw_id=" + std::to_string(raw_id));
            }
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    const double v = N(0, a) * N(0, b) * detJ * thickness;
                    me(2 * a + 0, 2 * b + 0) += v;
                    me(2 * a + 1, 2 * b + 1) += v;
                }
            }
        }
    }
    return me;
}

ElementMatrix Q4Element::compute_element_matrices(const FEM& fem, double nu, double thickness) const {
    ElementMatrix out;
    out.ndof = 8;
    out.ke = LocalDenseMatrix::Zero(out.ndof, out.ndof);
    out.me = LocalDenseMatrix::Zero(out.ndof, out.ndof);

    const double c = 1.0 / (1.0 - nu * nu);
    const std::vector<double> D = {
        c, c * nu, 0.0,
        c * nu, c, 0.0,
        0.0, 0.0, c * (1.0 - nu) / 2.0};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};

    for (double xi : gp) {
        for (double eta : gp) {
            const double N[4] = {
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta)};
            const double dNdxi[8] = {
                -0.25 * (1.0 - eta), -0.25 * (1.0 - xi),
                 0.25 * (1.0 - eta), -0.25 * (1.0 + xi),
                 0.25 * (1.0 + eta),  0.25 * (1.0 + xi),
                -0.25 * (1.0 + eta),  0.25 * (1.0 - xi)};

            double J[4] = {0.0, 0.0, 0.0, 0.0};
            for (int a = 0; a < 4; ++a) {
                const auto& p = fem.Node.at(static_cast<std::size_t>(node_ids[a]));
                J[0] += p[0] * dNdxi[2 * a + 0];
                J[1] += p[0] * dNdxi[2 * a + 1];
                J[2] += p[1] * dNdxi[2 * a + 0];
                J[3] += p[1] * dNdxi[2 * a + 1];
            }
            const double detJ = J[0] * J[3] - J[1] * J[2];
            if (!(detJ > 0.0)) {
                throw std::runtime_error("Q4Element::compute_element_matrices: non-positive detJ for element local_id=" +
                                         std::to_string(local_id) +
                                         ", raw_id=" + std::to_string(raw_id));
            }
            const double invJ[4] = {J[3] / detJ, -J[1] / detJ, -J[2] / detJ, J[0] / detJ};

            std::vector<double> B(3 * out.ndof, 0.0);
            for (int a = 0; a < 4; ++a) {
                const double dx = invJ[0] * dNdxi[2 * a + 0] + invJ[1] * dNdxi[2 * a + 1];
                const double dy = invJ[2] * dNdxi[2 * a + 0] + invJ[3] * dNdxi[2 * a + 1];
                B[0 * out.ndof + 2 * a + 0] = dx;
                B[1 * out.ndof + 2 * a + 1] = dy;
                B[2 * out.ndof + 2 * a + 0] = dy;
                B[2 * out.ndof + 2 * a + 1] = dx;
            }
            add_btdb(out.ke, B, D, 3, out.ndof, detJ * thickness);

            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    const double v = N[a] * N[b] * detJ * thickness;
                    out.me(2 * a + 0, 2 * b + 0) += v;
                    out.me(2 * a + 1, 2 * b + 1) += v;
                }
            }
        }
    }
    return out;
}

std::vector<double> Q4Element::signed_detJ_values(const FEM& fem, const std::vector<int>& node_order) const {
    const std::vector<int>& ids = node_order.empty() ? node_ids : node_order;
    if (ids.size() != 4) return {};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};
    std::vector<double> values;
    values.reserve(4);
    for (double xi : gp) {
        for (double eta : gp) {
            const double dNdxi[8] = {
                -0.25 * (1.0 - eta), -0.25 * (1.0 - xi),
                 0.25 * (1.0 - eta), -0.25 * (1.0 + xi),
                 0.25 * (1.0 + eta),  0.25 * (1.0 + xi),
                -0.25 * (1.0 + eta),  0.25 * (1.0 - xi)};
            double J[4] = {0.0, 0.0, 0.0, 0.0};
            for (int a = 0; a < 4; ++a) {
                const auto& p = fem.Node.at(static_cast<std::size_t>(ids[static_cast<std::size_t>(a)]));
                J[0] += p[0] * dNdxi[2 * a + 0];
                J[1] += p[0] * dNdxi[2 * a + 1];
                J[2] += p[1] * dNdxi[2 * a + 0];
                J[3] += p[1] * dNdxi[2 * a + 1];
            }
            values.push_back(J[0] * J[3] - J[1] * J[2]);
        }
    }
    return values;
}

std::vector<ElementOrientationPermutation> Q4Element::orientation_check_permutations() const {
    return {
        {"as_file_order 1-2-3-4", {0, 1, 2, 3}},
        {"reverse_order 1-4-3-2", {0, 3, 2, 1}},
        {"rotate_only 2-3-4-1", {1, 2, 3, 0}},
        {"swap_1_2 2-1-3-4", {1, 0, 2, 3}},
    };
}
