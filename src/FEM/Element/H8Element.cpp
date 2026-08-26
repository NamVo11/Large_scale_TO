#include "FEM/Element/ElementBase.hpp"

#include <cmath>
#include <stdexcept>

Eigen::MatrixXd H8Element::shape_function_matrix(const Eigen::Vector3d& natural_coord) const {
    const double xi = natural_coord(0);
    const double eta = natural_coord(1);
    const double zeta = natural_coord(2);
    constexpr int s[8][3] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}};
    Eigen::Matrix<double, 1, 8> N;
    for (int a = 0; a < 8; ++a) {
        N(a) = 0.125 * (1.0 + s[a][0] * xi) * (1.0 + s[a][1] * eta) * (1.0 + s[a][2] * zeta);
    }
    return N;
}

double H8Element::volume(const FEM& fem) const {
    if (volume_is_cached) return volume_value;
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};
    constexpr int s[8][3] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}};
    volume_value = 0.0;
    for (double xi : gp) {
        for (double eta : gp) {
            for (double zeta : gp) {
                std::array<double, 9> J{};
                for (int a = 0; a < 8; ++a) {
                    const Eigen::Vector3d x = node_vec(fem, node_ids[static_cast<std::size_t>(a)]);
                    const double dN_dxi = 0.125 * s[a][0] * (1.0 + s[a][1] * eta) * (1.0 + s[a][2] * zeta);
                    const double dN_deta = 0.125 * s[a][1] * (1.0 + s[a][0] * xi) * (1.0 + s[a][2] * zeta);
                    const double dN_dzeta = 0.125 * s[a][2] * (1.0 + s[a][0] * xi) * (1.0 + s[a][1] * eta);
                    J[0] += x(0) * dN_dxi;   J[1] += x(0) * dN_deta;   J[2] += x(0) * dN_dzeta;
                    J[3] += x(1) * dN_dxi;   J[4] += x(1) * dN_deta;   J[5] += x(1) * dN_dzeta;
                    J[6] += x(2) * dN_dxi;   J[7] += x(2) * dN_deta;   J[8] += x(2) * dN_dzeta;
                }
                const double signed_detJ = det3(J);
                if (!(signed_detJ > 0.0)) {
                    detJ_is_positive = false;
                }
                volume_value += std::abs(signed_detJ);
            }
        }
    }
    volume_is_cached = true;
    return volume_value;
}

double H8Element::mesh_size(const FEM& fem) const {
    return std::cbrt(volume(fem));
}

LocalDenseMatrix H8Element::element_stiffness(const FEM& fem, double nu, double) const {
    constexpr int ndof = 24;
    LocalDenseMatrix ke = LocalDenseMatrix::Zero(ndof, ndof);
    const double c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu));
    std::vector<double> D(6 * 6, 0.0);
    D[0 * 6 + 0] = c * (1.0 - nu);
    D[0 * 6 + 1] = c * nu;
    D[0 * 6 + 2] = c * nu;
    D[1 * 6 + 0] = c * nu;
    D[1 * 6 + 1] = c * (1.0 - nu);
    D[1 * 6 + 2] = c * nu;
    D[2 * 6 + 0] = c * nu;
    D[2 * 6 + 1] = c * nu;
    D[2 * 6 + 2] = c * (1.0 - nu);
    D[3 * 6 + 3] = c * (1.0 - 2.0 * nu) / 2.0;
    D[4 * 6 + 4] = c * (1.0 - 2.0 * nu) / 2.0;
    D[5 * 6 + 5] = c * (1.0 - 2.0 * nu) / 2.0;
    constexpr int s[8][3] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};
    for (double xi : gp) {
        for (double eta : gp) {
            for (double zeta : gp) {
                std::array<double, 24> dNdxi{};
                for (int i = 0; i < 8; ++i) {
                    dNdxi[i * 3 + 0] = 0.125 * s[i][0] * (1.0 + s[i][1] * eta) * (1.0 + s[i][2] * zeta);
                    dNdxi[i * 3 + 1] = 0.125 * s[i][1] * (1.0 + s[i][0] * xi) * (1.0 + s[i][2] * zeta);
                    dNdxi[i * 3 + 2] = 0.125 * s[i][2] * (1.0 + s[i][0] * xi) * (1.0 + s[i][1] * eta);
                }
                std::array<double, 9> J{};
                for (int a = 0; a < 8; ++a) {
                    const auto& p = fem.Node.at(static_cast<std::size_t>(node_ids[a]));
                    for (int alpha = 0; alpha < 3; ++alpha) {
                        J[0 * 3 + alpha] += p[0] * dNdxi[a * 3 + alpha];
                        J[1 * 3 + alpha] += p[1] * dNdxi[a * 3 + alpha];
                        J[2 * 3 + alpha] += p[2] * dNdxi[a * 3 + alpha];
                    }
                }
                const double detJ = det3(J);
                if (!(detJ > 0.0)) {
                    throw std::runtime_error("H8Element::element_stiffness: non-positive detJ for element local_id=" +
                                             std::to_string(local_id) +
                                             ", raw_id=" + std::to_string(raw_id));
                }
                const std::array<double, 9> iJ = inv3(J);
                std::vector<double> B(6 * ndof, 0.0);
                for (int a = 0; a < 8; ++a) {
                    double d[3]{};
                    for (int beta = 0; beta < 3; ++beta) {
                        for (int alpha = 0; alpha < 3; ++alpha) d[beta] += dNdxi[a * 3 + alpha] * iJ[alpha * 3 + beta];
                    }
                    const int col = 3 * a;
                    B[0 * ndof + col + 0] = d[0];
                    B[1 * ndof + col + 1] = d[1];
                    B[2 * ndof + col + 2] = d[2];
                    B[3 * ndof + col + 0] = d[1];
                    B[3 * ndof + col + 1] = d[0];
                    B[4 * ndof + col + 1] = d[2];
                    B[4 * ndof + col + 2] = d[1];
                    B[5 * ndof + col + 0] = d[2];
                    B[5 * ndof + col + 2] = d[0];
                }
                add_btdb(ke, B, D, 6, ndof, detJ);
            }
        }
    }
    return ke;
}

LocalDenseMatrix H8Element::element_mass(const FEM& fem, double) const {
    constexpr int ndof = 24;
    LocalDenseMatrix me = LocalDenseMatrix::Zero(ndof, ndof);
    constexpr int s[8][3] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};
    for (double xi : gp) {
        for (double eta : gp) {
            for (double zeta : gp) {
                const Eigen::MatrixXd N = shape_function_matrix(Eigen::Vector3d(xi, eta, zeta));
                std::array<double, 9> J{};
                for (int a = 0; a < 8; ++a) {
                    const Eigen::Vector3d x = node_vec(fem, node_ids[static_cast<std::size_t>(a)]);
                    const double dN_dxi = 0.125 * s[a][0] * (1.0 + s[a][1] * eta) * (1.0 + s[a][2] * zeta);
                    const double dN_deta = 0.125 * s[a][1] * (1.0 + s[a][0] * xi) * (1.0 + s[a][2] * zeta);
                    const double dN_dzeta = 0.125 * s[a][2] * (1.0 + s[a][0] * xi) * (1.0 + s[a][1] * eta);
                    J[0] += x(0) * dN_dxi;   J[1] += x(0) * dN_deta;   J[2] += x(0) * dN_dzeta;
                    J[3] += x(1) * dN_dxi;   J[4] += x(1) * dN_deta;   J[5] += x(1) * dN_dzeta;
                    J[6] += x(2) * dN_dxi;   J[7] += x(2) * dN_deta;   J[8] += x(2) * dN_dzeta;
                }
                const double detJ = det3(J);
                if (!(detJ > 0.0)) {
                    throw std::runtime_error("H8Element::element_mass: non-positive detJ for element local_id=" +
                                             std::to_string(local_id) +
                                             ", raw_id=" + std::to_string(raw_id));
                }
                for (int a = 0; a < 8; ++a) {
                    for (int b = 0; b < 8; ++b) {
                        const double v = N(0, a) * N(0, b) * detJ;
                        for (int d = 0; d < 3; ++d) me(3 * a + d, 3 * b + d) += v;
                    }
                }
            }
        }
    }
    return me;
}

ElementMatrix H8Element::compute_element_matrices(const FEM& fem, double nu, double) const {
    ElementMatrix out;
    out.ndof = 24;
    out.ke = LocalDenseMatrix::Zero(out.ndof, out.ndof);
    out.me = LocalDenseMatrix::Zero(out.ndof, out.ndof);

    const double c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu));
    std::vector<double> D(6 * 6, 0.0);
    D[0 * 6 + 0] = c * (1.0 - nu);
    D[0 * 6 + 1] = c * nu;
    D[0 * 6 + 2] = c * nu;
    D[1 * 6 + 0] = c * nu;
    D[1 * 6 + 1] = c * (1.0 - nu);
    D[1 * 6 + 2] = c * nu;
    D[2 * 6 + 0] = c * nu;
    D[2 * 6 + 1] = c * nu;
    D[2 * 6 + 2] = c * (1.0 - nu);
    D[3 * 6 + 3] = c * (1.0 - 2.0 * nu) / 2.0;
    D[4 * 6 + 4] = c * (1.0 - 2.0 * nu) / 2.0;
    D[5 * 6 + 5] = c * (1.0 - 2.0 * nu) / 2.0;

    constexpr int s[8][3] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};

    for (double xi : gp) {
        for (double eta : gp) {
            for (double zeta : gp) {
                std::array<double, 8> N{};
                std::array<double, 24> dNdxi{};
                for (int i = 0; i < 8; ++i) {
                    const double sx = s[i][0];
                    const double sy = s[i][1];
                    const double sz = s[i][2];
                    N[i] = 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta);
                    dNdxi[i * 3 + 0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);
                    dNdxi[i * 3 + 1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta);
                    dNdxi[i * 3 + 2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta);
                }

                std::array<double, 9> J{};
                for (int a = 0; a < 8; ++a) {
                    const auto& p = fem.Node.at(static_cast<std::size_t>(node_ids[a]));
                    for (int alpha = 0; alpha < 3; ++alpha) {
                        J[0 * 3 + alpha] += p[0] * dNdxi[a * 3 + alpha];
                        J[1 * 3 + alpha] += p[1] * dNdxi[a * 3 + alpha];
                        J[2 * 3 + alpha] += p[2] * dNdxi[a * 3 + alpha];
                    }
                }
                const double detJ = det3(J);
                if (!(detJ > 0.0)) {
                    throw std::runtime_error("H8Element::compute_element_matrices: non-positive detJ for element local_id=" +
                                             std::to_string(local_id) +
                                             ", raw_id=" + std::to_string(raw_id));
                }
                const std::array<double, 9> iJ = inv3(J);

                std::vector<double> B(6 * out.ndof, 0.0);
                for (int a = 0; a < 8; ++a) {
                    double d[3]{};
                    for (int beta = 0; beta < 3; ++beta) {
                        for (int alpha = 0; alpha < 3; ++alpha) d[beta] += dNdxi[a * 3 + alpha] * iJ[alpha * 3 + beta];
                    }
                    const int col = 3 * a;
                    B[0 * out.ndof + col + 0] = d[0];
                    B[1 * out.ndof + col + 1] = d[1];
                    B[2 * out.ndof + col + 2] = d[2];
                    B[3 * out.ndof + col + 0] = d[1];
                    B[3 * out.ndof + col + 1] = d[0];
                    B[4 * out.ndof + col + 1] = d[2];
                    B[4 * out.ndof + col + 2] = d[1];
                    B[5 * out.ndof + col + 0] = d[2];
                    B[5 * out.ndof + col + 2] = d[0];
                }
                add_btdb(out.ke, B, D, 6, out.ndof, detJ);

                for (int a = 0; a < 8; ++a) {
                    for (int b = 0; b < 8; ++b) {
                        const double v = N[a] * N[b] * detJ;
                        for (int d = 0; d < 3; ++d) out.me(3 * a + d, 3 * b + d) += v;
                    }
                }
            }
        }
    }
    return out;
}

std::vector<double> H8Element::signed_detJ_values(const FEM& fem, const std::vector<int>& node_order) const {
    const std::vector<int>& ids = node_order.empty() ? node_ids : node_order;
    if (ids.size() != 8) return {};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};
    constexpr int s[8][3] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}};
    std::vector<double> values;
    values.reserve(8);
    for (double xi : gp) {
        for (double eta : gp) {
            for (double zeta : gp) {
                std::array<double, 9> J{};
                for (int a = 0; a < 8; ++a) {
                    const Eigen::Vector3d x = node_vec(fem, ids[static_cast<std::size_t>(a)]);
                    const double dN_dxi = 0.125 * s[a][0] * (1.0 + s[a][1] * eta) * (1.0 + s[a][2] * zeta);
                    const double dN_deta = 0.125 * s[a][1] * (1.0 + s[a][0] * xi) * (1.0 + s[a][2] * zeta);
                    const double dN_dzeta = 0.125 * s[a][2] * (1.0 + s[a][0] * xi) * (1.0 + s[a][1] * eta);
                    J[0] += x(0) * dN_dxi;   J[1] += x(0) * dN_deta;   J[2] += x(0) * dN_dzeta;
                    J[3] += x(1) * dN_dxi;   J[4] += x(1) * dN_deta;   J[5] += x(1) * dN_dzeta;
                    J[6] += x(2) * dN_dxi;   J[7] += x(2) * dN_deta;   J[8] += x(2) * dN_dzeta;
                }
                values.push_back(det3(J));
            }
        }
    }
    return values;
}

std::vector<ElementOrientationPermutation> H8Element::orientation_check_permutations() const {
    return {
        {"as_file_order 1-2-3-4-5-6-7-8", {0, 1, 2, 3, 4, 5, 6, 7}},
        {"reverse_two_faces_keep_pairs 1-4-3-2-5-8-7-6", {0, 3, 2, 1, 4, 7, 6, 5}},
    };
}
