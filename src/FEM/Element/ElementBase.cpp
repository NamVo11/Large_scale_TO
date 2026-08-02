#include "FEM/Element/ElementBase.hpp"

#include <cmath>
#include <stdexcept>

ElementBase::ElementBase(const FEM& fem, int elem_local_id)
    : local_id(elem_local_id),
      raw_id(elem_local_id >= 0 && elem_local_id < static_cast<int>(fem.element_ids.size())
                 ? fem.element_ids[static_cast<std::size_t>(elem_local_id)]
                 : elem_local_id + 1),
      type(elem_local_id >= 0 && elem_local_id < static_cast<int>(fem.ElementTypes.size())
               ? fem.ElementTypes[static_cast<std::size_t>(elem_local_id)]
               : ElementType::Unknown),
      dim(fem.dim),
      node_ids(fem.Element.at(static_cast<std::size_t>(elem_local_id))) {
    dof_ids.resize(node_ids.size() * static_cast<std::size_t>(dim));
    for (std::size_t a = 0; a < node_ids.size(); ++a) {
        for (int d = 0; d < dim; ++d) {
            dof_ids[a * static_cast<std::size_t>(dim) + static_cast<std::size_t>(d)] =
                static_cast<PetscInt>(dim * node_ids[a] + d);
        }
    }
}

std::array<double, 3> ElementBase::center(const FEM& fem) const {
    std::array<double, 3> c{0.0, 0.0, 0.0};
    for (int n : node_ids) {
        const auto& p = fem.Node.at(static_cast<std::size_t>(n));
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    const double inv = node_ids.empty() ? 0.0 : 1.0 / static_cast<double>(node_ids.size());
    return {c[0] * inv, c[1] * inv, c[2] * inv};
}

Eigen::Vector3d ElementBase::node_vec(const FEM& fem, int node_id) {
    const auto& p = fem.Node.at(static_cast<std::size_t>(node_id));
    return Eigen::Vector3d(p[0], p[1], p[2]);
}

double ElementBase::triangle_area(const Eigen::Vector3d& a,
                                  const Eigen::Vector3d& b,
                                  const Eigen::Vector3d& c) {
    return 0.5 * ((b - a).cross(c - a)).norm();
}

double ElementBase::tetra_volume(const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b,
                                 const Eigen::Vector3d& c,
                                 const Eigen::Vector3d& d) {
    return std::abs((b - a).dot((c - a).cross(d - a))) / 6.0;
}

double ElementBase::det3(const std::array<double, 9>& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7])
         - a[1] * (a[3] * a[8] - a[5] * a[6])
         + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

std::array<double, 9> ElementBase::inv3(const std::array<double, 9>& a) {
    const double d = det3(a);
    if (std::abs(d) < 1e-30) throw std::runtime_error("Singular 3D element Jacobian");
    return {
         (a[4] * a[8] - a[5] * a[7]) / d,
        -(a[1] * a[8] - a[2] * a[7]) / d,
         (a[1] * a[5] - a[2] * a[4]) / d,
        -(a[3] * a[8] - a[5] * a[6]) / d,
         (a[0] * a[8] - a[2] * a[6]) / d,
        -(a[0] * a[5] - a[2] * a[3]) / d,
         (a[3] * a[7] - a[4] * a[6]) / d,
        -(a[0] * a[7] - a[1] * a[6]) / d,
         (a[0] * a[4] - a[1] * a[3]) / d};
}

void ElementBase::add_btdb(LocalDenseMatrix& ke,
                           const std::vector<double>& B,
                           const std::vector<double>& D,
                           int strain_dim,
                           int ndof,
                           double weight) {
    for (int i = 0; i < ndof; ++i) {
        for (int j = 0; j < ndof; ++j) {
            double v = 0.0;
            for (int a = 0; a < strain_dim; ++a) {
                double db = 0.0;
                for (int b = 0; b < strain_dim; ++b) db += D[a * strain_dim + b] * B[b * ndof + j];
                v += B[a * ndof + i] * db;
            }
            ke(i, j) += v * weight;
        }
    }
}

std::vector<double> ElementBase::signed_detJ_values(const FEM&, const std::vector<int>&) const {
    return {};
}

std::vector<ElementOrientationPermutation> ElementBase::orientation_check_permutations() const {
    return {};
}

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

std::shared_ptr<ElementBase> make_element_object(const FEM& fem, int elem_local_id) {
    const ElementType type = elem_local_id >= 0 && elem_local_id < static_cast<int>(fem.ElementTypes.size())
        ? fem.ElementTypes[static_cast<std::size_t>(elem_local_id)]
        : ElementType::Unknown;
    switch (type) {
        case ElementType::Q4:
            return std::make_shared<Q4Element>(fem, elem_local_id);
        case ElementType::TRI3:
            return std::make_shared<Tri3Element>(fem, elem_local_id);
        case ElementType::H8:
            return std::make_shared<H8Element>(fem, elem_local_id);
        case ElementType::T4:
            return std::make_shared<T4Element>(fem, elem_local_id);
        case ElementType::P6:
            return std::make_shared<P6Element>(fem, elem_local_id);
        default:
            throw std::runtime_error("Unsupported element object type: " + element_type_name(type));
    }
}
