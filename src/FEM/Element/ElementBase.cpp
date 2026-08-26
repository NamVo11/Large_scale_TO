#include "FEM/Element/ElementBase.hpp"

#include <cmath>
#include <stdexcept>

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
