#include "Optimization/Filtering/Filtering.hpp"

#include "helper/KDTree3D.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr int kPreOptimizationThreads = 8;

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
}

double sech2(double x) {
    const double c = std::cosh(x);
    return 1.0 / (c * c);
}

} // namespace

void Filtering::initialize(const FEM& fem) {
    neighbors.assign(static_cast<std::size_t>(fem.NElem), {});
    weights.assign(static_cast<std::size_t>(fem.NElem), {});
    Hs.assign(static_cast<std::size_t>(fem.NElem), 0.0);
    if (static_cast<int>(fem.volume.size()) != fem.NElem ||
        static_cast<int>(fem.mesh_size.size()) != fem.NElem) {
        throw std::runtime_error("Filtering::initialize: fem volume/mesh_size are not initialized");
    }

    KDTree3D kd_tree;
    for (int e = 0; e < fem.NElem; ++e) {
        const auto& c = fem.elem_center[static_cast<std::size_t>(e)];
        kd_tree.insert(Eigen::Vector3d(c[0], c[1], c[2]), e);
    }

#pragma omp parallel for num_threads(kPreOptimizationThreads) schedule(static)
    for (int i = 0; i < fem.NElem; ++i) {
        const auto& ci_raw = fem.elem_center[static_cast<std::size_t>(i)];
        const Eigen::Vector3d ci(ci_raw[0], ci_raw[1], ci_raw[2]);
        const double radius = rmin * fem.mesh_size[static_cast<std::size_t>(i)];
        if (!(radius > 0.0)) {
            neighbors[static_cast<std::size_t>(i)].push_back(i);
            weights[static_cast<std::size_t>(i)].push_back(fem.volume[static_cast<std::size_t>(i)]);
            Hs[static_cast<std::size_t>(i)] = fem.volume[static_cast<std::size_t>(i)];
            continue;
        }

        std::vector<int> ids = kd_tree.searchWithinRadius(ci, radius);
        for (int j : ids) {
            const auto& cj_raw = fem.elem_center[static_cast<std::size_t>(j)];
            const Eigen::Vector3d cj(cj_raw[0], cj_raw[1], cj_raw[2]);
            const double dist = (ci - cj).norm();
            const double w = radius - dist;
            if (w > 0.0) {
                const double weighted_volume = w * fem.volume[static_cast<std::size_t>(j)];
                neighbors[static_cast<std::size_t>(i)].push_back(j);
                weights[static_cast<std::size_t>(i)].push_back(weighted_volume);
                Hs[static_cast<std::size_t>(i)] += weighted_volume;
            }
        }
        if (!(Hs[static_cast<std::size_t>(i)] > 0.0)) {
            neighbors[static_cast<std::size_t>(i)].push_back(i);
            weights[static_cast<std::size_t>(i)].push_back(fem.volume[static_cast<std::size_t>(i)]);
            Hs[static_cast<std::size_t>(i)] = fem.volume[static_cast<std::size_t>(i)];
        }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    for (int e = 0; e < fem.NElem; ++e) {
        const auto& ids = neighbors[static_cast<std::size_t>(e)];
        const auto& ws = weights[static_cast<std::size_t>(e)];
        for (std::size_t p = 0; p < ids.size(); ++p) triplets.emplace_back(e, ids[p], ws[p]);
    }
    H.resize(fem.NElem, fem.NElem);
    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();
    Hs_eigen.resize(fem.NElem);
    for (int e = 0; e < fem.NElem; ++e) Hs_eigen(e) = Hs[static_cast<std::size_t>(e)];
    neighbors.clear();
    weights.clear();
}

void Filtering::calculate(const Eigen::MatrixXd& x,
                          const std::vector<int>& x_label) {
    const int nelem = static_cast<int>(x.rows());
    const int nvar = nvar_of(x);
    xTilde = H * x;
    xTilde.array().colwise() /= Hs_eigen.array();
    const double denom = std::tanh(beta * T) + std::tanh(beta * (1.0 - T));
    xPhys = (std::tanh(beta * T) + (beta * (xTilde.array() - T)).tanh()).matrix() / denom;

    if (static_cast<int>(x_label.size()) != nelem) {
        throw std::runtime_error("Filtering::calculate: x_label size mismatch");
    }
    for (int j = 0; j < nvar; ++j) {
        double* column = xPhys.col(j).data();
        for (int e = 0; e < nelem; ++e) {
            const int label = x_label[static_cast<std::size_t>(e)];
            if (label > 0) {
                column[e] = 1.0;
            } else if (label < 0) {
                column[e] = 0.0;
            }
        }
    }
    if (xPhys.cols() != nvar) throw std::runtime_error("Filtering::calculate: invalid xPhys columns");
}

Eigen::MatrixXd Filtering::sensitivity(const Eigen::MatrixXd& df_dx_phys) const {
    const int nelem = static_cast<int>(df_dx_phys.rows());
    const int nvar = nvar_of(df_dx_phys);
    Eigen::MatrixXd tmp(nelem, nvar);
    const double denom = std::tanh(beta * T) + std::tanh(beta * (1.0 - T));
    for (int v = 0; v < nvar; ++v) {
        tmp.col(v) = (df_dx_phys.col(v).array() *
                      (beta * (beta * (xTilde.col(v).array() - T)).unaryExpr(
                                  [](double x) { return sech2(x); }) / denom) /
                      Hs_eigen.array()).matrix();
    }
    return H.transpose() * tmp;
}

void DensityFiltering::initialize(const FEM& fem) {
    Filtering::initialize(fem);
}

void DensityFiltering::calculate(const Eigen::MatrixXd& x,
                                 const std::vector<int>& x_label) {
    Filtering::calculate(x, x_label);
}

Eigen::MatrixXd DensityFiltering::sensitivity(const Eigen::MatrixXd& df_dx_phys) const {
    return Filtering::sensitivity(df_dx_phys);
}

void SensitivityFiltering::initialize(const FEM& fem) {
    Filtering::initialize(fem);
}

void SensitivityFiltering::calculate(const Eigen::MatrixXd& x,
                                     const std::vector<int>& x_label) {
    Filtering::calculate(x, x_label);
}

Eigen::MatrixXd SensitivityFiltering::sensitivity(const Eigen::MatrixXd& df_dx_phys) const {
    return Filtering::sensitivity(df_dx_phys);
}
