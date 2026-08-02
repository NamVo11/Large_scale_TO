#include "Optimization/Filtering.hpp"
#include "helper/KDTree3D.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
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

void filtering_initilization(const FEM& fem, Filtering& filtering, FilteringState& state) {
    state.neighbors.assign(static_cast<std::size_t>(fem.NElem), {});
    state.weights.assign(static_cast<std::size_t>(fem.NElem), {});
    state.Hs.assign(static_cast<std::size_t>(fem.NElem), 0.0);
    if (static_cast<int>(fem.volume.size()) != fem.NElem ||
        static_cast<int>(fem.mesh_size.size()) != fem.NElem) {
        throw std::runtime_error("filtering_initilization: fem volume/mesh_size are not initialized");
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
        const double radius = filtering.rmin * fem.mesh_size[static_cast<std::size_t>(i)];
        if (!(radius > 0.0)) {
            state.neighbors[static_cast<std::size_t>(i)].push_back(i);
            state.weights[static_cast<std::size_t>(i)].push_back(fem.volume[static_cast<std::size_t>(i)]);
            state.Hs[static_cast<std::size_t>(i)] = fem.volume[static_cast<std::size_t>(i)];
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
                state.neighbors[static_cast<std::size_t>(i)].push_back(j);
                state.weights[static_cast<std::size_t>(i)].push_back(weighted_volume);
                state.Hs[static_cast<std::size_t>(i)] += weighted_volume;
            }
        }
        if (!(state.Hs[static_cast<std::size_t>(i)] > 0.0)) {
            state.neighbors[static_cast<std::size_t>(i)].push_back(i);
            state.weights[static_cast<std::size_t>(i)].push_back(fem.volume[static_cast<std::size_t>(i)]);
            state.Hs[static_cast<std::size_t>(i)] = fem.volume[static_cast<std::size_t>(i)];
        }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    for (int e = 0; e < fem.NElem; ++e) {
        const auto& ids = state.neighbors[static_cast<std::size_t>(e)];
        const auto& ws = state.weights[static_cast<std::size_t>(e)];
        for (std::size_t p = 0; p < ids.size(); ++p) triplets.emplace_back(e, ids[p], ws[p]);
    }
    state.H.resize(fem.NElem, fem.NElem);
    state.H.setFromTriplets(triplets.begin(), triplets.end());
    state.H.makeCompressed();
    state.Hs_eigen.resize(fem.NElem);
    for (int e = 0; e < fem.NElem; ++e) state.Hs_eigen(e) = state.Hs[static_cast<std::size_t>(e)];
    state.neighbors.clear();
    state.weights.clear();
}

void filtering_calculation(const Eigen::MatrixXd& x,
                           const std::vector<int>& x_label,
                           const Filtering& filtering,
                           FilteringState& state) {
    const int nelem = static_cast<int>(x.rows());
    const int nvar = nvar_of(x);
    state.xTilde = state.H * x;
    state.xTilde.array().colwise() /= state.Hs_eigen.array();
    const double denom = std::tanh(filtering.beta * filtering.T) + std::tanh(filtering.beta * (1.0 - filtering.T));
    state.xPhys = (std::tanh(filtering.beta * filtering.T) +
                   (filtering.beta * (state.xTilde.array() - filtering.T)).tanh()).matrix() / denom;
    for (int e = 0; e < nelem; ++e) {
        if (x_label[static_cast<std::size_t>(e)] > 0) state.xPhys.row(e).setOnes();
        if (x_label[static_cast<std::size_t>(e)] < 0) state.xPhys.row(e).setZero();
    }
    if (state.xPhys.cols() != nvar) throw std::runtime_error("filtering_calculation: invalid xPhys columns");
}

Eigen::MatrixXd filtering_sensitivity(const Eigen::MatrixXd& df_dx_phys,
                                      const Filtering& filtering,
                                      const FilteringState& state) {
    const int nelem = static_cast<int>(df_dx_phys.rows());
    const int nvar = nvar_of(df_dx_phys);
    Eigen::MatrixXd tmp(nelem, nvar);
    const double denom = std::tanh(filtering.beta * filtering.T) + std::tanh(filtering.beta * (1.0 - filtering.T));
    for (int v = 0; v < nvar; ++v) {
        tmp.col(v) = (df_dx_phys.col(v).array() *
                      (filtering.beta * (filtering.beta * (state.xTilde.col(v).array() - filtering.T)).unaryExpr(
                                            [](double x) { return sech2(x); }) / denom) /
                      state.Hs_eigen.array()).matrix();
    }
    return state.H.transpose() * tmp;
}

