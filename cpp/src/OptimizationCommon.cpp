#include "OptimizationCommon.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
}

Eigen::MatrixXd zeros(int nrow, int ncol) {
    return Eigen::MatrixXd::Zero(nrow, ncol);
}

void add_scaled(Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double scale) {
    if (b.size() == 0) return;
    if (a.size() == 0) {
        a = scale * b;
        return;
    }
    a.noalias() += scale * b;
}

double sech2(double x) {
    const double c = std::cosh(x);
    return 1.0 / (c * c);
}

void L2_compliance_elastic(const FEM& fem,
                           const Analysis& analysis,
                           const Eigen::MatrixXd& xPhys,
                           ObjectiveTerm& term) {
    const int ndof = fem.ndof;
    const int nstep = analysis.Nstep;
    const int nvar = nvar_of(xPhys);
    term.dfdu_all = zeros(ndof, nstep + 1);
    term.dfdv_all = zeros(ndof, nstep + 1);
    term.dfda_all = zeros(ndof, nstep + 1);
    term.dfdFext_all = zeros(ndof, nstep + 1);
    term.df_dx = zeros(fem.NElem, nvar);
    term.value = 0.0;

    for (int k = 0; k <= nstep; ++k) {
        const auto u = analysis.results.U.col(k);
        const auto f = analysis.results.F_ext.col(k);
        const double s = f.dot(u);
        term.value += s * s;
        term.dfdu_all.col(k).noalias() = 2.0 * s * f;
        term.dfdFext_all.col(k).noalias() = 2.0 * s * u;
        for (int d : analysis.results.SPCdofs) {
            term.dfdu_all(d, k) = 0.0;
            term.dfdv_all(d, k) = 0.0;
            term.dfda_all(d, k) = 0.0;
        }
        for (int d : analysis.results.free) term.dfdFext_all(d, k) = 0.0;
    }
}

void volume_fraction(const FEM& fem,
                     const Analysis& analysis,
                     const Eigen::MatrixXd& xPhys,
                     Constraint& c) {
    const int ndof = fem.ndof;
    const int nstep = analysis.Nstep;
    const int nvar = nvar_of(xPhys);
    const int val_id = c.mat_id - 2;
    if (val_id < 0 || val_id >= nvar) throw std::runtime_error("volume_fraction: invalid mat_id");
    c.value = xPhys.col(val_id).mean();
    c.d_explicit_dfdu_all = zeros(ndof, nstep + 1);
    c.d_explicit_dfdv_all = zeros(ndof, nstep + 1);
    c.d_explicit_dfda_all = zeros(ndof, nstep + 1);
    c.d_explicit_dfdFext_all = zeros(ndof, nstep + 1);
    c.d_explicit_df_dx = zeros(fem.NElem, nvar);
    for (int e = 0; e < fem.NElem; ++e) c.d_explicit_df_dx(e, val_id) = 1.0 / static_cast<double>(fem.NElem);
}

} // namespace

void filtering_initilization(const FEM& fem, Filtering& filtering, FilteringState& state) {
    const int r = std::max(0, static_cast<int>(std::ceil(filtering.rmin)) - 1);
    state.neighbors.assign(static_cast<std::size_t>(fem.NElem), {});
    state.weights.assign(static_cast<std::size_t>(fem.NElem), {});
    state.Hs.assign(static_cast<std::size_t>(fem.NElem), 0.0);
    auto eid2 = [&fem](int i, int j) { return i * fem.nely + j; };
    auto eid3 = [&fem](int i, int j, int k) { return k * fem.nelx * fem.nely + i * fem.nely + j; };
    if (fem.dim == 2) {
        for (int i1 = 0; i1 < fem.nelx; ++i1) {
            for (int j1 = 0; j1 < fem.nely; ++j1) {
                const int e1 = eid2(i1, j1);
                for (int i2 = std::max(0, i1 - r); i2 <= std::min(fem.nelx - 1, i1 + r); ++i2) {
                    for (int j2 = std::max(0, j1 - r); j2 <= std::min(fem.nely - 1, j1 + r); ++j2) {
                        const double dist = std::sqrt((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2));
                        const double w = filtering.rmin - dist;
                        if (w > 0.0) {
                            state.neighbors[static_cast<std::size_t>(e1)].push_back(eid2(i2, j2));
                            state.weights[static_cast<std::size_t>(e1)].push_back(w);
                            state.Hs[static_cast<std::size_t>(e1)] += w;
                        }
                    }
                }
            }
        }
    } else {
        for (int k1 = 0; k1 < fem.nelz; ++k1) {
            for (int i1 = 0; i1 < fem.nelx; ++i1) {
                for (int j1 = 0; j1 < fem.nely; ++j1) {
                    const int e1 = eid3(i1, j1, k1);
                    for (int k2 = std::max(0, k1 - r); k2 <= std::min(fem.nelz - 1, k1 + r); ++k2) {
                        for (int i2 = std::max(0, i1 - r); i2 <= std::min(fem.nelx - 1, i1 + r); ++i2) {
                            for (int j2 = std::max(0, j1 - r); j2 <= std::min(fem.nely - 1, j1 + r); ++j2) {
                                const double dist = std::sqrt((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2) + (k1 - k2) * (k1 - k2));
                                const double w = filtering.rmin - dist;
                                if (w > 0.0) {
                                    state.neighbors[static_cast<std::size_t>(e1)].push_back(eid3(i2, j2, k2));
                                    state.weights[static_cast<std::size_t>(e1)].push_back(w);
                                    state.Hs[static_cast<std::size_t>(e1)] += w;
                                }
                            }
                        }
                    }
                }
            }
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

void evaluate_objective(const FEM& fem,
                        std::vector<Analysis>& analysis_all,
                        const Eigen::MatrixXd& xPhys,
                        std::vector<ObjectiveGroup>& objective_groups,
                        double& obj_value) {
    obj_value = 0.0;
    for (auto& group : objective_groups) {
        Analysis& analysis = analysis_all[static_cast<std::size_t>(group.analysis_id - 1)];
        group.value = 0.0;
        group.d_explicit_dfdu_all.resize(0, 0);
        group.d_explicit_dfdv_all.resize(0, 0);
        group.d_explicit_dfda_all.resize(0, 0);
        group.d_explicit_dfdFext_all.resize(0, 0);
        group.d_explicit_df_dx.resize(0, 0);
        for (auto& term : group.terms) {
            if (lower(term.function_name) != "l2_compliance_elastic") throw std::runtime_error("Unknown objective term");
            L2_compliance_elastic(fem, analysis, xPhys, term);
            if (term.value_initial == 0.0) term.value_initial = term.value;
            const double alpha = term.weight / term.value_initial;
            group.value += alpha * term.value;
            add_scaled(group.d_explicit_dfdu_all, term.dfdu_all, alpha);
            add_scaled(group.d_explicit_dfdv_all, term.dfdv_all, alpha);
            add_scaled(group.d_explicit_dfda_all, term.dfda_all, alpha);
            add_scaled(group.d_explicit_dfdFext_all, term.dfdFext_all, alpha);
            add_scaled(group.d_explicit_df_dx, term.df_dx, alpha);
        }
        obj_value += group.value;
    }
}

void evaluate_constraint(const FEM& fem,
                         std::vector<Analysis>& analysis_all,
                         const Eigen::MatrixXd& xPhys,
                         std::vector<Constraint>& constraints,
                         std::vector<double>& constr_value_vec) {
    constr_value_vec.assign(constraints.size(), 0.0);
    for (std::size_t i = 0; i < constraints.size(); ++i) {
        Constraint& c = constraints[i];
        Analysis& analysis = analysis_all[static_cast<std::size_t>(c.analysis_id - 1)];
        if (lower(c.function_name) != "volume_fraction") throw std::runtime_error("Unknown constraint");
        volume_fraction(fem, analysis, xPhys, c);
        if (c.value_initial == 0.0) c.value_initial = c.value;
        constr_value_vec[i] = c.value / c.value_initial - c.limit;
        c.d_explicit_df_dx *= 1.0 / c.value_initial;
    }
}
