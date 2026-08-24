#pragma once

#include "Analysis/Analysis.hpp"
#include "myjson.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>

struct Filtering {
    struct FilteringState {
        Eigen::MatrixXd xTilde;
        Eigen::MatrixXd xPhys;
        std::vector<std::vector<int>> neighbors;
        std::vector<std::vector<double>> weights;
        Eigen::SparseMatrix<double> H;
        std::vector<double> Hs;
        Eigen::VectorXd Hs_eigen;
    };

    double rmin = 3.0;
    double beta = 1.0;
    int beta_loop = 50;
    double beta_max = 8.0;
    double T = 0.5;

    Filtering() = default;

    explicit Filtering(const myjson::json& j) {
        rmin = myjson::double_value(j, "rmin", rmin);
        beta = myjson::double_value(j, "beta", beta);
        beta_loop = myjson::int_value(j, "beta_loop", beta_loop);
        beta_max = myjson::double_value(j, "beta_max", beta_max);
        T = myjson::double_value(j, "T", T);
    }

    void initialize(const FEM& fem, FilteringState& state);
    void calculate(const Eigen::MatrixXd& x,
                   const std::vector<int>& x_label,
                   FilteringState& state) const;
    Eigen::MatrixXd sensitivity(const Eigen::MatrixXd& df_dx_phys,
                                const FilteringState& state) const;
};

using FilteringState = Filtering::FilteringState;
