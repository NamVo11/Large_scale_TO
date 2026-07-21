#pragma once

#include "MatlabLikeData.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>

struct FilteringState {
    Eigen::MatrixXd xTilde;
    Eigen::MatrixXd xPhys;
    std::vector<std::vector<int>> neighbors;
    std::vector<std::vector<double>> weights;
    Eigen::SparseMatrix<double> H;
    std::vector<double> Hs;
    Eigen::VectorXd Hs_eigen;
};

void filtering_initilization(const FEM& fem, Filtering& filtering, FilteringState& state);
void filtering_calculation(const Eigen::MatrixXd& x,
                           const std::vector<int>& x_label,
                           const Filtering& filtering,
                           FilteringState& state);
Eigen::MatrixXd filtering_sensitivity(const Eigen::MatrixXd& df_dx_phys,
                                      const Filtering& filtering,
                                      const FilteringState& state);

void evaluate_objective(const FEM& fem,
                        std::vector<Analysis>& analysis_all,
                        const Eigen::MatrixXd& xPhys,
                        std::vector<ObjectiveGroup>& objective_groups,
                        double& obj_value);
void evaluate_constraint(const FEM& fem,
                         std::vector<Analysis>& analysis_all,
                         const Eigen::MatrixXd& xPhys,
                         std::vector<Constraint>& constraints,
                         std::vector<double>& constr_value_vec);
