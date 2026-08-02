#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <petscksp.h>

struct Filtering {
    double rmin = 3.0;
    double beta = 1.0;
    int beta_loop = 50;
    double beta_max = 8.0;
    double T = 0.5;
};

struct Algorithm {
    std::string type = "MMA";
    double move = 0.1;
};

struct ObjectiveTerm {
    std::string function_name;
    double weight = 1.0;
    double value_initial = 0.0;
    double value = 0.0;
    Eigen::MatrixXd dfdu_all;
    Eigen::MatrixXd dfdv_all;
    Eigen::MatrixXd dfda_all;
    Eigen::MatrixXd dfdFext_all;
    Eigen::MatrixXd df_dx;
};

struct ObjectiveGroup {
    int analysis_id = 1;
    std::string sens_type;
    std::string sens_function;
    std::string type;
    std::shared_ptr<ROMBase> ROM;
    std::vector<ObjectiveTerm> terms;
    double value = 0.0;
    Eigen::MatrixXd d_explicit_dfdu_all;
    Eigen::MatrixXd d_explicit_dfdv_all;
    Eigen::MatrixXd d_explicit_dfda_all;
    Eigen::MatrixXd d_explicit_dfdFext_all;
    Eigen::MatrixXd d_explicit_df_dx;
};

struct Constraint {
    int analysis_id = 1;
    std::string function_name;
    double limit = 0.0;
    std::string sens_type;
    std::string sens_function;
    std::string type;
    int mat_id = 0;
    double value_initial = 0.0;
    double value = 0.0;
    Eigen::MatrixXd d_explicit_dfdu_all;
    Eigen::MatrixXd d_explicit_dfdv_all;
    Eigen::MatrixXd d_explicit_dfda_all;
    Eigen::MatrixXd d_explicit_dfdFext_all;
    Eigen::MatrixXd d_explicit_df_dx;
};

double RAMP(const std::vector<double>& para, double x, double p);

void init_mat(const std::vector<Material>& Mat, Analysis& A);
void init_interp(const Interp& interp, Analysis& A);
void attach_bcs_to_analysis_by_id(Analysis& A,
                                  const std::vector<BoundaryCondition>& Dirich,
                                  const std::vector<BoundaryCondition>& Neum);
Eigen::MatrixXd init_design_variables(const FEM& fem,
                                      const std::vector<Region>& regions,
                                      const std::vector<Material>& Mat,
                                      double x0,
                                      std::vector<int>& x_label);

double bc_value(const BoundaryCondition& BC, const Analysis& A, int istep);
