#pragma once

#include "Optimization/Filtering/Filtering.hpp"
#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"
#include "myjson.hpp"

#include <Eigen/Dense>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class ROMBase;
struct Analysis;
struct FEM;
struct OutputData;

struct GroupFunctionalBase {
    int analysis_id = 1;
    std::string sens_type = "AVM";
    std::string sens_function;
    std::string type;

    std::shared_ptr<ROMBase> ROM;
    std::unique_ptr<SensitivityMethodBase> sensitivity_method;

    double value = 0.0;
    Eigen::MatrixXd d_explicit_dfdu_all;
    Eigen::MatrixXd d_explicit_dfdv_all;
    Eigen::MatrixXd d_explicit_dfda_all;
    Eigen::MatrixXd d_explicit_dfdFext_all;
    Eigen::MatrixXd d_explicit_df_dx;

    virtual ~GroupFunctionalBase() = default;

    virtual void initialize_terms(const FEM& fem,
                                  Analysis& analysis,
                                  const Eigen::MatrixXd& xPhys) = 0;

    virtual void calculate_terms(const FEM& fem,
                                 Analysis& analysis,
                                 const Eigen::MatrixXd& xPhys) = 0;

    virtual Eigen::MatrixXd AVM(const FEM& fem,
                                Analysis& analysis,
                                const Eigen::MatrixXd& x,
                                const std::vector<int>& x_label,
                                const Filtering& filtering,
                                int Iter) = 0;

    virtual void debug_FDM(const FEM& fem,
                           Analysis& analysis,
                           const Eigen::MatrixXd& x,
                           const std::vector<int>& x_label,
                           const Filtering& filtering,
                           const OutputData& output,
                           double fd_epsilon,
                           int max_checks,
                           int Iter,
                           const Eigen::MatrixXd& avm);

protected:
    void reset_accumulated_derivatives();
    void add_term_contribution(double alpha,
                               double term_value,
                               const Eigen::MatrixXd& dfdu_all,
                               const Eigen::MatrixXd& dfdv_all,
                               const Eigen::MatrixXd& dfda_all,
                               const Eigen::MatrixXd& dfdFext_all,
                               const Eigen::MatrixXd& df_dx);
};

std::unique_ptr<GroupFunctionalBase> InitializeGroupFunctional(const myjson::json& j,
                                                              const std::string& linear_backend,
                                                              const myjson::json& analysis_all_json,
                                                              const myjson::json& run_json);
