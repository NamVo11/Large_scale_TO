#pragma once

#include "Analysis/Analysis.hpp"
#include "Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.hpp"
#include "Analysis/SensitivityBase.hpp"
#include "Analysis/StaticLinearElastic/StaticLinearElasticAVM.hpp"
#include "Optimization/Algorithm/Algorithm.hpp"
#include "Optimization/Constraint/Constraint.hpp"
#include "Optimization/Filtering/Filtering.hpp"
#include "Optimization/Objective/Objective.hpp"
#include "Optimization/Volume/VolumeFractionConstraint.hpp"
#include "Output/OutputData.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct OptimizationHistory {
    std::vector<int> Iter;
    std::vector<double> objective;
    std::vector<double> constraint;
    std::vector<double> change;
    std::vector<double> rms_change;
    std::vector<double> forward_time;
    std::vector<double> objective_time;
    std::vector<double> constraint_time;
    std::vector<double> objective_sensitivity_time;
    std::vector<double> constraint_sensitivity_time;
    std::vector<double> mma_time;
    std::vector<double> iteration_time;
};

struct Optimization {
    Eigen::MatrixXd x0;
    Eigen::MatrixXd x;
    std::vector<int> x_label;
    std::vector<std::shared_ptr<Analysis>> analysis_all;
    int MaxIter = 300;
    double Tol = 0.0;
    std::unique_ptr<Filtering> filtering;
    std::unique_ptr<Algorithm> algorithm;
    myjson::json algorithm_json;
    OutputData Output;
    Objective objective;
    Constraint constraint;
    OptimizationHistory history;

    Optimization() = default;

    Optimization(const myjson::json& optimization_json,
                 const myjson::json& output_json, const myjson::json& run_json,
                 const std::string& input_json_path, const FEM& fem,
                 const std::vector<std::shared_ptr<Analysis>>& analysis_all_in,
                 const myjson::json& analysis_all_json) {
        if (analysis_all_in.empty()) throw std::runtime_error("Optimization: analysis_all is empty");
        analysis_all = analysis_all_in;
        init_design_variables(fem, optimization_json.at("x0").get<double>());
        MaxIter = optimization_json.at("MaxIter").get<int>();
        Tol = optimization_json.at("Tol").get<double>();
        filtering = InitializeFiltering(optimization_json.at("Filtering"));
        algorithm_json = optimization_json.at("Algorithm");

        const std::string linear_backend = run_json.at("linear_backend").get<std::string>();
        objective = Objective(optimization_json.at("ObjectiveGroups"),
                              linear_backend,
                              analysis_all,
                              analysis_all_json,
                              run_json);
        constraint = Constraint(optimization_json.at("Constraints"),
                                linear_backend,
                                analysis_all,
                                analysis_all_json,
                                run_json);
        Output = OutputData(output_json, run_json, input_json_path);
    }

    void run(const FEM& fem);
    void run_nonlinear(const FEM& fem);
    void evaluate_nonlinear(const FEM& fem, int Iter);
    void init_design_variables(const FEM& fem, double x0);
    static std::vector<double> flatten(const Eigen::MatrixXd& values);
    static Eigen::MatrixXd unflatten(const std::vector<double>& values, int nelem, int nvar);
};

inline Optimization InitializeOptimization(const myjson::json& optimization_json,
                                           const myjson::json& output_json,
                                           const myjson::json& run_json,
                                           const std::string& input_json_path,
                                           const FEM& fem,
                                           const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                                           const myjson::json& analysis_all_json) {
    return Optimization(optimization_json,
                        output_json,
                        run_json,
                        input_json_path,
                        fem,
                        analysis_all,
                        analysis_all_json);
}
