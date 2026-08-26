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
    std::vector<std::unique_ptr<Analysis>> analysis_all;
    int MaxIter = 300;
    double Tol = 0.0;
    std::unique_ptr<Filtering> filtering;
    std::unique_ptr<Algorithm> algorithm;
    myjson::json algorithm_json;
    OutputData Output;
    Objective objective;
    Constraint constraint;
    OptimizationHistory history;

    void run(const FEM& fem);
    void init_design_variables(const FEM& fem, double x0);
};

inline myjson::json analysis_json_by_id(const myjson::json& analysis_all_json, int analysis_id) {
    for (const auto& analysis_json : analysis_all_json) {
        if (myjson::int_value(analysis_json, "id", 0) == analysis_id) return analysis_json;
    }
    throw std::runtime_error("Analysis id not found in JSON: " + std::to_string(analysis_id));
}

inline Optimization InitializeOptimization(const myjson::json& optimization_json,
                                           const myjson::json& analysis_all_json,
                                           const myjson::json& output_json,
                                           const myjson::json& run_json,
                                           const std::string& input_json_path,
                                           const FEM& fem,
                                           std::vector<std::unique_ptr<Analysis>> analysis_all) {
    if (analysis_all.empty()) throw std::runtime_error("InitializeOptimization: analysis_all is empty");
    Optimization opt;
    opt.init_design_variables(fem, myjson::double_value(optimization_json, "x0", 0.5));
    opt.MaxIter = myjson::int_value(optimization_json, "MaxIter", opt.MaxIter);
    opt.Tol = myjson::double_value(optimization_json, "Tol", opt.Tol);
    opt.filtering = InitializeFiltering(optimization_json.at("Filtering"));
    opt.algorithm_json = optimization_json.at("Algorithm");

    const std::string linear_backend = myjson::string_value(run_json, "linear_backend", "petsc");
    for (const auto& group_json : optimization_json.value("ObjectiveGroups", myjson::json::array())) {
        opt.objective.add_group(InitializeGroupFunctional(group_json, linear_backend));
    }
    for (const auto& constraint_json : optimization_json.value("Constraints", myjson::json::array())) {
        opt.constraint.add_group(InitializeGroupFunctional(constraint_json, linear_backend),
                                 constraint_limit_from_json(constraint_json));
    }

    opt.Output = InitializeOutput(output_json, run_json, input_json_path);
    for (auto& group : opt.objective.groups) {
        myjson::json target_analysis = analysis_all_json.empty() ? myjson::json::object() : analysis_json_by_id(analysis_all_json, group->analysis_id);
        group->ROM = InitializeROM(target_analysis.at("ROM"), run_json);
    }
    for (auto& group : opt.constraint.groups) {
        myjson::json target_analysis = analysis_all_json.empty() ? myjson::json::object() : analysis_json_by_id(analysis_all_json, group->analysis_id);
        group->ROM = InitializeROM(target_analysis.at("ROM"), run_json);
    }
    opt.analysis_all = std::move(analysis_all);
    return opt;
}
