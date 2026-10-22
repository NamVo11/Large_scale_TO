#pragma once

#include "Analysis/SensitivityBase.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Constraint {
    std::vector<std::unique_ptr<GroupFunctionalBase>> groups;
    std::vector<double> limits;
    std::vector<double> values;
    std::vector<Eigen::MatrixXd> sensitivity;

    Constraint() = default;

    Constraint(const myjson::json& constraints_json,
               const std::string& linear_backend,
               const std::vector<std::shared_ptr<Analysis>>& analysis_all,
               const myjson::json& analysis_all_json,
               const myjson::json& run_json) {
        for (const auto& constraint_json : constraints_json) {
            groups.push_back(InitializeGroupFunctional(constraint_json,
                                                       linear_backend,
                                                       analysis_all,
                                                       analysis_all_json,
                                                       run_json));
            limits.push_back(constraint_json.at("limit").get<double>());
            values.push_back(0.0);
            sensitivity.emplace_back();
        }
    }

    void calculate(const FEM& fem,
                   const Eigen::MatrixXd& xPhys) {
        for (std::size_t i = 0; i < groups.size(); ++i) {
            GroupFunctionalBase& group = *groups.at(i);
            Analysis& analysis = *group.analysis;
            group.calculate_terms(fem, analysis, xPhys);
            values.at(i) = group.value - limits.at(i);
        }
    }

    void calculate_sensitivity(const FEM& fem,
                               const Eigen::MatrixXd& x,
                               const std::vector<int>& x_label,
                               const Filtering& filtering,
                               const OutputData& output,
                               int Iter) {
        sensitivity.resize(groups.size());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            auto& group = groups.at(i);
            if (!group->sensitivity_method) throw std::runtime_error("Constraint group has no sensitivity method");
            Analysis& analysis = *group->analysis;
            sensitivity.at(i) = group->sensitivity_method->run(fem, analysis, x, x_label, *group,
                                                               filtering, output, Iter);
        }
    }
};
