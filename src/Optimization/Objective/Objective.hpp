#pragma once

#include "Analysis/SensitivityBase.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Objective {
    std::vector<std::unique_ptr<GroupFunctionalBase>> groups;
    double value = 0.0;
    Eigen::MatrixXd sensitivity;

    Objective() = default;

    Objective(const myjson::json& objective_groups_json,
              const std::string& linear_backend,
              const std::vector<std::shared_ptr<Analysis>>& analysis_all,
              const myjson::json& run_json) {
        for (const auto& group_json : objective_groups_json) {
            groups.push_back(InitializeGroupFunctional(group_json,
                                                       linear_backend,
                                                       analysis_all,
                                                       run_json));
        }
    }

    void initialize(const FEM& fem,
                    const Eigen::MatrixXd& xPhys) {
        value = 0.0;
        for (auto& group : groups) {
            Analysis& analysis = *group->analysis;
            group->initialize_terms(fem, analysis, xPhys);
            value += group->value;
        }
    }

    void calculate(const FEM& fem,
                   const Eigen::MatrixXd& xPhys) {
        value = 0.0;
        for (auto& group : groups) {
            Analysis& analysis = *group->analysis;
            group->calculate_terms(fem, analysis, xPhys);
            value += group->value;
        }
    }

    void calculate_sensitivity(const FEM& fem,
                               const Eigen::MatrixXd& x,
                               const std::vector<int>& x_label,
                               const Filtering& filtering,
                               const OutputData& output,
                               int Iter) {
        sensitivity.resize(0, 0);
        for (auto& group : groups) {
            if (!group->sensitivity_method) throw std::runtime_error("Objective group has no sensitivity method");
            Analysis& analysis = *group->analysis;
            Eigen::MatrixXd group_derivative = group->sensitivity_method->run(fem, analysis, x, x_label, *group,
                                                                              filtering, output,
                                                                              Iter);
            if (sensitivity.size() == 0) sensitivity = std::move(group_derivative);
            else sensitivity.noalias() += group_derivative;
        }
    }
};
