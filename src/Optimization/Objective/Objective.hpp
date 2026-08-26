#pragma once

#include "Analysis/SensitivityBase.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

struct Objective {
    std::vector<std::unique_ptr<GroupFunctionalBase>> groups;
    double value = 0.0;
    Eigen::MatrixXd sensitivity;

    Objective() = default;

    void add_group(std::unique_ptr<GroupFunctionalBase> group) {
        groups.push_back(std::move(group));
    }

    void initialize(const FEM& fem,
                    std::vector<std::unique_ptr<Analysis>>& analysis_all,
                    const Eigen::MatrixXd& xPhys) {
        value = 0.0;
        for (auto& group : groups) {
            Analysis& analysis = *analysis_all.at(static_cast<std::size_t>(group->analysis_id - 1));
            group->initialize_terms(fem, analysis, xPhys);
            value += group->value;
        }
    }

    void calculate(const FEM& fem,
                   std::vector<std::unique_ptr<Analysis>>& analysis_all,
                   const Eigen::MatrixXd& xPhys) {
        value = 0.0;
        for (auto& group : groups) {
            Analysis& analysis = *analysis_all.at(static_cast<std::size_t>(group->analysis_id - 1));
            group->calculate_terms(fem, analysis, xPhys);
            value += group->value;
        }
    }

    void calculate_sensitivity(const FEM& fem,
                               std::vector<std::unique_ptr<Analysis>>& analysis_all,
                               const Eigen::MatrixXd& x,
                               const std::vector<int>& x_label,
                               const Filtering& filtering,
                               const OutputData& output,
                               int Iter) {
        sensitivity.resize(0, 0);
        for (auto& group : groups) {
            if (!group->sensitivity_method) throw std::runtime_error("Objective group has no sensitivity method");
            Analysis& analysis = *analysis_all.at(static_cast<std::size_t>(group->analysis_id - 1));
            Eigen::MatrixXd group_derivative = group->sensitivity_method->run(fem, analysis, x, x_label, *group,
                                                                              filtering, output,
                                                                              Iter);
            if (sensitivity.size() == 0) sensitivity = std::move(group_derivative);
            else sensitivity.noalias() += group_derivative;
        }
    }
};
