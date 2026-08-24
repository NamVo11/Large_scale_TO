#pragma once

#include "Analysis/SensitivityBase.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

struct Constraint {
    std::vector<std::unique_ptr<GroupFunctionalBase>> groups;
    std::vector<double> limits;
    std::vector<double> values;
    std::vector<Eigen::MatrixXd> sensitivity;

    Constraint() = default;

    void add_group(std::unique_ptr<GroupFunctionalBase> group, double limit) {
        groups.push_back(std::move(group));
        limits.push_back(limit);
        values.push_back(0.0);
        sensitivity.emplace_back();
    }

    void calculate(const FEM& fem,
                   std::vector<std::unique_ptr<Analysis>>& analysis_all,
                   const Eigen::MatrixXd& xPhys) {
        for (std::size_t i = 0; i < groups.size(); ++i) {
            GroupFunctionalBase& group = *groups.at(i);
            Analysis& analysis = *analysis_all.at(static_cast<std::size_t>(group.analysis_id - 1));
            group.calculate_terms(fem, analysis, xPhys);
            values.at(i) = group.value - limits.at(i);
        }
    }

    void calculate_sensitivity(const FEM& fem,
                               std::vector<std::unique_ptr<Analysis>>& analysis_all,
                               const Eigen::MatrixXd& x,
                               const std::vector<int>& x_label,
                               const Filtering& filtering,
                               const FilteringState& filtering_state,
                               const OutputData& output,
                               double fd_epsilon,
                               int max_checks,
                               int Iter) {
        sensitivity.resize(groups.size());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            auto& group = groups.at(i);
            if (!group->sensitivity_method) throw std::runtime_error("Constraint group has no sensitivity method");
            Analysis& analysis = *analysis_all.at(static_cast<std::size_t>(group->analysis_id - 1));
            sensitivity.at(i) = group->sensitivity_method->run(fem, analysis, x, x_label, *group,
                                                               filtering, filtering_state, output,
                                                               fd_epsilon, max_checks, Iter);
        }
    }
};
