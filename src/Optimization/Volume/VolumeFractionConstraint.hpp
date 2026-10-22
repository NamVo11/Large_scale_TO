#pragma once

#include "Analysis/SensitivityBase.hpp"
#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"
#include "Optimization/Volume/VolumeFunctional.hpp"

#include <memory>
#include <vector>

struct VolumeFractionConstraint : GroupFunctionalBase {
    std::vector<std::unique_ptr<VolumeFunctionalBase>> terms;

    VolumeFractionConstraint(const myjson::json& j,
                             const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                             const myjson::json& analysis_all_json,
                             const myjson::json& run_json)
        : GroupFunctionalBase(j, analysis_all, analysis_all_json, run_json) {
        for (const auto& term_json : j.at("terms")) {
            terms.push_back(InitializeVolumeFunctional(term_json));
        }
    }

    void initialize_terms(const FEM& fem,
                          Analysis& analysis,
                          const Eigen::MatrixXd& xPhys) override;

    void calculate_terms(const FEM& fem,
                         Analysis& analysis,
                         const Eigen::MatrixXd& xPhys) override;

    Eigen::VectorXd dfdu(const FEM& fem,
                         Analysis& analysis,
                         const Eigen::MatrixXd& xPhys,
                         int istep) override;

    Eigen::VectorXd dfdFext(const FEM& fem,
                            Analysis& analysis,
                            const Eigen::MatrixXd& xPhys,
                            int istep) override;

    Eigen::VectorXd dfdx(const FEM& fem,
                         Analysis& analysis,
                         const Eigen::MatrixXd& xPhys,
                         int val_id) override;

    Eigen::MatrixXd AVM(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        const Filtering& filtering,
                        int Iter) override;

};
