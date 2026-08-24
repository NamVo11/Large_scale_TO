#pragma once

#include "Analysis/SensitivityBase.hpp"
#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"
#include "Optimization/Volume/VolumeFunctional.hpp"

#include <memory>
#include <vector>

struct VolumeFractionConstraint : GroupFunctionalBase {
    std::vector<std::unique_ptr<VolumeFunctionalBase>> terms;

    explicit VolumeFractionConstraint(const myjson::json& j) {
        analysis_id = myjson::int_value(j, "analysis_id", analysis_id);
        sens_type = myjson::string_value(j, "sens_type", sens_type);
        sens_function = myjson::string_value(j, "sens_function", sens_function);
        type = myjson::string_value(j, "type", type);
        sensitivity_method = InitializeSensitivityMethod(sens_type);
        const myjson::json terms_json = j.value("terms", myjson::json::array());
        if (terms_json.empty()) {
            terms.push_back(InitializeVolumeFunctional(j));
        } else {
            for (const auto& term_json : terms_json) {
                terms.push_back(InitializeVolumeFunctional(term_json));
            }
        }
    }

    void initialize_terms(const FEM& fem,
                          Analysis& analysis,
                          const Eigen::MatrixXd& xPhys) override;

    void calculate_terms(const FEM& fem,
                         Analysis& analysis,
                         const Eigen::MatrixXd& xPhys) override;

    Eigen::MatrixXd AVM(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        const Filtering& filtering,
                        const FilteringState& filtering_state,
                        int Iter) override;

private:
    void accumulate_terms();
};
