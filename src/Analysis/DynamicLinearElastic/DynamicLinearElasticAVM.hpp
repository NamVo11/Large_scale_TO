#pragma once

#include "Analysis/DynamicLinearElastic/DynamicLinearElasticFunctional.hpp"
#include "Analysis/SensitivityBase.hpp"
#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"

#include <memory>
#include <vector>

struct DynamicLinearElasticAVM : GroupFunctionalBase {
    std::vector<std::unique_ptr<DynamicLinearElasticFunctionalBase>> terms;

    explicit DynamicLinearElasticAVM(const myjson::json& j) {
        analysis_id = myjson::int_value(j, "analysis_id", analysis_id);
        sens_type = myjson::string_value(j, "sens_type", sens_type);
        sens_function = myjson::string_value(j, "sens_function", sens_function);
        type = myjson::string_value(j, "type", type);
        sensitivity_method = InitializeSensitivityMethod(sens_type);

        for (const auto& term_json : j.value("terms", myjson::json::array())) {
            terms.push_back(InitializeDynamicLinearElasticFunctional(term_json));
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
                        int Iter) override;

private:
    void accumulate_terms();
};

struct EigenDynamicLinearElasticAVM : DynamicLinearElasticAVM {
    explicit EigenDynamicLinearElasticAVM(const myjson::json& j)
        : DynamicLinearElasticAVM(j) {}

    Eigen::MatrixXd AVM(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        const Filtering& filtering,
                        int Iter) override;
};

struct PetscDynamicLinearElasticAVM : DynamicLinearElasticAVM {
    explicit PetscDynamicLinearElasticAVM(const myjson::json& j)
        : DynamicLinearElasticAVM(j) {}

    Eigen::MatrixXd AVM(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        const Filtering& filtering,
                        int Iter) override;
};
