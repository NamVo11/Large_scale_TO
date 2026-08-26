#pragma once

#include "Analysis/SensitivityBase.hpp"
#include "Analysis/StaticLinearElastic/StaticLinearElasticFunctional.hpp"
#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"

#include <memory>
#include <vector>

struct StaticLinearElasticAVM : GroupFunctionalBase {
    std::vector<std::unique_ptr<StaticLinearElasticFunctionalBase>> terms;

    explicit StaticLinearElasticAVM(const myjson::json& j) {
        analysis_id = myjson::int_value(j, "analysis_id", analysis_id);
        sens_type = myjson::string_value(j, "sens_type", sens_type);
        sens_function = myjson::string_value(j, "sens_function", sens_function);
        type = myjson::string_value(j, "type", type);
        sensitivity_method = InitializeSensitivityMethod(sens_type);

        for (const auto& term_json : j.value("terms", myjson::json::array())) {
            terms.push_back(InitializeStaticLinearElasticFunctional(term_json));
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

struct EigenStaticLinearElasticAVM : StaticLinearElasticAVM {
    explicit EigenStaticLinearElasticAVM(const myjson::json& j)
        : StaticLinearElasticAVM(j) {}

    Eigen::MatrixXd AVM(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        const Filtering& filtering,
                        int Iter) override;
};

struct PetscStaticLinearElasticAVM : StaticLinearElasticAVM {
    explicit PetscStaticLinearElasticAVM(const myjson::json& j)
        : StaticLinearElasticAVM(j) {}

    Eigen::MatrixXd AVM(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        const Filtering& filtering,
                        int Iter) override;
};

inline std::unique_ptr<GroupFunctionalBase> InitializeStaticLinearElasticGroupFunctional(const myjson::json& j,
                                                                                        const std::string& backend) {
    if (backend == "eigen") return std::make_unique<EigenStaticLinearElasticAVM>(j);
    return std::make_unique<PetscStaticLinearElasticAVM>(j);
}
