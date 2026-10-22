#pragma once

#include "Analysis/SensitivityBase.hpp"
#include "Analysis/StaticLinearElastic/StaticLinearElasticFunctional.hpp"
#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"

#include <memory>
#include <vector>

struct StaticLinearElasticAVM : GroupFunctionalBase {
    std::vector<std::unique_ptr<StaticLinearElasticFunctionalBase>> terms;

    StaticLinearElasticAVM(const myjson::json& j,
                           const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                           const myjson::json& analysis_all_json,
                           const myjson::json& run_json)
        : GroupFunctionalBase(j, analysis_all, analysis_all_json, run_json) {
        if (!std::dynamic_pointer_cast<StaticLinearElastic>(analysis)) {
            throw std::runtime_error("StaticLinearElasticAVM requires static_elastic_linear analysis, id: " +
                                     std::to_string(analysis_id));
        }
        for (const auto& term_json : j.at("terms")) {
            terms.push_back(InitializeStaticLinearElasticFunctional(term_json));
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

struct EigenStaticLinearElasticAVM : StaticLinearElasticAVM {
    EigenStaticLinearElasticAVM(const myjson::json& j,
                                const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                                const myjson::json& analysis_all_json,
                                const myjson::json& run_json)
        : StaticLinearElasticAVM(j, analysis_all, analysis_all_json, run_json) {
        if (!std::dynamic_pointer_cast<EigenStaticLinearElastic>(analysis)) {
            throw std::runtime_error("EigenStaticLinearElasticAVM requires EigenStaticLinearElastic analysis, id: " +
                                     std::to_string(analysis_id));
        }
    }

    Eigen::MatrixXd AVM(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        const Filtering& filtering,
                        int Iter) override;
};

struct PetscStaticLinearElasticAVM : StaticLinearElasticAVM {
    PetscStaticLinearElasticAVM(const myjson::json& j,
                                const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                                const myjson::json& analysis_all_json,
                                const myjson::json& run_json)
        : StaticLinearElasticAVM(j, analysis_all, analysis_all_json, run_json) {
        if (!std::dynamic_pointer_cast<PetscStaticLinearElastic>(analysis)) {
            throw std::runtime_error("PetscStaticLinearElasticAVM requires PetscStaticLinearElastic analysis, id: " +
                                     std::to_string(analysis_id));
        }
    }

    Eigen::MatrixXd AVM(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        const Filtering& filtering,
                        int Iter) override;
};
