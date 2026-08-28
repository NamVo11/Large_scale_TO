#pragma once

#include "Analysis/DynamicLinearElastic/DynamicLinearElasticFunctional.hpp"
#include "Analysis/SensitivityBase.hpp"
#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"

#include <memory>
#include <vector>

struct DynamicLinearElasticAVM : GroupFunctionalBase {
    std::vector<std::unique_ptr<DynamicLinearElasticFunctionalBase>> terms;

    DynamicLinearElasticAVM(const myjson::json& j,
                            const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                            const myjson::json& run_json)
        : GroupFunctionalBase(j, analysis_all, run_json) {
        if (!std::dynamic_pointer_cast<DynamicLinearElastic>(analysis)) {
            throw std::runtime_error("DynamicLinearElasticAVM requires dynamic_elastic_linear analysis, id: " +
                                     std::to_string(analysis_id));
        }
        for (const auto& term_json : j.at("terms")) {
            terms.push_back(InitializeDynamicLinearElasticFunctional(term_json));
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

    Eigen::VectorXd dfdv(const FEM& fem,
                         Analysis& analysis,
                         const Eigen::MatrixXd& xPhys,
                         int istep);

    Eigen::VectorXd dfda(const FEM& fem,
                         Analysis& analysis,
                         const Eigen::MatrixXd& xPhys,
                         int istep);

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

struct EigenDynamicLinearElasticAVM : DynamicLinearElasticAVM {
    EigenDynamicLinearElasticAVM(const myjson::json& j,
                                 const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                                 const myjson::json& run_json)
        : DynamicLinearElasticAVM(j, analysis_all, run_json) {
        if (!std::dynamic_pointer_cast<EigenDynamicLinearElastic>(analysis)) {
            throw std::runtime_error("EigenDynamicLinearElasticAVM requires EigenDynamicLinearElastic analysis, id: " +
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

struct PetscDynamicLinearElasticAVM : DynamicLinearElasticAVM {
    PetscDynamicLinearElasticAVM(const myjson::json& j,
                                 const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                                 const myjson::json& run_json)
        : DynamicLinearElasticAVM(j, analysis_all, run_json) {
        if (!std::dynamic_pointer_cast<PetscDynamicLinearElastic>(analysis)) {
            throw std::runtime_error("PetscDynamicLinearElasticAVM requires PetscDynamicLinearElastic analysis, id: " +
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
