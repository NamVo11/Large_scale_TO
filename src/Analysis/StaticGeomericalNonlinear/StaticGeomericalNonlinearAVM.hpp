#pragma once

#include "Analysis/SensitivityBase.hpp"
#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearFunctional.hpp"

#include <memory>
#include <vector>

// As in StaticLinearElasticAVM: common nonlinear group logic in base;
// backend-specific solver and element derivative live in Eigen derived class.
// No Petsc nonlinear AVM class is created in this stage.
struct StaticGeomericalNonlinearAVM : GroupFunctionalBase {
    std::vector<std::unique_ptr<StaticGeomericalNonlinearFunctionalBase>> terms;

    StaticGeomericalNonlinearAVM(const myjson::json& j,
                                const std::vector<std::shared_ptr<Analysis>>& analyses,
                                const myjson::json& analyses_json,
                                const myjson::json& run_json);

    void initialize_terms(const FEM&, Analysis&, const Eigen::MatrixXd&) override;
    void calculate_terms(const FEM&, Analysis&, const Eigen::MatrixXd&) override;
    Eigen::VectorXd dfdu(const FEM&, Analysis&, const Eigen::MatrixXd&, int) override;
    Eigen::VectorXd dfdFext(const FEM&, Analysis&, const Eigen::MatrixXd&, int) override;
    Eigen::VectorXd dfdx(const FEM&, Analysis&, const Eigen::MatrixXd&, int) override;
    Eigen::MatrixXd AVM(const FEM&, Analysis&, const Eigen::MatrixXd&,
                        const std::vector<int>&, const Filtering&, int) override;
};

struct EigenStaticGeomericalNonlinearAVM final : StaticGeomericalNonlinearAVM {
    EigenStaticGeomericalNonlinearAVM(const myjson::json& j,
                                     const std::vector<std::shared_ptr<Analysis>>& analyses,
                                     const myjson::json& analyses_json,
                                     const myjson::json& run_json);
    // Each term's raw dJ_n/d(xPhys). No filtering, no normalization.
    std::vector<Eigen::MatrixXd> calculate_raw_gradients(
        const FEM&, StaticGeomericalNonlinear&, const Eigen::MatrixXd& xPhys);
    Eigen::MatrixXd AVM(const FEM&, Analysis&, const Eigen::MatrixXd&,
                        const std::vector<int>&, const Filtering&, int) override;
};
