#pragma once

#include "Analysis/SensitivityBase.hpp"
#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearFunctional.hpp"

#include <memory>
#include <vector>

// Same grouping hierarchy as StaticLinearElasticAVM.
struct StaticGeomericalNonlinearAVM : GroupFunctionalBase {
    std::vector<std::unique_ptr<StaticGeomericalNonlinearFunctionalBase>> terms;

    StaticGeomericalNonlinearAVM(const myjson::json& j,
                                const std::vector<std::shared_ptr<Analysis>>& analyses,
                                const myjson::json& analyses_json,
                                const myjson::json& run_json);

    void initialize_terms(const FEM&, Analysis&, const Eigen::MatrixXd&) override;
    void calculate_terms(const FEM&, Analysis&, const Eigen::MatrixXd&) override;
    double step_value(const FEM&, Analysis&, const Eigen::MatrixXd&);
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
    // Main sensitivity implementation, called by SensitivityMethod::run().
    // ONE load step per call, same as the linear AVM's interface.
    Eigen::MatrixXd AVM(const FEM&, Analysis&, const Eigen::MatrixXd&,
                        const std::vector<int>&, const Filtering&, int) override;

private:
    static int sens_threads();
    Eigen::MatrixXd explicit_gradient(const FEM& fem, StaticGeomericalNonlinear& analysis,
                                      const Eigen::MatrixXd& xPhys, const Eigen::VectorXd& lambda);
};
