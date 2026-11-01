#pragma once

#include "myjson.hpp"
#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <string>

struct Analysis;
struct FEM;
struct StaticGeomericalNonlinear;

// One functional term = sum of its responses over ALL load steps.
// value is reset once per optimization iteration, not per load step.
struct StaticGeomericalNonlinearFunctionalBase {
    std::string function_name;
    double weight = 1.0;
    double initial_value = 0.0;
    double scale = 1.0;
    double value = 0.0;

    explicit StaticGeomericalNonlinearFunctionalBase(const myjson::json& j)
        : weight(myjson::double_value(j, "weight", 1.0)) {}
    virtual ~StaticGeomericalNonlinearFunctionalBase() = default;

    virtual void initialize(const FEM& fem, Analysis& analysis,
                            const Eigen::MatrixXd& xPhys);
    virtual void calculate(const FEM& fem, Analysis& analysis,
                           const Eigen::MatrixXd& xPhys);
    virtual double step_value(const FEM& fem, Analysis& analysis,
                              const Eigen::MatrixXd& xPhys) = 0;
    virtual Eigen::VectorXd dfdu(const FEM& fem, Analysis& analysis,
                                 const Eigen::MatrixXd& xPhys, int step) = 0;
    virtual Eigen::VectorXd dfdFext(const FEM& fem, Analysis& analysis,
                                    const Eigen::MatrixXd& xPhys, int step) = 0;
    virtual Eigen::VectorXd dfdx(const FEM& fem, Analysis& analysis,
                                 const Eigen::MatrixXd& xPhys, int val_id) = 0;

protected:
    static StaticGeomericalNonlinear& current_analysis(Analysis& analysis);
};

struct GeometricalNonlinearCompliance final : StaticGeomericalNonlinearFunctionalBase {
    explicit GeometricalNonlinearCompliance(const myjson::json& j)
        : StaticGeomericalNonlinearFunctionalBase(j) { function_name = "compliance"; }

    double step_value(const FEM& fem, Analysis& analysis,
                      const Eigen::MatrixXd& xPhys) override;
    Eigen::VectorXd dfdu(const FEM& fem, Analysis& analysis,
                         const Eigen::MatrixXd& xPhys, int step) override;
    Eigen::VectorXd dfdFext(const FEM& fem, Analysis& analysis,
                            const Eigen::MatrixXd& xPhys, int step) override;
    Eigen::VectorXd dfdx(const FEM&, Analysis&, const Eigen::MatrixXd&, int) override {
        // External loads do not explicitly depend on physical densities here.
        return Eigen::VectorXd();
    }
};

inline std::unique_ptr<StaticGeomericalNonlinearFunctionalBase>
InitializeStaticGeomericalNonlinearFunctional(const myjson::json& j) {
    const std::string name = myjson::lower_ascii(myjson::string_value(j, "function_name"));
    if (name == "compliance" || name == "static_compliance" ||
        name == "nonlinear_compliance")
        return std::make_unique<GeometricalNonlinearCompliance>(j);
    throw std::runtime_error("Unknown nonlinear functional: " + name);
}
