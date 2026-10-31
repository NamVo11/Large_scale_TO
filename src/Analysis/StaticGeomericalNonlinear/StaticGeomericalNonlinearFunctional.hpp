#pragma once

#include "myjson.hpp"
#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <string>

struct Analysis;
struct FEM;

// One functional term computes ONE CURRENT load step, never the total trajectory.
// Its scale is set by Optimization::evaluate_nonlinear() AFTER collecting the
// complete reference trajectory at optimization iteration zero.
struct StaticGeomericalNonlinearFunctionalBase {
    std::string function_name;
    double weight = 1.0;
    double initial_value = 1.0;
    double scale = 1.0;
    double value = 0.0; // raw value of the current load step
    bool normalized = false;

    explicit StaticGeomericalNonlinearFunctionalBase(const myjson::json& j)
        : weight(myjson::double_value(j, "weight", 1.0)) {}
    virtual ~StaticGeomericalNonlinearFunctionalBase() = default;

    // Keeps the linear functional class pattern, but defers normalization.
    // The wrapper must normalize using the complete multi-step objective.
    virtual void initialize(const FEM& fem, Analysis& analysis,
                            const Eigen::MatrixXd& xPhys) {
        calculate(fem, analysis, xPhys);
    }
    virtual void calculate(const FEM& fem, Analysis& analysis,
                           const Eigen::MatrixXd& xPhys) = 0;
    virtual Eigen::VectorXd dfdu(const FEM& fem, Analysis& analysis,
                                 const Eigen::MatrixXd& xPhys, int istep) = 0;
    virtual Eigen::VectorXd dfdFext(const FEM& fem, Analysis& analysis,
                                    const Eigen::MatrixXd& xPhys, int istep) = 0;
    virtual Eigen::VectorXd dfdx(const FEM& fem, Analysis& analysis,
                                 const Eigen::MatrixXd& xPhys, int val_id) = 0;
};

struct GeometricalNonlinearCompliance final : StaticGeomericalNonlinearFunctionalBase {
    explicit GeometricalNonlinearCompliance(const myjson::json& j)
        : StaticGeomericalNonlinearFunctionalBase(j) { function_name = "compliance"; }
    void calculate(const FEM& fem, Analysis& analysis,
                   const Eigen::MatrixXd& xPhys) override;
    Eigen::VectorXd dfdu(const FEM& fem, Analysis& analysis,
                         const Eigen::MatrixXd& xPhys, int istep) override;
    Eigen::VectorXd dfdFext(const FEM& fem, Analysis& analysis,
                            const Eigen::MatrixXd& xPhys, int istep) override;
    Eigen::VectorXd dfdx(const FEM&, Analysis&, const Eigen::MatrixXd&, int) override {
        // External point loads are independent of design density for this model.
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
