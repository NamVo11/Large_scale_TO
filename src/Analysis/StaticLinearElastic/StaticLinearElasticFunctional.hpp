#pragma once

#include "myjson.hpp"

#include <Eigen/Dense>

#include <memory>
#include <stdexcept>
#include <string>

struct Analysis;
struct FEM;

struct StaticLinearElasticFunctionalBase {
    std::string function_name;
    double weight = 1.0;
    double initial_value = 1.0;
    double scale = 1.0;
    double value = 0.0;

    StaticLinearElasticFunctionalBase() = default;
    explicit StaticLinearElasticFunctionalBase(const myjson::json& j) {
        weight = myjson::double_value(j, "weight", weight);
        scale = weight / initial_value;
    }
    virtual ~StaticLinearElasticFunctionalBase() = default;

    virtual void initialize(const FEM& fem,
                            Analysis& analysis,
                            const Eigen::MatrixXd& xPhys);

    virtual void calculate(const FEM& fem,
                           Analysis& analysis,
                           const Eigen::MatrixXd& xPhys) = 0;

    virtual Eigen::VectorXd dfdu(const FEM& fem,
                                 Analysis& analysis,
                                 const Eigen::MatrixXd& xPhys,
                                 int istep) = 0;

    virtual Eigen::VectorXd dfdFext(const FEM& fem,
                                    Analysis& analysis,
                                    const Eigen::MatrixXd& xPhys,
                                    int istep) = 0;

    virtual Eigen::VectorXd dfdx(const FEM& fem,
                                 Analysis& analysis,
                                 const Eigen::MatrixXd& xPhys,
                                 int val_id) = 0;
};

struct Compliance : StaticLinearElasticFunctionalBase {
    explicit Compliance(const myjson::json& j) : StaticLinearElasticFunctionalBase(j) {
        function_name = "compliance";
    }

    void calculate(const FEM& fem,
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

    Eigen::VectorXd dfdx(const FEM&,
                         Analysis&,
                         const Eigen::MatrixXd&,
                         int) override {
        return Eigen::VectorXd();
    }
};

inline std::unique_ptr<StaticLinearElasticFunctionalBase>
InitializeStaticLinearElasticFunctional(const myjson::json& j) {
    const std::string function_name = myjson::lower_ascii(myjson::string_value(j, "function_name"));
    if (function_name == "compliance" || function_name == "static_compliance") {
        return std::make_unique<Compliance>(j);
    }
    throw std::runtime_error("Unknown StaticLinearElastic functional: " + function_name);
}
