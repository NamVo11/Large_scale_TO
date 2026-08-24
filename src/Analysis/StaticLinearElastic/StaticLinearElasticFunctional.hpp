#pragma once

#include "myjson.hpp"

#include <Eigen/Dense>

#include <memory>
#include <stdexcept>
#include <string>

struct FEM;
struct StaticLinearElastic;

struct StaticLinearElasticFunctionalBase {
    std::string function_name;
    double weight = 1.0;
    double initial_value = 0.0;
    double value = 0.0;

    Eigen::MatrixXd dfdu_all;
    Eigen::MatrixXd dfdv_all;
    Eigen::MatrixXd dfda_all;
    Eigen::MatrixXd dfdFext_all;
    Eigen::MatrixXd df_dx;

    StaticLinearElasticFunctionalBase() = default;
    explicit StaticLinearElasticFunctionalBase(const myjson::json& j) {
        function_name = myjson::string_value(j, "function_name", function_name);
        weight = myjson::double_value(j, "weight", weight);
        initial_value = myjson::double_value(j, "value_initial", initial_value);
    }
    virtual ~StaticLinearElasticFunctionalBase() = default;

    virtual void initialize(const FEM& fem,
                            StaticLinearElastic& analysis,
                            const Eigen::MatrixXd& xPhys);

    virtual void calculate(const FEM& fem,
                           StaticLinearElastic& analysis,
                           const Eigen::MatrixXd& xPhys) = 0;
};

struct Compliance : StaticLinearElasticFunctionalBase {
    explicit Compliance(const myjson::json& j)
        : StaticLinearElasticFunctionalBase(j) {}

    void calculate(const FEM& fem,
                   StaticLinearElastic& analysis,
                   const Eigen::MatrixXd& xPhys) override;
};

inline std::unique_ptr<StaticLinearElasticFunctionalBase>
InitializeStaticLinearElasticFunctional(const myjson::json& j) {
    const std::string function_name = myjson::lower_ascii(myjson::string_value(j, "function_name"));
    if (function_name == "compliance" || function_name == "static_compliance") {
        return std::make_unique<Compliance>(j);
    }
    throw std::runtime_error("Unknown StaticLinearElastic functional: " + function_name);
}
