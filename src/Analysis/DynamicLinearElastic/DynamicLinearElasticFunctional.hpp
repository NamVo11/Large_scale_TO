#pragma once

#include "myjson.hpp"

#include <Eigen/Dense>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct DynamicLinearElastic;
struct FEM;

struct DynamicLinearElasticFunctionalBase {
    std::string function_name;
    double weight = 1.0;
    double initial_value = 0.0;
    double value = 0.0;

    Eigen::MatrixXd dfdu_all;
    Eigen::MatrixXd dfdv_all;
    Eigen::MatrixXd dfda_all;
    Eigen::MatrixXd dfdFext_all;
    Eigen::MatrixXd df_dx;

    DynamicLinearElasticFunctionalBase() = default;
    explicit DynamicLinearElasticFunctionalBase(const myjson::json& j) {
        function_name = myjson::string_value(j, "function_name", function_name);
        weight = myjson::double_value(j, "weight", weight);
        initial_value = myjson::double_value(j, "value_initial", initial_value);
    }
    virtual ~DynamicLinearElasticFunctionalBase() = default;

    virtual void initialize(const FEM& fem,
                            DynamicLinearElastic& analysis,
                            const Eigen::MatrixXd& xPhys);

    virtual void calculate(const FEM& fem,
                           DynamicLinearElastic& analysis,
                           const Eigen::MatrixXd& xPhys) = 0;
};

struct L2ComplianceElastic : DynamicLinearElasticFunctionalBase {
    explicit L2ComplianceElastic(const myjson::json& j)
        : DynamicLinearElasticFunctionalBase(j) {}

    void calculate(const FEM& fem,
                   DynamicLinearElastic& analysis,
                   const Eigen::MatrixXd& xPhys) override;
};

inline std::unique_ptr<DynamicLinearElasticFunctionalBase>
InitializeDynamicLinearElasticFunctional(const myjson::json& j) {
    const std::string function_name = myjson::lower_ascii(myjson::string_value(j, "function_name"));
    if (function_name == "l2_compliance_elastic") {
        return std::make_unique<L2ComplianceElastic>(j);
    }
    throw std::runtime_error("Unknown DynamicLinearElastic functional: " + function_name);
}
