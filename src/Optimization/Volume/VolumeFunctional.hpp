#pragma once

#include "myjson.hpp"

#include <Eigen/Dense>

#include <memory>
#include <stdexcept>
#include <string>

struct Analysis;
struct FEM;

struct VolumeFunctionalBase {
    std::string function_name;
    double weight = 1.0;
    double initial_value = 1.0;
    double value = 0.0;

    Eigen::MatrixXd dfdu_all;
    Eigen::MatrixXd dfdv_all;
    Eigen::MatrixXd dfda_all;
    Eigen::MatrixXd dfdFext_all;
    Eigen::MatrixXd df_dx;

    VolumeFunctionalBase() = default;
    explicit VolumeFunctionalBase(const myjson::json& j) {
        function_name = myjson::string_value(j, "function_name", function_name);
        weight = myjson::double_value(j, "weight", weight);
        initial_value = myjson::double_value(j, "value_initial", initial_value);
        if (initial_value == 0.0) initial_value = 1.0;
    }
    virtual ~VolumeFunctionalBase() = default;

    virtual void initialize(const FEM& fem,
                            Analysis& analysis,
                            const Eigen::MatrixXd& xPhys);

    virtual void calculate(const FEM& fem,
                           Analysis& analysis,
                           const Eigen::MatrixXd& xPhys) = 0;
};

struct VolumeFractionTerm : VolumeFunctionalBase {
    int mat_id = 0;

    explicit VolumeFractionTerm(const myjson::json& j)
        : VolumeFunctionalBase(j) {
        mat_id = myjson::int_value(j, "mat_id", mat_id);
    }

    void calculate(const FEM& fem,
                   Analysis& analysis,
                   const Eigen::MatrixXd& xPhys) override;
};

inline std::unique_ptr<VolumeFunctionalBase> InitializeVolumeFunctional(const myjson::json& j) {
    const std::string function_name = myjson::lower_ascii(myjson::string_value(j, "function_name"));
    if (function_name == "volume_fraction") {
        return std::make_unique<VolumeFractionTerm>(j);
    }
    throw std::runtime_error("Unknown volume functional: " + function_name);
}
