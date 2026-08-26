#pragma once

#include "Optimization/Filtering/Filtering.hpp"
#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"
#include "myjson.hpp"

#include <Eigen/Dense>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class ROMBase;
struct Analysis;
struct FEM;
struct OutputData;

struct GroupFunctionalBase {
    int analysis_id = 1;
    std::string sens_type = "AVM";
    std::string sens_function;
    std::string type;

    std::shared_ptr<ROMBase> ROM;
    std::unique_ptr<SensitivityMethodBase> sensitivity_method;

    double value = 0.0;
    Eigen::MatrixXd d_explicit_dfdu_all;
    Eigen::MatrixXd d_explicit_dfdv_all;
    Eigen::MatrixXd d_explicit_dfda_all;
    Eigen::MatrixXd d_explicit_dfdFext_all;
    Eigen::MatrixXd d_explicit_df_dx;

    virtual ~GroupFunctionalBase() = default;

    virtual void initialize_terms(const FEM& fem,
                                  Analysis& analysis,
                                  const Eigen::MatrixXd& xPhys) = 0;

    virtual void calculate_terms(const FEM& fem,
                                 Analysis& analysis,
                                 const Eigen::MatrixXd& xPhys) = 0;

    virtual Eigen::MatrixXd AVM(const FEM& fem,
                                Analysis& analysis,
                                const Eigen::MatrixXd& x,
                                const std::vector<int>& x_label,
                                const Filtering& filtering,
                                int Iter) = 0;

    virtual void debug_FDM(const FEM& fem,
                           Analysis& analysis,
                           const Eigen::MatrixXd& x,
                           const std::vector<int>& x_label,
                           const Filtering& filtering,
                           const OutputData& output,
                           double fd_epsilon,
                           int max_checks,
                           int Iter,
                           const Eigen::MatrixXd& avm);

protected:
    void reset_accumulated_derivatives();
    void add_term_contribution(double alpha,
                               double term_value,
                               const Eigen::MatrixXd& dfdu_all,
                               const Eigen::MatrixXd& dfdv_all,
                               const Eigen::MatrixXd& dfda_all,
                               const Eigen::MatrixXd& dfdFext_all,
                               const Eigen::MatrixXd& df_dx);
};

inline std::string group_function_key(const myjson::json& j) {
    const std::string sens_function = myjson::lower_ascii(myjson::string_value(j, "sens_function"));
    if (!sens_function.empty()) return sens_function;
    std::string function_name = myjson::lower_ascii(myjson::string_value(j, "function_name"));
    if (function_name.empty() && j.contains("terms") && j.at("terms").is_array() && !j.at("terms").empty()) {
        function_name = myjson::lower_ascii(myjson::string_value(j.at("terms").front(), "function_name"));
    }
    return function_name;
}

std::unique_ptr<GroupFunctionalBase> InitializeDynamicLinearElasticGroupFunctional(const myjson::json& j,
                                                                                  const std::string& backend);
std::unique_ptr<GroupFunctionalBase> InitializeStaticLinearElasticGroupFunctional(const myjson::json& j,
                                                                                 const std::string& backend);
std::unique_ptr<GroupFunctionalBase> InitializeVolumeGroupFunctional(const myjson::json& j);

inline std::unique_ptr<GroupFunctionalBase> InitializeGroupFunctional(const myjson::json& j,
                                                                     const std::string& linear_backend) {
    const std::string key = group_function_key(j);
    const std::string backend = myjson::lower_ascii(linear_backend.empty() ? "petsc" : linear_backend);

    if (key == "dynamic_elastic_base" ||
        key == "dynamic_compliance_avm" ||
        key == "dynamic_linear_elastic_avm") {
        return InitializeDynamicLinearElasticGroupFunctional(j, backend);
    }

    if (key == "static_elastic_base" ||
        key == "static_compliance_avm" ||
        key == "static_linear_elastic_avm") {
        return InitializeStaticLinearElasticGroupFunctional(j, backend);
    }

    if (key == "volume_fraction" || key == "volume_sens_base") {
        return InitializeVolumeGroupFunctional(j);
    }

    throw std::runtime_error("Unknown group functional: " + key);
}

inline double constraint_limit_from_json(const myjson::json& j) {
    return j.at("limit").get<double>();
}
