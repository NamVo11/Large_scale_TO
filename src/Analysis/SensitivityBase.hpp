#pragma once

#include "Optimization/Filtering/Filtering.hpp"
#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"
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
    std::string sens_function;

    std::shared_ptr<Analysis> analysis;
    std::shared_ptr<ROMBase> ROM;
    std::unique_ptr<SensitivityMethodBase> sensitivity_method;

    double value = 0.0;

    GroupFunctionalBase() = default;

    GroupFunctionalBase(const myjson::json& j,
                        const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                        const myjson::json& run_json) {
        (void)run_json;
        analysis_id = j.at("analysis_id").get<int>();
        sens_function = j.at("sens_function").get<std::string>();

        const std::string sens_type = myjson::string_value(j, "sens_type", "AVM");
        sensitivity_method = InitializeSensitivityMethod(sens_type);

        for (const auto& candidate : analysis_all) {
            if (candidate && candidate->id == analysis_id) {
                analysis = candidate;
                ROM = candidate->ROM;
                return;
            }
        }
        throw std::runtime_error("Analysis id not found in JSON: " + std::to_string(analysis_id));
    }

    virtual ~GroupFunctionalBase() = default;

    virtual void initialize_terms(const FEM& fem,
                                  Analysis& analysis,
                                  const Eigen::MatrixXd& xPhys) = 0;

    virtual void calculate_terms(const FEM& fem,
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

    virtual Eigen::MatrixXd AVM(const FEM& fem,
                                Analysis& analysis,
                                const Eigen::MatrixXd& x,
                                const std::vector<int>& x_label,
                                const Filtering& filtering,
                                int Iter) = 0;

protected:
    void add_scaled(Eigen::VectorXd& dst,
                    const Eigen::VectorXd& src,
                    double scale) const;
};

std::unique_ptr<GroupFunctionalBase> InitializeGroupFunctional(const myjson::json& j,
                                                              const std::string& linear_backend,
                                                              const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                                                              const myjson::json& run_json);
