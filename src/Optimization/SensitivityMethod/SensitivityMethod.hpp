#pragma once

#include "Optimization/Filtering/Filtering.hpp"

#include <Eigen/Dense>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "myjson.hpp"

struct Analysis;
struct FEM;
struct OutputData;
struct GroupFunctionalBase;

struct SensitivityMethodBase {
    std::string type;

    virtual ~SensitivityMethodBase() = default;

    virtual Eigen::MatrixXd run(const FEM& fem,
                                Analysis& analysis,
                                const Eigen::MatrixXd& x,
                                const std::vector<int>& x_label,
                                GroupFunctionalBase& object,
                                const Filtering& filtering,
                                const OutputData& output,
                                int Iter) = 0;
};

struct AVMSensitivityMethod : SensitivityMethodBase {
    AVMSensitivityMethod() {
        type = "AVM";
    }

    Eigen::MatrixXd run(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        GroupFunctionalBase& object,
                        const Filtering& filtering,
                        const OutputData& output,
                        int Iter) override;
};

struct DebugSensitivityMethod : SensitivityMethodBase {
    double fd_epsilon = 1e-6;
    int max_checks = 20;

    DebugSensitivityMethod() {
        type = "Debug";
    }

    Eigen::MatrixXd run(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        GroupFunctionalBase& object,
                        const Filtering& filtering,
                        const OutputData& output,
                        int Iter) override;
};

struct DebugNonlinearSensitivityMethod final : DebugSensitivityMethod {
    DebugNonlinearSensitivityMethod() { type = "DebugNonlinear"; }
    // ===========================================

    Eigen::MatrixXd run(const FEM& fem, Analysis& analysis, const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label, GroupFunctionalBase& object,
                        const Filtering& filtering, const OutputData& output, int Iter) override;
};

inline std::unique_ptr<SensitivityMethodBase> InitializeSensitivityMethod(const std::string& sens_type,
                                                                          const std::string& analysis_solver_name) {
    const std::string key = myjson::lower_ascii(sens_type.empty() ? "avm" : sens_type);
    if (key == "avm") return std::make_unique<AVMSensitivityMethod>();
    if (key == "debug") {
        if (analysis_solver_name == "static_geomerical_nonlinear")
            return std::make_unique<DebugNonlinearSensitivityMethod>();
        return std::make_unique<DebugSensitivityMethod>();
    }
    throw std::runtime_error("Unknown sensitivity method: " + sens_type);
}
// ===========================================
