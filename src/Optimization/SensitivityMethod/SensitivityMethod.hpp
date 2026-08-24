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
                                const FilteringState& filtering_state,
                                const OutputData& output,
                                double fd_epsilon,
                                int max_checks,
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
                        const FilteringState& filtering_state,
                        const OutputData& output,
                        double fd_epsilon,
                        int max_checks,
                        int Iter) override;
};

struct DebugSensitivityMethod : SensitivityMethodBase {
    DebugSensitivityMethod() {
        type = "Debug";
    }

    Eigen::MatrixXd run(const FEM& fem,
                        Analysis& analysis,
                        const Eigen::MatrixXd& x,
                        const std::vector<int>& x_label,
                        GroupFunctionalBase& object,
                        const Filtering& filtering,
                        const FilteringState& filtering_state,
                        const OutputData& output,
                        double fd_epsilon,
                        int max_checks,
                        int Iter) override;
};

inline std::unique_ptr<SensitivityMethodBase> InitializeSensitivityMethod(const std::string& sens_type) {
    const std::string key = myjson::lower_ascii(sens_type.empty() ? "avm" : sens_type);
    if (key == "avm") return std::make_unique<AVMSensitivityMethod>();
    if (key == "debug") return std::make_unique<DebugSensitivityMethod>();
    throw std::runtime_error("Unknown sensitivity method: " + sens_type);
}
