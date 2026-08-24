#pragma once

#include "Analysis/Analysis.hpp"
#include "Analysis/SensitivityBase.hpp"
#include "Optimization/Constraint/Constraint.hpp"
#include "Optimization/Filtering/Filtering.hpp"
#include "Optimization/Objective/Objective.hpp"
#include "Output/OutputData.hpp"

#include <memory>
#include <string>
#include <vector>

struct Algorithm {
    std::string type = "MMA";
    double move = 0.1;

    Algorithm() = default;

    explicit Algorithm(const myjson::json& j) {
        type = myjson::string_value(j, "type", type);
        move = myjson::double_value(j, "move", move);
    }
};

struct OptimizationHistory {
    std::vector<int> Iter;
    std::vector<double> objective;
    std::vector<double> constraint;
    std::vector<double> change;
    std::vector<double> rms_change;
    std::vector<double> forward_time;
    std::vector<double> objective_time;
    std::vector<double> constraint_time;
    std::vector<double> objective_sensitivity_time;
    std::vector<double> constraint_sensitivity_time;
    std::vector<double> mma_time;
    std::vector<double> iteration_time;
};

struct Optimization {
    Eigen::MatrixXd x0;
    Eigen::MatrixXd x;
    std::vector<int> x_label;
    int MaxIter = 300;
    double Tol = 0.0;
    bool plot_during_optimization = false;
    Filtering filtering;
    Algorithm algorithm;
    double debug_fd_epsilon = 1e-6;
    int debug_fd_max_checks = 20;
    OutputData Output;
    Objective objective;
    Constraint constraint;
    OptimizationHistory history;
};

struct OptimizationBase : Optimization {
    std::vector<std::unique_ptr<Analysis>> analysis_all;

    virtual ~OptimizationBase() = default;

    Analysis& primary_analysis();
    const Analysis& primary_analysis() const;

    virtual const char* backend_name() const = 0;
    virtual void validate_backend() const {}

    void run(const FEM& fem);
};

struct PetscOptimization : OptimizationBase {
    const char* backend_name() const override { return "petsc"; }
};

struct EigenOptimization : OptimizationBase {
    const char* backend_name() const override { return "eigen"; }
    void validate_backend() const override;
};

std::unique_ptr<OptimizationBase> make_optimization_solver(std::string linear_backend,
                                                           std::vector<std::unique_ptr<Analysis>> analysis_all,
                                                           Optimization opt);

Eigen::MatrixXd init_design_variables(const FEM& fem,
                                      const std::vector<Region>& regions,
                                      const std::vector<Material>& Mat,
                                      double x0,
                                      std::vector<int>& x_label);
