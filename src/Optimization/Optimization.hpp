#pragma once

#include "Analysis/Analysis.hpp"
#include "MatlabLikeData.hpp"
#include "Output/OutputData.hpp"
#include "Sensitivity/Sensitivity.hpp"

#include <memory>
#include <string>
#include <vector>

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
    bool save_density_each_iter = true;
    int save_density_interval = 1;
    OutputData Output;
    std::vector<ObjectiveGroup> objective_groups;
    std::vector<Constraint> constraints;
    OptimizationHistory history;
};

struct OptimizationBase : Optimization {
    std::vector<std::unique_ptr<DynamicLinearElastic>> analysis_all;
    std::unique_ptr<SensitivityBase> sensitivity;

    virtual ~OptimizationBase() = default;

    DynamicLinearElastic& primary_analysis();
    const DynamicLinearElastic& primary_analysis() const;

    virtual const char* backend_name() const = 0;
    virtual void validate_backend() const {}

    void run(const FEM& fem);
};

struct PetscOptimization : OptimizationBase {
    PetscOptimization() {
        sensitivity = std::make_unique<PetscSensitivity>();
    }

    const char* backend_name() const override { return "petsc"; }
};

struct EigenOptimization : OptimizationBase {
    EigenOptimization() {
        sensitivity = std::make_unique<EigenSensitivity>();
    }

    const char* backend_name() const override { return "eigen"; }
    void validate_backend() const override;
};

std::unique_ptr<OptimizationBase> make_optimization_solver(std::string linear_backend,
                                                           std::unique_ptr<DynamicLinearElastic> analysis,
                                                           Optimization opt);
