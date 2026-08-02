#pragma once

#include "Optimization/Filtering.hpp"

#include <Eigen/Dense>

#include <vector>

struct Analysis;
struct BoundaryCondition;
struct Constraint;
struct FEM;
struct ObjectiveGroup;
struct OutputData;

struct SensitivityBase {
    virtual ~SensitivityBase() = default;

    virtual Eigen::MatrixXd objective_sensitivity(const FEM& fem,
                                                  Analysis& analysis,
                                                  const Eigen::MatrixXd& x,
                                                  const std::vector<int>& x_label,
                                                  ObjectiveGroup& object,
                                                  const Filtering& filtering,
                                                  const FilteringState& filtering_state,
                                                  int Iter) = 0;

    virtual Eigen::MatrixXd constraint_sensitivity(const FEM& fem,
                                                   Analysis& analysis,
                                                   const Eigen::MatrixXd& x,
                                                   const std::vector<int>& x_label,
                                                   Constraint& object,
                                                   const Filtering& filtering,
                                                   const FilteringState& filtering_state,
                                                   int Iter);

    virtual Eigen::MatrixXd debug_objective_sensitivity(const FEM& fem,
                                                        Analysis& analysis,
                                                        const Eigen::MatrixXd& x,
                                                        const std::vector<int>& x_label,
                                                        ObjectiveGroup& object,
                                                        const Filtering& filtering,
                                                        const FilteringState& filtering_state,
                                                        const OutputData& output,
                                                        double fd_epsilon,
                                                        int max_checks,
                                                        int Iter);
};

struct PetscSensitivity : SensitivityBase {
    Eigen::MatrixXd objective_sensitivity(const FEM& fem,
                                          Analysis& analysis,
                                          const Eigen::MatrixXd& x,
                                          const std::vector<int>& x_label,
                                          ObjectiveGroup& object,
                                          const Filtering& filtering,
                                          const FilteringState& filtering_state,
                                          int Iter) override;
};

struct EigenSensitivity : SensitivityBase {
    Eigen::MatrixXd objective_sensitivity(const FEM& fem,
                                          Analysis& analysis,
                                          const Eigen::MatrixXd& x,
                                          const std::vector<int>& x_label,
                                          ObjectiveGroup& object,
                                          const Filtering& filtering,
                                          const FilteringState& filtering_state,
                                          int Iter) override;
};
