#pragma once

#include "stage2/MatlabLikeData.hpp"

#include <vector>

namespace stage2 {

struct FilteringState {
    std::vector<std::vector<double>> xTilde;
    std::vector<std::vector<double>> xPhys;
    std::vector<std::vector<int>> neighbors;
    std::vector<std::vector<double>> weights;
    std::vector<double> Hs;
};

void filtering_initilization(const FEM& fem, Filtering& filtering, FilteringState& state);
void filtering_calculation(const std::vector<std::vector<double>>& x,
                           const std::vector<int>& x_label,
                           const Filtering& filtering,
                           FilteringState& state);
std::vector<std::vector<double>> filtering_sensitivity(const std::vector<std::vector<double>>& df_dx_phys,
                                                       const Filtering& filtering,
                                                       const FilteringState& state);

void evaluate_objective(const FEM& fem,
                        std::vector<Analysis>& analysis_all,
                        const std::vector<std::vector<double>>& xPhys,
                        std::vector<ObjectiveGroup>& objective_groups,
                        double& obj_value);
void evaluate_constraint(const FEM& fem,
                         std::vector<Analysis>& analysis_all,
                         const std::vector<std::vector<double>>& xPhys,
                         std::vector<Constraint>& constraints,
                         std::vector<double>& constr_value_vec);

std::vector<std::vector<double>> AVM(const FEM& fem,
                                     Analysis& analysis,
                                     const std::vector<std::vector<double>>& x,
                                     const std::vector<int>& x_label,
                                     ObjectiveGroup& object,
                                     const Filtering& filtering,
                                     const FilteringState& filtering_state,
                                     int Iter);
std::vector<std::vector<double>> AVM(const FEM& fem,
                                     Analysis& analysis,
                                     const std::vector<std::vector<double>>& x,
                                     const std::vector<int>& x_label,
                                     Constraint& object,
                                     const Filtering& filtering,
                                     const FilteringState& filtering_state,
                                     int Iter);

std::vector<std::vector<double>> Debug(const FEM& fem,
                                       Analysis& analysis,
                                       const std::vector<std::vector<double>>& x,
                                       const std::vector<int>& x_label,
                                       ObjectiveGroup& object,
                                       const Filtering& filtering,
                                       const FilteringState& filtering_state,
                                       const OutputData& output,
                                       double fd_epsilon,
                                       int max_checks,
                                       int Iter);

void optimization11(const FEM& fem,
                    std::vector<Analysis>& analysis_all,
                    Optimization& Opt,
                    const OutputData& Output);

} // namespace stage2
