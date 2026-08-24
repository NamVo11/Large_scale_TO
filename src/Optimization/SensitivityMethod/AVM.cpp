#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"

#include "Analysis/SensitivityBase.hpp"

Eigen::MatrixXd AVMSensitivityMethod::run(const FEM& fem,
                                          Analysis& analysis,
                                          const Eigen::MatrixXd& x,
                                          const std::vector<int>& x_label,
                                          GroupFunctionalBase& object,
                                          const Filtering& filtering,
                                          const FilteringState& filtering_state,
                                          const OutputData&,
                                          double,
                                          int,
                                          int Iter) {
    return object.AVM(fem, analysis, x, x_label, filtering, filtering_state, Iter);
}
