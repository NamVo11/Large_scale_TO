#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"

#include "Analysis/SensitivityBase.hpp"

Eigen::MatrixXd DebugSensitivityMethod::run(const FEM& fem,
                                            Analysis& analysis,
                                            const Eigen::MatrixXd& x,
                                            const std::vector<int>& x_label,
                                            GroupFunctionalBase& object,
                                            const Filtering& filtering,
                                            const OutputData& output,
                                            int Iter) {
    Eigen::MatrixXd avm = object.AVM(fem, analysis, x, x_label, filtering, Iter);
    object.debug_FDM(fem, analysis, x, x_label, filtering, output, fd_epsilon, max_checks, Iter, avm);
    return avm;
}
