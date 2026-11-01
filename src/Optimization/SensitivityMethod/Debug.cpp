#include "Optimization/SensitivityMethod/SensitivityMethod.hpp"

#include "Analysis/Analysis.hpp"
#include "Analysis/SensitivityBase.hpp"
#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearAVM.hpp"
#include "Output/OutputData.hpp"

#include <petscsys.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

Eigen::MatrixXd DebugSensitivityMethod::run(const FEM& fem,
                                            Analysis& analysis,
                                            const Eigen::MatrixXd& x,
                                            const std::vector<int>& x_label,
                                            GroupFunctionalBase& object,
                                            const Filtering& filtering,
                                            const OutputData& output,
                                            int Iter) {
    Eigen::MatrixXd avm = object.AVM(fem, analysis, x, x_label, filtering, Iter);

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    std::ofstream out;
    if (rank == 0) {
        std::filesystem::create_directories(output.outDir_mat);
        const std::string path = output.outDir_mat + "/debug_sensitivity_fd.csv";
        static std::set<std::string> initialized_paths;
        const bool reset_file = Iter == 0 && initialized_paths.insert(path).second;
        out.open(path, reset_file ? std::ios::out : std::ios::app);
        if (!out) throw std::runtime_error("Cannot write debug_sensitivity_fd.csv");
        if (reset_file) {
            out << "Iter,group,elem,var,x,AVM,FD,abs_err,rel_err,value_plus,value_minus\n";
        }
    }

    const double current_value = object.value;
    int checked = 0;
    for (int e = 0; e < x.rows() && checked < max_checks; ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) continue;
        for (int v = 0; v < x.cols() && checked < max_checks; ++v) {
            const double x0 = x(e, v);
            const double h_plus = std::min(fd_epsilon, 1.0 - x0);
            const double h_minus = std::min(fd_epsilon, x0);
            if (h_plus <= 0.0 && h_minus <= 0.0) continue;

            double value_plus = current_value;
            double value_minus = current_value;
            if (h_plus > 0.0) {
                Eigen::MatrixXd xp = x;
                Filtering state_plus = filtering;
                xp(e, v) = x0 + h_plus;
                state_plus.calculate(xp, x_label);
                analysis.solver(fem, state_plus.xPhys, Iter);
                object.calculate_terms(fem, analysis, state_plus.xPhys);
                value_plus = object.value;
            }
            if (h_minus > 0.0) {
                Eigen::MatrixXd xm = x;
                Filtering state_minus = filtering;
                xm(e, v) = x0 - h_minus;
                state_minus.calculate(xm, x_label);
                analysis.solver(fem, state_minus.xPhys, Iter);
                object.calculate_terms(fem, analysis, state_minus.xPhys);
                value_minus = object.value;
            }

            double fd = 0.0;
            if (h_plus > 0.0 && h_minus > 0.0) fd = (value_plus - value_minus) / (h_plus + h_minus);
            else if (h_plus > 0.0) fd = (value_plus - current_value) / h_plus;
            else fd = (current_value - value_minus) / h_minus;

            if (rank == 0) {
                const double g = avm(e, v);
                const double abs_err = std::abs(g - fd);
                const double rel_err = abs_err / std::max(1e-14, std::abs(fd));
                out << Iter << "," << object.sens_function << "," << e << "," << v << "," << x0 << ","
                    << g << "," << fd << "," << abs_err << "," << rel_err << ","
                    << value_plus << "," << value_minus << "\n";
            }
            ++checked;
        }
    }

    analysis.solver(fem, filtering.xPhys, Iter);
    object.calculate_terms(fem, analysis, filtering.xPhys);
    return avm;
}
// ===========================================

Eigen::MatrixXd DebugNonlinearSensitivityMethod::run(const FEM& fem, Analysis& analysis,
                                                      const Eigen::MatrixXd& x,
                                                      const std::vector<int>& x_label,
                                                      GroupFunctionalBase& object,
                                                      const Filtering& filtering,
                                                      const OutputData& output, int Iter) {
    (void)output;
    auto& A = dynamic_cast<StaticGeomericalNonlinear&>(analysis);
    auto& group = dynamic_cast<StaticGeomericalNonlinearAVM&>(object);
    Eigen::MatrixXd avm = object.AVM(fem, analysis, x, x_label, filtering, Iter);
    if (avm.rows() != x.rows() || avm.cols() != x.cols())
        throw std::runtime_error("Nonlinear Debug: AVM gradient dimensions mismatch");

    const int step = A.current_step;
    const Eigen::VectorXd nominal_u = A.U.col(0);
    const double nominal_value = group.step_value(fem, A, filtering.xPhys);
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    bool trial_solved = false;
    try {
        int checked = 0;
        for (int e = 0; e < x.rows() && checked < max_checks; ++e) {
            if (x_label.at(static_cast<std::size_t>(e)) != 0) continue;
            for (int v = 0; v < x.cols() && checked < max_checks; ++v) {
                const double x0 = x(e, v);
                const double h_plus = std::min(fd_epsilon, 1.0 - x0);
                const double h_minus = std::min(fd_epsilon, x0);
                if (h_plus <= 0.0 && h_minus <= 0.0) continue;

                double value_plus = nominal_value;
                double value_minus = nominal_value;
                if (h_plus > 0.0) {
                    Eigen::MatrixXd xp = x;
                    Filtering state_plus = filtering;
                    xp(e, v) = x0 + h_plus;
                    state_plus.calculate(xp, x_label);
                    trial_solved = true;
                    A.solve_trial_step(fem, state_plus.xPhys, step, nominal_u);
                    value_plus = group.step_value(fem, A, state_plus.xPhys);
                }
                if (h_minus > 0.0) {
                    Eigen::MatrixXd xm = x;
                    Filtering state_minus = filtering;
                    xm(e, v) = x0 - h_minus;
                    state_minus.calculate(xm, x_label);
                    trial_solved = true;
                    A.solve_trial_step(fem, state_minus.xPhys, step, nominal_u);
                    value_minus = group.step_value(fem, A, state_minus.xPhys);
                }

                double fd = 0.0;
                if (h_plus > 0.0 && h_minus > 0.0) fd = (value_plus - value_minus) / (h_plus + h_minus);
                else if (h_plus > 0.0) fd = (value_plus - nominal_value) / h_plus;
                else fd = (nominal_value - value_minus) / h_minus;

                if (rank == 0) {
                    const double abs_err = std::abs(avm(e, v) - fd);
                    const double rel_err = abs_err / std::max(1e-14, std::abs(fd));
                    std::cout << "nonlinear debug: Iter=" << Iter << " step=" << step
                              << " elem=" << e << " var=" << v << " AVM=" << avm(e, v)
                              << " FD=" << fd << " abs_err=" << abs_err << " rel_err=" << rel_err << '\n';
                }
                ++checked;
            }
        }
    } catch (...) {
        if (trial_solved) A.solve_trial_step(fem, filtering.xPhys, step, nominal_u);
        throw;
    }

    if (trial_solved) A.solve_trial_step(fem, filtering.xPhys, step, nominal_u);
    return avm;
}
// ===========================================
