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
#include <limits>
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
    if (avm.rows() != x.rows() || avm.cols() != x.cols())
        throw std::runtime_error("Debug: AVM gradient dimensions mismatch");

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    const double current_value = object.value;
    Eigen::MatrixXd fd_matrix = Eigen::MatrixXd::Zero(x.rows(), x.cols());
    double squared_error = 0.0;
    std::size_t checked = 0;
    for (int e = 0; e < x.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) continue;
        for (int v = 0; v < x.cols(); ++v) {
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

            fd_matrix(e, v) = fd;
            squared_error += std::pow(avm(e, v) - fd, 2);
            ++checked;
        }
    }

    analysis.solver(fem, filtering.xPhys, Iter);
    object.calculate_terms(fem, analysis, filtering.xPhys);
    if (rank == 0) {
        const auto dir = std::filesystem::path(output.outDir_mat).parent_path() / "AVM_debug";
        std::filesystem::create_directories(dir);
        std::ofstream out(dir / "debug_sensitivity_fd.csv", std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot write debug_sensitivity_fd.csv");
        out.precision(std::numeric_limits<double>::max_digits10);
        for (int v = 0; v < x.cols(); ++v) out << (v ? "," : "") << "AVM_" << v + 1;
        for (int v = 0; v < x.cols(); ++v) out << ",FD_" << v + 1;
        out << '\n';
        for (int e = 0; e < x.rows(); ++e) {
            for (int v = 0; v < x.cols(); ++v) out << (v ? "," : "") << avm(e, v);
            for (int v = 0; v < x.cols(); ++v) out << "," << fd_matrix(e, v);
            out << '\n';
        }
        std::cout << "debug MSE: Iter=" << Iter << " mse="
                  << (checked ? squared_error / static_cast<double>(checked) : 0.0) << '\n';
    }
    return avm;
}
// ===========================================

Eigen::MatrixXd DebugNonlinearSensitivityMethod::run(const FEM& fem, Analysis& analysis,
                                                      const Eigen::MatrixXd& x,
                                                      const std::vector<int>& x_label,
                                                      GroupFunctionalBase& object,
                                                      const Filtering& filtering,
                                                      const OutputData& output, int Iter) {
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

    Eigen::MatrixXd fd_step = Eigen::MatrixXd::Zero(x.rows(), x.cols());
    bool trial_solved = false;
    try {
        for (int e = 0; e < x.rows(); ++e) {
            if (x_label.at(static_cast<std::size_t>(e)) != 0) continue;
            for (int v = 0; v < x.cols(); ++v) {
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

                fd_step(e, v) = fd;
            }
        }
    } catch (...) {
        if (trial_solved) A.solve_trial_step(fem, filtering.xPhys, step, nominal_u);
        throw;
    }

    if (trial_solved) A.solve_trial_step(fem, filtering.xPhys, step, nominal_u);
    if (Iter != cumulative_iter_ || step == 1) {
        avm_cumulative_ = Eigen::MatrixXd::Zero(x.rows(), x.cols());
        fd_cumulative_ = Eigen::MatrixXd::Zero(x.rows(), x.cols());
        cumulative_iter_ = Iter;
        cumulative_step_ = 0;
    }
    if (step != cumulative_step_ + 1 || avm_cumulative_.rows() != x.rows() || avm_cumulative_.cols() != x.cols())
        throw std::runtime_error("Nonlinear Debug: load steps must be processed in order");
    avm_cumulative_ += avm;
    fd_cumulative_ += fd_step;
    cumulative_step_ = step;

    if (rank == 0) {
        const auto dir = std::filesystem::path(output.outDir_mat).parent_path() / "AVM_debug";
        std::filesystem::create_directories(dir);
        std::ofstream out(dir / "debug_sensitivity_fd.csv", std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot write debug_sensitivity_fd.csv");
        out.precision(std::numeric_limits<double>::max_digits10);
        for (int v = 0; v < x.cols(); ++v) out << (v ? "," : "") << "AVM_" << v + 1;
        for (int v = 0; v < x.cols(); ++v) out << ",FD_" << v + 1;
        out << '\n';
        double squared_error = 0.0;
        std::size_t checked = 0;
        for (int e = 0; e < x.rows(); ++e) {
            for (int v = 0; v < x.cols(); ++v) out << (v ? "," : "") << avm_cumulative_(e, v);
            for (int v = 0; v < x.cols(); ++v) out << "," << fd_cumulative_(e, v);
            out << '\n';
            if (x_label.at(static_cast<std::size_t>(e)) != 0) continue;
            for (int v = 0; v < x.cols(); ++v) {
                squared_error += std::pow(avm_cumulative_(e, v) - fd_cumulative_(e, v), 2);
                ++checked;
            }
        }
        std::cout << "nonlinear debug MSE: Iter=" << Iter << " step=" << step << " mse="
                  << (checked ? squared_error / static_cast<double>(checked) : 0.0) << '\n';
    }
    return avm;
}
// ===========================================
