#include "Optimization/Optimization.hpp"

#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearAVM.hpp"
#include "helper/MemoryDebug.hpp"

#include <petscsys.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void Optimization::evaluate_nonlinear(const FEM& fem, int Iter) {
    const Eigen::MatrixXd& xPhys = filtering->xPhys;
    const bool save_response = Output.save_vtk_response_interval > 0 &&
                               Iter % Output.save_vtk_response_interval == 0;
    if (Iter == 0) {
        // Establish a reference for each term without retaining the full load history.
        for (auto& group_base : objective.groups) {
            auto& group = dynamic_cast<StaticGeomericalNonlinearAVM&>(*group_base);
            for (auto& term : group.terms) {
                term->value = 0.0;
                term->scale = 1.0;
            }
        }
        for (auto& analysis_base : analysis_all) {
            auto& A = dynamic_cast<StaticGeomericalNonlinear&>(*analysis_base);
            A.initialize_solver(fem, xPhys, Iter);
            if (save_response) Output.export_response_vtk_step(fem, A, xPhys, Iter, 0, 0);
            for (int step = 1; step <= A.Nstep; ++step) {
                A.solve_step(fem, step);
                if (save_response) Output.export_response_vtk_step(fem, A, xPhys, Iter, step, 0);
                objective.calculate_step(fem, A, xPhys);
            }
        }
        for (auto& group_base : objective.groups) {
            auto& group = dynamic_cast<StaticGeomericalNonlinearAVM&>(*group_base);
            for (auto& term : group.terms) {
                term->initial_value = term->value;
                if (std::abs(term->initial_value) <= 1e-30)
                    throw std::runtime_error("Nonlinear initial objective term is zero: " + term->function_name);
                term->scale = term->weight / term->initial_value;
            }
        }
        return; // Temporary forward-only test: stop after the Iter=0 load trajectory.
    }

    // Scaled objective and sensitivity: a second trajectory only at Iter=0.
    for (auto& group_base : objective.groups) {
        auto& group = dynamic_cast<StaticGeomericalNonlinearAVM&>(*group_base);
        for (auto& term : group.terms) term->value = 0.0;
    }
    objective.sensitivity = Eigen::MatrixXd::Zero(fem.NElem, xPhys.cols());
    for (auto& analysis_base : analysis_all) {
        auto& A = dynamic_cast<StaticGeomericalNonlinear&>(*analysis_base);
        A.initialize_solver(fem, xPhys, Iter);
        if (save_response) Output.export_response_vtk_step(fem, A, xPhys, Iter, 0, 0);
        for (int step = 1; step <= A.Nstep; ++step) {
            A.solve_step(fem, step);
            if (save_response) Output.export_response_vtk_step(fem, A, xPhys, Iter, step, 0);
            objective.calculate_step(fem, A, xPhys);
            objective.calculate_sensitivity_step(fem, A, x, x_label, *filtering, Output, Iter);
        }
    }

    // All load steps are complete: assemble each group, then the total objective.
    objective.value = 0.0;
    for (auto& group_base : objective.groups) {
        auto& group = dynamic_cast<StaticGeomericalNonlinearAVM&>(*group_base);
        group.value = 0.0;
        for (const auto& term : group.terms) group.value += term->scale * term->value;
        objective.value += group.value;
    }
}
// ===========================================

void Optimization::run_nonlinear(const FEM& fem) {
    if (!filtering) throw std::runtime_error("Nonlinear optimization requires filtering");
    if (analysis_all.empty() || objective.groups.empty())
        throw std::runtime_error("Nonlinear optimization requires analyses and objective groups");
    for (const auto& analysis : analysis_all)
        if (!dynamic_cast<StaticGeomericalNonlinear*>(analysis.get()))
            throw std::runtime_error("Nonlinear optimization cannot mix analysis formulations");
    bool debug_selected = false;
    for (const auto& group : objective.groups) {
        if (!dynamic_cast<EigenStaticGeomericalNonlinearAVM*>(group.get()) ||
            !group->sensitivity_method ||
            (group->sensitivity_method->type != "AVM" && group->sensitivity_method->type != "DebugNonlinear"))
            throw std::runtime_error("Nonlinear optimization requires Eigen AVM or nonlinear Debug objective groups");
        debug_selected |= group->sensitivity_method->type == "DebugNonlinear";
    }
    if (debug_selected && objective.groups.size() != 1)
        throw std::runtime_error("Nonlinear Debug requires exactly one objective group");

    // For now, volume-only constraints are supported outside the load-step loop.
    for (const auto& group : constraint.groups)
        if (!dynamic_cast<VolumeFractionConstraint*>(group.get()))
            throw std::runtime_error("Nonlinear optimization supports volume constraints only");

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    Output.prepare_case_folders();
    Output.initialize_vtk_cache(fem);
    filtering->initialize(fem);

    const int nelem = fem.NElem;
    const int nvar = static_cast<int>(x.cols());
    algorithm = InitializeAlgorithm(algorithm_json, nelem * nvar,
                                    static_cast<int>(constraint.groups.size()));

    using Clock = std::chrono::steady_clock;
    auto elapsed = [](Clock::time_point t0) {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    };

    int Iter = 0;
    int loopbeta = 0;
    double Change = 1.0;

    while (Iter <= MaxIter && (Iter == 0 || Change > Tol)) {
        const auto iteration_start = Clock::now();
        double output_time = 0.0;
        if (Iter > 0) ++loopbeta;

        filtering->calculate(x, x_label);
        if (Output.save_density_each_iter != 0 &&
            Output.save_density_interval > 0 &&
            Iter % Output.save_density_interval == 0) {
            const auto start = Clock::now();
            Output.save_density_iteration(fem, filtering->xPhys, Iter);
            output_time += elapsed(start);
        }

        const auto eval_start = Clock::now();
        MemoryDebug::print("before_nonlinear_evaluation");
        evaluate_nonlinear(fem, Iter);
        MemoryDebug::print("after_nonlinear_evaluation");
        if (Iter == 0) return; // Temporary forward-only test; main performs PetscFinalize().
        const double evaluation_time = elapsed(eval_start);

        const double obj_value = objective.value;
        const Eigen::MatrixXd dObj_dx_mat = objective.sensitivity;
        const std::vector<double> dObj_dx_vec = flatten(dObj_dx_mat);

        // Constraints (volume) are independent of load steps.
        const auto constraint_start = Clock::now();
        constraint.calculate(fem, filtering->xPhys);
        const std::vector<double> constr_value_vec = constraint.values;
        const double constraint_time = elapsed(constraint_start);

        const auto constraint_sens_start = Clock::now();
        constraint.calculate_sensitivity(fem, x, x_label, *filtering, Output, Iter);
        std::vector<double> dConstr_dx;
        for (const auto& grad : constraint.sensitivity) {
            const auto flat = flatten(grad);
            dConstr_dx.insert(dConstr_dx.end(), flat.begin(), flat.end());
        }
        const double constraint_sens_time = elapsed(constraint_sens_start);

        if (Output.save_vtk_opt_interval > 0 &&
            Iter % Output.save_vtk_opt_interval == 0) {
            const auto start = Clock::now();
            Output.export_opt_vtk(fem, filtering->xPhys, dObj_dx_mat, Iter);
            output_time += elapsed(start);
        }

        history.Iter.push_back(Iter);
        history.objective.push_back(obj_value);
        history.constraint.push_back(constr_value_vec.empty() ? 0.0 : constr_value_vec.front());
        history.change.push_back(Iter == 0 ? 0.0 : Change);
        history.rms_change.push_back(0.0);
        history.forward_time.push_back(evaluation_time); // temporary combined evaluation time
        history.objective_time.push_back(0.0);
        history.constraint_time.push_back(constraint_time);
        history.objective_sensitivity_time.push_back(0.0);
        history.constraint_sensitivity_time.push_back(constraint_sens_time);
        history.mma_time.push_back(0.0);
        history.iteration_time.push_back(0.0);

        if (rank == 0) {
            std::cout << "Iter=" << Iter << " Obj=" << obj_value;
            if (!constr_value_vec.empty()) std::cout << " Constr=" << constr_value_vec.front();
            std::cout << " Change=" << Change << " nonlinear_eval=" << evaluation_time << " s\n";
        }

        if (Iter >= MaxIter) {
            history.iteration_time.back() = elapsed(iteration_start) - output_time;
            Output.save_history_csv(*this);
            Output.save_rom_histories(*this);
            break;
        }

        const auto mma_start = Clock::now();
        std::vector<double> xvec = flatten(x);
        std::vector<double> xmin(xvec.size()), xmax(xvec.size());
        for (std::size_t i = 0; i < xvec.size(); ++i) {
            xmin[i] = std::max(0.0, xvec[i] - algorithm->move);
            xmax[i] = std::min(1.0, xvec[i] + algorithm->move);
        }
        const std::vector<double> xold = xvec;
        algorithm->Update(xvec.data(), dObj_dx_vec.data(),
                          constr_value_vec.data(), dConstr_dx.data(),
                          xmin.data(), xmax.data());

        Change = 0.0;
        double rms = 0.0;
        for (std::size_t i = 0; i < xvec.size(); ++i) {
            xvec[i] = std::clamp(xvec[i], 0.0, 1.0);
            Change = std::max(Change, std::abs(xvec[i] - xold[i]));
            rms += (xvec[i] - xold[i]) * (xvec[i] - xold[i]);
        }
        history.rms_change.back() = std::sqrt(rms / static_cast<double>(xvec.size()));
        x = unflatten(xvec, nelem, nvar);
        history.mma_time.back() = elapsed(mma_start);
        history.iteration_time.back() = elapsed(iteration_start) - output_time;

        if (filtering->beta < filtering->beta_max &&
            (loopbeta >= filtering->beta_loop || Change <= Tol)) {
            filtering->beta = std::min(filtering->beta + 2.0, filtering->beta_max);
            loopbeta = 0;
            Change = 1.0;
            if (rank == 0) std::cout << "Parameter beta increased to " << filtering->beta << "\n";
        }

        Output.save_history_csv(*this);
        Output.save_rom_histories(*this);
        ++Iter;
    }
    Output.save_history_csv(*this);
    Output.save_rom_histories(*this);
}
// ===========================================
