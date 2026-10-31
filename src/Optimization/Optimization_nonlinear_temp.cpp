
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

namespace {

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
}

std::vector<double> flatten(const Eigen::MatrixXd& x) {
    return std::vector<double>(x.data(), x.data() + x.size());
}

Eigen::MatrixXd unflatten(const std::vector<double>& x, int nelem, int nvar) {
    return Eigen::Map<
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
    >(x.data(), nelem, nvar);
}

} // namespace
// ===========================================

void Optimization::evaluate_nonlinear(const FEM& fem, int Iter) {

    if (!filtering)
        throw std::runtime_error("Nonlinear evaluation requires filtering");

    if (analysis_all.size() != 1)
        throw std::runtime_error("Nonlinear evaluation requires one analysis");

    auto& A = dynamic_cast<StaticGeomericalNonlinear&>(*analysis_all.front());

    if (objective.groups.size() != 1)
        throw std::runtime_error("Nonlinear evaluation requires one objective group");

    auto* group = dynamic_cast<EigenStaticGeomericalNonlinearAVM*>(
        objective.groups.front().get()
    );

    if (!group || group->analysis.get() != &A)
        throw std::runtime_error("Invalid nonlinear objective group");

    if (group->terms.empty())
        throw std::runtime_error("Nonlinear objective has no functional terms");

    const Eigen::MatrixXd& xPhys = filtering->xPhys;
    const int nelem = fem.NElem;
    const int nvar = static_cast<int>(xPhys.cols());

    // ========================================================
    // 1. INITIALIZE NONLINEAR ANALYSIS
    // ========================================================

    A.initialize_solver(fem, xPhys, Iter);

    // Reset accumulated objective and sensitivity for this
    // optimization iteration.
    //
    // Each term represents the contribution from ALL load steps.
    // initial_value is only initialized at optimization Iter = 0.
    //
    for (auto& term : group->terms) {

        term->value = 0.0;
        term->raw_gradient = Eigen::MatrixXd::Zero(nelem, nvar);

        if (Iter == 0) {
            term->initial_value = 0.0;
        }
    }

    group->value = 0.0;
    objective.value = 0.0;

    // ========================================================
    // 2. LOAD STEP LOOP
    // ========================================================

    for (int step = 1; step <= A.Nstep; ++step) {

        A.current_step = step;

        // ----------------------------------------------------
        // 2.1 Forward nonlinear analysis
        // ----------------------------------------------------

        // Solver handles ONE load step.
        //
        // After convergence:
        //   A.U.col(1)   = current displacement
        //   A.F_ext.col(1) = current external force
        //   A.K_T        = converged tangent stiffness
        //   A.factorized_K_T = factorized converged tangent
        //
        // Gauss-point stress and strain are also available.
        //
        A.solver(fem, xPhys, Iter);

        // ----------------------------------------------------
        // 2.2 Current load-step objective
        // ----------------------------------------------------

        if (Iter == 0) {

            // Evaluate current-step objective and accumulate:
            //
            // term->initial_value += J_step
            // term->value         += J_step
            //
            // No normalization is performed here.
            //
            objective.initialize_temp(fem, xPhys);

        } else {

            // Evaluate current-step objective and accumulate:
            //
            // term->value += J_step
            //
            objective.calculate_temp(fem, xPhys);
        }

        // ----------------------------------------------------
        // 2.3 Current load-step adjoint and sensitivity
        // ----------------------------------------------------

        // Solve adjoint using converged A.K_T.
        //
        // Accumulate unnormalized sensitivity directly inside
        // each nonlinear functional:
        //
        // term->raw_gradient += dJ_step / dxPhys
        //
        // No density-filter chain rule is applied here.
        //
        group->accumulate_sensitivity_temp(fem, A, xPhys);

        // ----------------------------------------------------
        // 2.4 Optional response output
        // ----------------------------------------------------

        // The nonlinear VTK writer must export current_step
        // from U.col(1), without looping through all Nstep.
        //
        if (Iter % Output.save_vtk_response_interval == 0) {
            Output.export_response_vtk(fem, A, xPhys, Iter);
        }
    }

    // ========================================================
    // 3. NORMALIZE OBJECTIVE AND SENSITIVITY
    // ========================================================

    // All load steps have been completed.
    //
    // Each term contains:
    //
    // initial_value = sum(J_step at optimization Iter = 0)
    // value         = sum(J_step at current optimization Iter)
    // raw_gradient  = sum(dJ_step/dxPhys)
    //
    // Normalize only ONCE using the total initial value.
    //

    Eigen::MatrixXd dObj_dxPhys = Eigen::MatrixXd::Zero(nelem, nvar);

    for (auto& term : group->terms) {

        if (Iter == 0) {

            if (std::abs(term->initial_value) <= 1e-30)
                throw std::runtime_error(
                    "Nonlinear objective initial_value is zero"
                );

            term->scale = term->weight / term->initial_value;
        }

        group->value += term->scale * term->value;

        dObj_dxPhys.noalias() +=
            term->scale * term->raw_gradient;
    }

    objective.value = group->value;

    // ========================================================
    // 4. FILTER OBJECTIVE SENSITIVITY
    // ========================================================

    // Apply filtering chain rule only once after accumulating
    // sensitivities from all load steps.
    //
    objective.sensitivity = filtering->sensitivity(dObj_dxPhys);

    // Fixed solid/void elements are not optimization variables.
    for (int e = 0; e < nelem; ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0)
            objective.sensitivity.row(e).setZero();
    }
}
// ===========================================

void Optimization::run_temp(const FEM& fem) {

    if (analysis_all.size() != 1)
        throw std::runtime_error("run_temp requires one analysis");

    if (!dynamic_cast<StaticGeomericalNonlinear*>(analysis_all.front().get()))
        throw std::runtime_error("run_temp requires StaticGeomericalNonlinear");

    if (!filtering)
        throw std::runtime_error("Optimization filtering is not initialized");

    // Currently, only design-independent volume constraints are
    // evaluated outside the nonlinear load-step loop.
    for (const auto& group : constraint.groups) {
        if (!dynamic_cast<VolumeFractionConstraint*>(group.get()))
            throw std::runtime_error(
                "run_temp currently supports volume constraints only"
            );
    }

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    Output.prepare_case_folders();
    Output.initialize_vtk_cache(fem);

    filtering->initialize(fem);

    const int nelem = fem.NElem;
    const int nvar = nvar_of(x);

    algorithm = InitializeAlgorithm(
        algorithm_json,
        nelem * nvar,
        static_cast<int>(constraint.groups.size())
    );

    using Clock = std::chrono::steady_clock;

    auto elapsed = [](Clock::time_point t0, Clock::time_point t1) {
        return std::chrono::duration<double>(t1 - t0).count();
    };

    int Iter = 0;
    int loopbeta = 0;
    double Change = 1.0;

    std::vector<double> constr_value_vec;
    Eigen::MatrixXd dObj_dx_mat;
    std::vector<double> dObj_dx_vec;
    std::vector<double> dConstr_dx;

    double obj_value = 0.0;
    double forward_time = 0.0;
    double objective_time = 0.0;
    double constraint_time = 0.0;
    double objective_sensitivity_time = 0.0;
    double constraint_sensitivity_time = 0.0;
    double mma_time = 0.0;

    // ========================================================
    // 1. OPTIMIZATION LOOP
    // ========================================================

    while (Iter <= MaxIter && (Iter == 0 || Change > Tol)) {

        MemoryDebug::print("iter_begin");

        const auto iter_t0 = Clock::now();
        double output_time = 0.0;

        if (Iter > 0) ++loopbeta;

        // ====================================================
        // 2. DENSITY FILTERING
        // ====================================================

        MemoryDebug::print("before_filtering");

        filtering->calculate(x, x_label);

        MemoryDebug::print("after_filtering");

        if (Output.save_density_each_iter != 0 &&
            Iter % Output.save_density_interval == 0) {

            const auto out_t0 = Clock::now();

            Output.save_density_iteration(
                fem, filtering->xPhys, Iter
            );

            output_time += elapsed(out_t0, Clock::now());
        }

        // ====================================================
        // 3. NONLINEAR FORWARD + OBJECTIVE + AVM
        // ====================================================

        // Unlike Optimization::run(), these calculations
        // are performed together, one load step at a time.
        //
        const auto t_eval = Clock::now();

        MemoryDebug::print("before_nonlinear_evaluation");

        evaluate_nonlinear(fem, Iter);

        MemoryDebug::print("after_nonlinear_evaluation");

        const double evaluation_time = elapsed(t_eval, Clock::now());

        // Temporary timing convention:
        // forward_time records the combined nonlinear evaluation.
        // Individual times can be separated later.
        forward_time = evaluation_time;
        objective_time = 0.0;
        objective_sensitivity_time = 0.0;

        obj_value = objective.value;
        dObj_dx_mat = objective.sensitivity;
        dObj_dx_vec = flatten(dObj_dx_mat);

        // ====================================================
        // 4. CONSTRAINT VALUE
        // ====================================================

        // Volume constraint does not depend on load step.
        // Evaluate it once per optimization iteration.
        //
        auto t0 = Clock::now();

        MemoryDebug::print("before_constraint_value");

        constraint.calculate(fem, filtering->xPhys);
        constr_value_vec = constraint.values;

        MemoryDebug::print("after_constraint_value");

        auto t1 = Clock::now();
        constraint_time = elapsed(t0, t1);

        // ====================================================
        // 5. CONSTRAINT SENSITIVITY
        // ====================================================

        t0 = Clock::now();

        MemoryDebug::print("before_constraint_sensitivity");

        constraint.calculate_sensitivity(
            fem,
            x,
            x_label,
            *filtering,
            Output,
            Iter
        );

        dConstr_dx.clear();

        for (const auto& dc : constraint.sensitivity) {

            const std::vector<double> flat = flatten(dc);

            dConstr_dx.insert(
                dConstr_dx.end(),
                flat.begin(),
                flat.end()
            );
        }

        MemoryDebug::print("after_constraint_sensitivity");

        t1 = Clock::now();
        constraint_sensitivity_time = elapsed(t0, t1);

        // ====================================================
        // 6. OUTPUT
        // ====================================================

        if (Iter % Output.save_vtk_opt_interval == 0) {

            const auto out_t0 = Clock::now();

            Output.export_opt_vtk(
                fem,
                filtering->xPhys,
                dObj_dx_mat,
                Iter
            );

            output_time += elapsed(out_t0, Clock::now());
        }

        if (rank == 0) {

            std::cout << "-----------------------------------------------------\n";
            std::cout << "Iter=" << Iter
                      << " Obj=" << obj_value;

            if (!constr_value_vec.empty())
                std::cout << " Constr=" << constr_value_vec.front();

            std::cout << " Change=" << Change
                      << " nonlinear_evaluation[s]=" << evaluation_time
                      << "\n";
        }

        // ====================================================
        // 7. OPTIMIZATION HISTORY
        // ====================================================

        history.Iter.push_back(Iter);
        history.objective.push_back(obj_value);
        history.constraint.push_back(
            constr_value_vec.empty() ? 0.0 : constr_value_vec.front()
        );
        history.change.push_back(Iter == 0 ? 0.0 : Change);
        history.rms_change.push_back(0.0);

        history.forward_time.push_back(forward_time);
        history.objective_time.push_back(objective_time);
        history.constraint_time.push_back(constraint_time);
        history.objective_sensitivity_time.push_back(objective_sensitivity_time);
        history.constraint_sensitivity_time.push_back(constraint_sensitivity_time);
        history.mma_time.push_back(0.0);
        history.iteration_time.push_back(0.0);

        // ====================================================
        // 8. CHECK MAXIMUM ITERATION
        // ====================================================

        if (Iter >= MaxIter) {

            history.iteration_time.back() =
                elapsed(iter_t0, Clock::now()) - output_time;

            if (rank == 0)
                std::cout << " mma=0 iter_total="
                          << history.iteration_time.back() << "\n";

            Output.save_history_csv(*this);
            Output.save_rom_histories(*this);

            break;
        }

        // ====================================================
        // 9. MMA UPDATE
        // ====================================================

        t0 = Clock::now();

        std::vector<double> xvec = flatten(x);
        std::vector<double> xmin(xvec.size());
        std::vector<double> xmax(xvec.size());

        for (std::size_t i = 0; i < xvec.size(); ++i) {

            xmin[i] = std::max(0.0, xvec[i] - algorithm->move);
            xmax[i] = std::min(1.0, xvec[i] + algorithm->move);
        }

        const std::vector<double> xold = xvec;

        algorithm->Update(
            xvec.data(),
            dObj_dx_vec.data(),
            constr_value_vec.data(),
            dConstr_dx.data(),
            xmin.data(),
            xmax.data()
        );

        Change = 0.0;
        double rms = 0.0;

        for (std::size_t i = 0; i < xvec.size(); ++i) {

            xvec[i] = std::clamp(xvec[i], 0.0, 1.0);

            Change = std::max(
                Change,
                std::abs(xvec[i] - xold[i])
            );

            rms += (xvec[i] - xold[i]) * (xvec[i] - xold[i]);
        }

        rms = std::sqrt(rms / static_cast<double>(xvec.size()));

        history.rms_change.back() = rms;

        x = unflatten(xvec, nelem, nvar);

        t1 = Clock::now();
        mma_time = elapsed(t0, t1);

        history.mma_time.back() = mma_time;
        history.iteration_time.back() =
            elapsed(iter_t0, Clock::now()) - output_time;

        // ====================================================
        // 10. HEAVISIDE CONTINUATION
        // ====================================================

        if (filtering->beta < filtering->beta_max &&
            (loopbeta >= filtering->beta_loop || Change <= Tol)) {

            filtering->beta = std::min(
                filtering->beta + 2.0,
                filtering->beta_max
            );

            loopbeta = 0;
            Change = 1.0;

            if (rank == 0)
                std::cout << "Parameter beta increased to "
                          << filtering->beta << ".\n";
        }

        if (rank == 0) {
            std::cout << " mma=" << mma_time
                      << " iter_total="
                      << history.iteration_time.back() << "\n";
        }

        Output.save_history_csv(*this);
        Output.save_rom_histories(*this);

        ++Iter;
    }

    Output.save_history_csv(*this);
    Output.save_rom_histories(*this);
}
// ===========================================
