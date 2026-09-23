#include "Optimization/Optimization.hpp"

#include "MatlabMMA.hpp"
#include "Optimization/Filtering/Filtering.hpp"
#include "helper/MemoryDebug.hpp"
#include "myjson.hpp"

#include <petscsys.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
}

std::vector<double> flatten(const Eigen::MatrixXd& x) {
    std::vector<double> out(static_cast<std::size_t>(x.size()));
    std::copy(x.data(), x.data() + x.size(), out.begin());
    return out;
}

Eigen::MatrixXd unflatten(const std::vector<double>& x, int nelem, int nvar) {
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
        x.data(), nelem, nvar);
}

} // namespace

Analysis& OptimizationBase::primary_analysis() {
    if (analysis_all.empty()) throw std::runtime_error("Optimization has no Analysis object");
    return *analysis_all.front();
}

const Analysis& OptimizationBase::primary_analysis() const {
    if (analysis_all.empty()) throw std::runtime_error("Optimization has no Analysis object");
    return *analysis_all.front();
}

Eigen::MatrixXd init_design_variables(const FEM& fem,
                                      const std::vector<Region>& regions,
                                      const std::vector<Material>& Mat,
                                      double x0,
                                      std::vector<int>& x_label) {
    const int nvar = std::max(1, static_cast<int>(Mat.size()) - 1);

    if (x0 < 0.0 || x0 > 1.0) {
        throw std::runtime_error("init_design_variables: x0 must be between 0 and 1");
    }

    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(fem.NElem, nvar);
    x_label.assign(static_cast<std::size_t>(fem.NElem), 0);

    for (const auto& region : regions) {
        for (int val_id : region.mat_paras.val_ids) {
            if (val_id <= 0 || val_id > nvar) {
                throw std::runtime_error("init_design_variables: region val_id is outside x columns");
            }
        }
        for (int e : region.elem_ids) {
            if (e < 0 || e >= fem.NElem) throw std::runtime_error("init_design_variables: invalid element id in region");
        }

        const std::string status = myjson::lower_ascii(region.status);
        if (status == "solid") {
            for (int e : region.elem_ids) {
                x_label[static_cast<std::size_t>(e)] = 1;
                for (int val_id : region.mat_paras.val_ids) x(e, val_id - 1) = 1.0;
            }
        } else if (status == "void") {
            for (int e : region.elem_ids) {
                x_label[static_cast<std::size_t>(e)] = -1;
                x.row(e).setZero();
            }
        } else if (status == "design_val") {
            for (int e : region.elem_ids) {
                for (int val_id : region.mat_paras.val_ids) x(e, val_id - 1) = x0;
            }
        } else {
            throw std::runtime_error("init_design_variables: unsupported region status: " + region.status);
        }
    }
    return x;
}

void EigenOptimization::validate_backend() const {
    PetscMPIInt size = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size != 1) throw std::runtime_error("EigenOptimization requires mpi_size=1");
}

void OptimizationBase::run(const FEM& fem) {
    validate_backend();

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    Output.prepare_case_folders();
    Output.initialize_vtk_cache(fem);
    FilteringState filtering_state;
    filtering.initialize(fem, filtering_state);
    const int nelem = fem.NElem;
    const int nvar = nvar_of(x);
    MatlabMMA mma(nelem * nvar, static_cast<int>(constraint.groups.size()), 1.0, 0.0, 1000.0, 0.0);

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

    while (Iter <= MaxIter && (Iter == 0 || Change > Tol)) {
        MemoryDebug::print("iter_begin");
        const auto iter_t0 = Clock::now();
        double output_time = 0.0;
        if (Iter > 0) ++loopbeta;
        MemoryDebug::print("before_filtering");
        filtering.calculate(x, x_label, filtering_state);
        MemoryDebug::print("after_filtering");
        if (Output.save_density_each_iter != 0 && Iter % Output.save_density_interval == 0) {
            const auto out_t0 = Clock::now();
            Output.save_density_iteration(fem, filtering_state.xPhys, Iter);
            output_time += elapsed(out_t0, Clock::now());
        }
        auto t0 = Clock::now();
        MemoryDebug::print("before_forward");
        for (auto& A : analysis_all) A->solver(fem, filtering_state.xPhys, Iter);
        MemoryDebug::print("after_forward");
        auto t1 = Clock::now();
        forward_time = elapsed(t0, t1);
        for (auto& A : analysis_all) {
            if (Iter % Output.save_vtk_response_interval == 0) {
                const auto out_t0 = Clock::now();
                Output.export_response_vtk(fem, *A, filtering_state.xPhys, Iter);
                output_time += elapsed(out_t0, Clock::now());
            }
        }

        t0 = Clock::now();
        MemoryDebug::print("before_objective_value");
        if (Iter == 0) objective.initialize(fem, analysis_all, filtering_state.xPhys);
        else objective.calculate(fem, analysis_all, filtering_state.xPhys);
        obj_value = objective.value;
        MemoryDebug::print("after_objective_value");
        t1 = Clock::now();
        objective_time = elapsed(t0, t1);

        t0 = Clock::now();
        MemoryDebug::print("before_constraint_value");
        constraint.calculate(fem, analysis_all, filtering_state.xPhys);
        constr_value_vec = constraint.values;
        MemoryDebug::print("after_constraint_value");
        t1 = Clock::now();
        constraint_time = elapsed(t0, t1);

        if (rank == 0) {
            std::cout << "-----------------------------------------------------\n";
            std::cout << "Iter=" << Iter << " Obj=" << obj_value;
            if (!constr_value_vec.empty()) std::cout << " Constr=" << constr_value_vec.front();
            std::cout << " Change=" << Change << "\n";
            std::cout << "  timing[s]: forward=" << forward_time
                      << " obj_sens=" << objective_sensitivity_time
                      << " constr_sens=" << constraint_sensitivity_time << "\n";
        }

        dObj_dx_mat = Eigen::MatrixXd::Zero(nelem, nvar);
        t0 = Clock::now();
        MemoryDebug::print("before_objective_sensitivity");
        objective.calculate_sensitivity(fem, analysis_all, x, x_label, filtering, filtering_state, Output,
                                        debug_fd_epsilon, debug_fd_max_checks, Iter);
        dObj_dx_mat = objective.sensitivity;
        MemoryDebug::print("after_objective_sensitivity");
        t1 = Clock::now();
        objective_sensitivity_time = elapsed(t0, t1);
        dObj_dx_vec = flatten(dObj_dx_mat);
        if (Iter % Output.save_vtk_opt_interval == 0) {
            const auto out_t0 = Clock::now();
            Output.export_opt_vtk(fem, filtering_state.xPhys, dObj_dx_mat, Iter);
            output_time += elapsed(out_t0, Clock::now());
        }

        dConstr_dx.clear();
        t0 = Clock::now();
        MemoryDebug::print("before_constraint_sensitivity");
        constraint.calculate_sensitivity(fem, analysis_all, x, x_label, filtering, filtering_state,
                                         Output, debug_fd_epsilon, debug_fd_max_checks, Iter);
        for (const auto& dc : constraint.sensitivity) {
            std::vector<double> flat = flatten(dc);
            dConstr_dx.insert(dConstr_dx.end(), flat.begin(), flat.end());
        }
        MemoryDebug::print("after_constraint_sensitivity");
        t1 = Clock::now();
        constraint_sensitivity_time = elapsed(t0, t1);

        history.Iter.push_back(Iter);
        history.objective.push_back(obj_value);
        history.constraint.push_back(constr_value_vec.empty() ? 0.0 : constr_value_vec.front());
        history.change.push_back(Iter == 0 ? 0.0 : Change);
        history.rms_change.push_back(0.0);
        history.forward_time.push_back(forward_time);
        history.objective_time.push_back(objective_time);
        history.constraint_time.push_back(constraint_time);
        history.objective_sensitivity_time.push_back(objective_sensitivity_time);
        history.constraint_sensitivity_time.push_back(constraint_sensitivity_time);
        history.mma_time.push_back(0.0);
        history.iteration_time.push_back(0.0);

        if (Iter >= MaxIter) {
            history.iteration_time.back() = elapsed(iter_t0, Clock::now()) - output_time;
            if (rank == 0) std::cout << " mma=0 iter_total=" << history.iteration_time.back() << "\n";
            Output.save_history_csv(*this);
            Output.save_rom_histories(*this);
            break;
        }

        t0 = Clock::now();
        std::vector<double> xvec = flatten(x);
        std::vector<double> xmin(xvec.size()), xmax(xvec.size());
        for (std::size_t i = 0; i < xvec.size(); ++i) {
            xmin[i] = std::max(0.0, xvec[i] - algorithm.move);
            xmax[i] = std::min(1.0, xvec[i] + algorithm.move);
        }
        std::vector<double> xold = xvec;
        mma.Update(xvec.data(), dObj_dx_vec.data(), constr_value_vec.data(), dConstr_dx.data(), xmin.data(), xmax.data());
        Change = 0.0;
        double rms = 0.0;
        for (std::size_t i = 0; i < xvec.size(); ++i) {
            xvec[i] = std::clamp(xvec[i], 0.0, 1.0);
            Change = std::max(Change, std::abs(xvec[i] - xold[i]));
            rms += (xvec[i] - xold[i]) * (xvec[i] - xold[i]);
        }
        rms = std::sqrt(rms / static_cast<double>(xvec.size()));
        history.rms_change.back() = rms;
        x = unflatten(xvec, nelem, nvar);
        t1 = Clock::now();
        mma_time = elapsed(t0, t1);
        history.mma_time.back() = mma_time;
        history.iteration_time.back() = elapsed(iter_t0, Clock::now()) - output_time;

        if (filtering.beta < filtering.beta_max && (loopbeta >= filtering.beta_loop || Change <= Tol)) {
            filtering.beta = std::min(filtering.beta + 2.0, filtering.beta_max);
            loopbeta = 0;
            Change = 1.0;
            if (rank == 0) std::cout << "Parameter beta increased to " << filtering.beta << ".\n";
        }
        if (rank == 0) {
            std::cout << " mma=" << mma_time
                      << " iter_total=" << history.iteration_time.back() << "\n";
        }
        Output.save_history_csv(*this);
        Output.save_rom_histories(*this);
        ++Iter;
    }
    Output.save_history_csv(*this);
    Output.save_rom_histories(*this);
}

std::unique_ptr<OptimizationBase> make_optimization_solver(std::string linear_backend,
                                                           std::vector<std::unique_ptr<Analysis>> analysis_all,
                                                           Optimization opt) {
    std::unique_ptr<OptimizationBase> solver;
    if (myjson::lower_ascii(linear_backend) == "eigen") solver = std::make_unique<EigenOptimization>();
    else solver = std::make_unique<PetscOptimization>();
    static_cast<Optimization&>(*solver) = std::move(opt);
    solver->analysis_all = std::move(analysis_all);
    return solver;
}
