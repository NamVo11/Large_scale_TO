#include "Optimization/Optimization.hpp"

#include "MatlabMMA.hpp"
#include "Optimization/Filtering.hpp"
#include "ROM/ROM.hpp"
#include "helper/MemoryDebug.hpp"

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

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

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

void add_scaled(Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double scale) {
    if (b.size() == 0) return;
    if (a.size() == 0) {
        a = scale * b;
        return;
    }
    a.noalias() += scale * b;
}

void evaluate_objective_ptr(const FEM& fem,
                            OptimizationBase& Opt,
                            const Eigen::MatrixXd& xPhys,
                            double& obj_value) {
    (void)xPhys;
    obj_value = 0.0;
    for (auto& group : Opt.objective_groups) {
        Analysis& analysis = *Opt.analysis_all[static_cast<std::size_t>(group.analysis_id - 1)];
        group.value = 0.0;
        group.d_explicit_dfdu_all.resize(0, 0);
        group.d_explicit_dfdv_all.resize(0, 0);
        group.d_explicit_dfda_all.resize(0, 0);
        group.d_explicit_dfdFext_all.resize(0, 0);
        group.d_explicit_df_dx.resize(0, 0);
        for (auto& term : group.terms) {
            if (lower(term.function_name) != "l2_compliance_elastic") throw std::runtime_error("Unknown objective term");
            const int ndof = fem.ndof;
            const int nstep = analysis.Nstep;
            term.dfdu_all = Eigen::MatrixXd::Zero(ndof, nstep + 1);
            term.dfdv_all.resize(0, 0);
            term.dfda_all.resize(0, 0);
            term.dfdFext_all = Eigen::MatrixXd::Zero(ndof, nstep + 1);
            term.df_dx.resize(0, 0);
            term.value = 0.0;
            for (int k = 0; k <= nstep; ++k) {
                const auto u = analysis.results.U.col(k);
                const auto f = analysis.results.F_ext.col(k);
                const double s = f.dot(u);
                term.value += s * s;
                term.dfdu_all.col(k).noalias() = 2.0 * s * f;
                term.dfdFext_all.col(k).noalias() = 2.0 * s * u;
                for (int d : analysis.results.SPCdofs) {
                    term.dfdu_all(d, k) = 0.0;
                }
                for (int d : analysis.results.free) term.dfdFext_all(d, k) = 0.0;
            }
            if (term.value_initial == 0.0) term.value_initial = term.value;
            const double alpha = term.weight / term.value_initial;
            group.value += alpha * term.value;
            add_scaled(group.d_explicit_dfdu_all, term.dfdu_all, alpha);
            add_scaled(group.d_explicit_dfdv_all, term.dfdv_all, alpha);
            add_scaled(group.d_explicit_dfda_all, term.dfda_all, alpha);
            add_scaled(group.d_explicit_dfdFext_all, term.dfdFext_all, alpha);
            add_scaled(group.d_explicit_df_dx, term.df_dx, alpha);
        }
        obj_value += group.value;
    }
}

void evaluate_constraint_ptr(const FEM& fem,
                             OptimizationBase& Opt,
                             const Eigen::MatrixXd& xPhys,
                             std::vector<double>& constr_value_vec) {
    constr_value_vec.assign(Opt.constraints.size(), 0.0);
    for (std::size_t i = 0; i < Opt.constraints.size(); ++i) {
        Constraint& c = Opt.constraints[i];
        if (lower(c.function_name) != "volume_fraction") throw std::runtime_error("Unknown constraint");
        const int nvar = nvar_of(xPhys);
        const int val_id = c.mat_id - 2;
        if (val_id < 0 || val_id >= nvar) throw std::runtime_error("volume_fraction: invalid mat_id");
        if (static_cast<int>(fem.volume.size()) != fem.NElem) {
            throw std::runtime_error("volume_fraction: fem volume is not initialized");
        }
        const Eigen::Map<const Eigen::VectorXd> volume_vec(fem.volume.data(), fem.NElem);
        const double total_volume = volume_vec.sum();
        const double material_volume = volume_vec.dot(xPhys.col(val_id));
        if (!(total_volume > 0.0)) throw std::runtime_error("volume_fraction: non-positive total volume");
        c.value = material_volume / total_volume;
        c.d_explicit_dfdu_all.resize(0, 0);
        c.d_explicit_dfdv_all.resize(0, 0);
        c.d_explicit_dfda_all.resize(0, 0);
        c.d_explicit_dfdFext_all.resize(0, 0);
        c.d_explicit_df_dx = Eigen::MatrixXd::Zero(fem.NElem, nvar);
        c.d_explicit_df_dx.col(val_id) = volume_vec / total_volume;
        if (c.value_initial == 0.0) c.value_initial = c.value;
        constr_value_vec[i] = c.value / c.value_initial - c.limit;
        c.d_explicit_df_dx *= 1.0 / c.value_initial;
    }
}

} // namespace

DynamicLinearElastic& OptimizationBase::primary_analysis() {
    if (analysis_all.empty()) throw std::runtime_error("Optimization has no Analysis object");
    return *analysis_all.front();
}

const DynamicLinearElastic& OptimizationBase::primary_analysis() const {
    if (analysis_all.empty()) throw std::runtime_error("Optimization has no Analysis object");
    return *analysis_all.front();
}

void EigenOptimization::validate_backend() const {
    PetscMPIInt size = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size != 1) throw std::runtime_error("EigenOptimization requires mpi_size=1");
}

void OptimizationBase::run(const FEM& fem) {
    validate_backend();
    if (!sensitivity) throw std::runtime_error("Optimization sensitivity object is not initialized");

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    Output.prepare_case_folders();
    Output.initialize_vtk_cache(fem);
    FilteringState filtering_state;
    filtering_initilization(fem, filtering, filtering_state);
    const int nelem = fem.NElem;
    const int nvar = nvar_of(x);
    MatlabMMA mma(nelem * nvar, static_cast<int>(constraints.size()), 1.0, 0.0, 1000.0, 0.0);

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
        filtering_calculation(x, x_label, filtering, filtering_state);
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
        evaluate_objective_ptr(fem, *this, filtering_state.xPhys, obj_value);
        MemoryDebug::print("after_objective_value");
        t1 = Clock::now();
        objective_time = elapsed(t0, t1);

        t0 = Clock::now();
        MemoryDebug::print("before_constraint_value");
        evaluate_constraint_ptr(fem, *this, filtering_state.xPhys, constr_value_vec);
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
        for (auto& group : objective_groups) {
            Analysis& A = *analysis_all[static_cast<std::size_t>(group.analysis_id - 1)];
            Eigen::MatrixXd dg = lower(group.sens_type) == "debug"
                ? sensitivity->debug_objective_sensitivity(fem, A, x, x_label, group, filtering, filtering_state,
                                                           Output, debug_fd_epsilon, debug_fd_max_checks, Iter)
                : sensitivity->objective_sensitivity(fem, A, x, x_label, group, filtering, filtering_state, Iter);
            add_scaled(dObj_dx_mat, dg, 1.0);
        }
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
        for (auto& c : constraints) {
            Analysis& A = *analysis_all[static_cast<std::size_t>(c.analysis_id - 1)];
            Eigen::MatrixXd dc = sensitivity->constraint_sensitivity(fem, A, x, x_label, c, filtering, filtering_state, Iter);
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
                                                           std::unique_ptr<DynamicLinearElastic> analysis,
                                                           Optimization opt) {
    std::unique_ptr<OptimizationBase> solver;
    if (lower(std::move(linear_backend)) == "eigen") solver = std::make_unique<EigenOptimization>();
    else solver = std::make_unique<PetscOptimization>();
    static_cast<Optimization&>(*solver) = std::move(opt);
    solver->analysis_all.push_back(std::move(analysis));
    return solver;
}
