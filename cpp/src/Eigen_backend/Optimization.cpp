#include "Eigen_backend/Optimization.hpp"

#include "Eigen_backend/DynamicElasticLinear.hpp"
#include "Eigen_backend/ROM/ROM_method/ROM_Base.hpp"
#include "FEMAssembly.hpp"
#include "OptimizationCommon.hpp"
#include "MatlabMMA.hpp"

#include <petscsys.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace eigen_backend {
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

Eigen::MatrixXd zeros(int nrow, int ncol) {
    return Eigen::MatrixXd::Zero(nrow, ncol);
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

double ramp_derivative(const std::vector<double>& para, double x, double p) {
    const double denom = 1.0 + p * (1.0 - x);
    const double dg = (1.0 + p) / (denom * denom);
    return (para[1] - para[0]) * dg;
}

void L2_compliance_elastic_for_debug(const FEM& fem,
                                     const Analysis& analysis,
                                     const Eigen::MatrixXd& xPhys,
                                     ObjectiveTerm& term) {
    const int ndof = fem.ndof;
    const int nstep = analysis.Nstep;
    const int nvar = nvar_of(xPhys);
    term.dfdu_all = zeros(ndof, nstep + 1);
    term.dfdv_all = zeros(ndof, nstep + 1);
    term.dfda_all = zeros(ndof, nstep + 1);
    term.dfdFext_all = zeros(ndof, nstep + 1);
    term.df_dx = zeros(fem.NElem, nvar);
    term.value = 0.0;

    for (int k = 0; k <= nstep; ++k) {
        const auto u = analysis.results.U.col(k);
        const auto f = analysis.results.F_ext.col(k);
        const double val = f.dot(u);
        term.value += val * val;
        term.dfdu_all.col(k).noalias() = 2.0 * val * f;
        term.dfdFext_all.col(k).noalias() = 2.0 * val * u;
        for (int d : analysis.results.SPCdofs) {
            term.dfdu_all(d, k) = 0.0;
            term.dfdv_all(d, k) = 0.0;
            term.dfda_all(d, k) = 0.0;
        }
        for (int d : analysis.results.free) term.dfdFext_all(d, k) = 0.0;
    }
}

void set_from_column(Eigen::VectorXd& v, const Eigen::MatrixXd& values, int k, double scale = 1.0) {
    v.setZero();
    if (values.size() == 0) return;
    if (k < 0 || k >= values.cols() || values.rows() != v.size()) {
        throw std::runtime_error("eigen_backend set_from_column: size mismatch");
    }
    v.noalias() = scale * values.col(k);
}

void set_fixed_from_column(Eigen::VectorXd& v,
                           const std::vector<int>& fixed,
                           const Eigen::MatrixXd& values,
                           int k,
                           double scale = 1.0) {
    if (values.size() == 0) return;
    if (k < 0 || k >= values.cols() || values.rows() != v.size()) {
        throw std::runtime_error("eigen_backend set_fixed_from_column: size mismatch");
    }
    for (int id : fixed) v(id) = scale * values(id, k);
}

void zero_fixed(Eigen::VectorXd& v, const std::vector<int>& fixed) {
    for (int id : fixed) v(id) = 0.0;
}

ROMContext make_adjoint_context(const Analysis& analysis, int Iter, int time_step) {
    ROMContext ctx;
    ctx.analysis_id = analysis.id;
    ctx.opt_iter = Iter;
    ctx.solve_step = time_step + 1;
    ctx.time_step = time_step;
    ctx.time = time_step * analysis.dt;
    ctx.role = "adjoint";
    ctx.system_type = "newmark_adjoint_effective";
    ctx.is_end_of_analysis = time_step == 0;
    return ctx;
}

Eigen::MatrixXd explicit_gradient_dynamic(const FEM& fem,
                                          const Analysis& analysis,
                                          const Eigen::MatrixXd& xPhys,
                                          const ObjectiveGroup& object,
                                          const Eigen::MatrixXd& lambda_all) {
    using Clock = std::chrono::steady_clock;
    const auto timer_begin = Clock::now();
    auto elapsed = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const int nstep = analysis.Nstep;
    const int nvar = nvar_of(xPhys);
    const Region& region = analysis.regions.front();
    const double nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();

    int sens_element_threads = 1;
    if (const char* env = std::getenv("SENS_ELEMENT_THREADS")) sens_element_threads = std::max(1, std::atoi(env));
    if (size != 1) sens_element_threads = 1;

    auto t0 = Clock::now();
    std::vector<double> dE(static_cast<std::size_t>(fem.NElem * nvar), 0.0);
    std::vector<double> drho(static_cast<std::size_t>(fem.NElem * nvar), 0.0);
    for (int e = 0; e < fem.NElem; ++e) {
        for (int v = 0; v < nvar; ++v) {
            const double x = xPhys(e, v);
            const std::size_t id = static_cast<std::size_t>(e * nvar + v);
            dE[id] = ramp_derivative(region.mat_paras.E_inf, x, region.interp.p_E_elastic);
            drho[id] = ramp_derivative(region.mat_paras.rho, x, region.interp.p_rho);
        }
    }
    auto t1 = Clock::now();
    const double timing_interp_derivative = elapsed(t0, t1);

    t0 = Clock::now();
    ElementMatrix uniform_em;
    if (fem.uniform_element_matrices) {
        uniform_em = compute_element_matrices(fem, analysis.regions.front().elem_ids.front(), nu, fem.Thickness);
    }
    t1 = Clock::now();
    const double timing_element_matrix = elapsed(t0, t1);

    t0 = Clock::now();
    Eigen::MatrixXd df_dx = Eigen::MatrixXd::Zero(fem.NElem, nvar);
    t1 = Clock::now();
    const double timing_alloc_gradient = elapsed(t0, t1);

    t0 = Clock::now();

    if (rank == 0) {
        int actual_threads = 1;
        #pragma omp parallel num_threads(sens_element_threads) if(sens_element_threads > 1)
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }
        /*
        std::cout << "  thread_debug explicit_gradient_dynamic: requested_threads="
                  << sens_element_threads
                  << " actual_threads=" << actual_threads
                  << " omp_get_max_threads=" << omp_get_max_threads()
                  << "\n";
        */
    }

    #pragma omp parallel for schedule(static) num_threads(sens_element_threads) if(sens_element_threads > 1)
    for (int e = 0; e < fem.NElem; ++e) {
        ElementMatrix local_em;
        const ElementMatrix* em_ptr = &uniform_em;
        if (!fem.uniform_element_matrices) {
            local_em = compute_element_matrices(fem, e, nu, fem.Thickness);
            em_ptr = &local_em;
        }
        const ElementMatrix& em = *em_ptr;
        if (em.ndof > 24) throw std::runtime_error("explicit_gradient_dynamic: element ndof > 24 is not supported by stack buffer");
        const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(e));
        Eigen::Matrix<double, 24, 1> u_elem_buffer;
        Eigen::Matrix<double, 24, 1> acc_elem_buffer;
        Eigen::Matrix<double, 24, 1> lambda_elem_buffer;
        Eigen::Matrix<double, 24, 1> Ku_elem_buffer;
        Eigen::Matrix<double, 24, 1> Ma_elem_buffer;

        for (int k = 0; k <= nstep; ++k) {
            const auto u = analysis.results.U.col(k);
            const auto acc = analysis.results.A.col(k);
            const auto lambda = lambda_all.col(k);
            for (int i = 0; i < em.ndof; ++i) {
                const int dof_i = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                u_elem_buffer(i) = u(dof_i);
                acc_elem_buffer(i) = acc(dof_i);
                lambda_elem_buffer(i) = lambda(dof_i);
            }
            auto u_elem = u_elem_buffer.head(em.ndof);
            auto acc_elem = acc_elem_buffer.head(em.ndof);
            auto lambda_elem = lambda_elem_buffer.head(em.ndof);
            auto Ku_elem = Ku_elem_buffer.head(em.ndof);
            auto Ma_elem = Ma_elem_buffer.head(em.ndof);
            Ku_elem.noalias() = em.ke * u_elem;
            if (k != 0) Ma_elem.noalias() = em.me * acc_elem;
            else Ma_elem.setZero();
            const double lambda_dot_Ku = lambda_elem.dot(Ku_elem);
            const double lambda_dot_Ma = lambda_elem.dot(Ma_elem);
            for (int v = 0; v < nvar; ++v) {
                const std::size_t ev = static_cast<std::size_t>(e * nvar + v);
                df_dx(e, v) += dE[ev] * lambda_dot_Ku + drho[ev] * lambda_dot_Ma;
            }
        }
    }
    t1 = Clock::now();
    const double timing_element_loop = elapsed(t0, t1);

    t0 = Clock::now();
    const double timing_flatten = 0.0;
    t1 = Clock::now();

    t0 = Clock::now();
    const double timing_allreduce = 0.0;
    t1 = Clock::now();

    t0 = Clock::now();
    add_scaled(df_dx, object.d_explicit_df_dx, 1.0);
    t1 = Clock::now();
    const double timing_unflatten_add = elapsed(t0, t1);

    if (rank == 0) {
        const double timing_total = elapsed(timer_begin, Clock::now());
        /*
        std::cout << "  explicit_gradient_dynamic[s]:"
                  << " threads=" << sens_element_threads
                  << " interp_derivative=" << timing_interp_derivative
                  << " element_matrix=" << timing_element_matrix
                  << " alloc_gradient=" << timing_alloc_gradient
                  << " element_loop=" << timing_element_loop
                  << " flatten=" << timing_flatten
                  << " allreduce=" << timing_allreduce
                  << " unflatten_add=" << timing_unflatten_add
                  << " total=" << timing_total
                  << "\n";
        */
    }

    return df_dx;
}

Eigen::MatrixXd dynamic_elastic_base_full_impl(const FEM& fem,
                                               Analysis& analysis,
                                               ObjectiveGroup& object,
                                               const Filtering& filtering,
                                               const FilteringState& filtering_state,
                                               int Iter) {
    const int nstep = analysis.Nstep;
    const double beta = 0.25;
    const double gamma = 0.5;
    const double dt = analysis.dt;
    const double a0c = 1.0 / (beta * dt * dt);
    const double dHdu = a0c;
    const double dLdu = gamma / (beta * dt);
    const double dHda_new = -(0.5 / beta - 1.0);
    const double dHdv_new = -1.0 / (beta * dt);
    const double dHdu_new = -a0c;
    const double dLda_new = (1.0 - 0.5 * gamma / beta) * dt;
    const double dLdv_new = 1.0 - gamma / beta;
    const double dLdu_new = -gamma / (beta * dt);

    const auto& K0 = analysis.results.K_inf_eigen;
    const auto& Keff = analysis.results.K_eff_eigen;
    const auto& M = analysis.results.M_eigen;
    if (K0.rows() == 0 || Keff.rows() == 0 || M.rows() == 0) {
        throw std::runtime_error("eigen_backend sensitivity requires Eigen forward matrices");
    }

    if (!object.backend_ROM) object.backend_ROM = make_eigen_rom_from_objective(object);
    std::shared_ptr<ROMBase>& rom = object.backend_ROM;

    const int ndof = fem.ndof;
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd lambda_new = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd mu_new = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd eta = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd eta_new = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd Mlambda = Eigen::VectorXd::Zero(ndof);
    Eigen::MatrixXd lambda_all(ndof, nstep + 1);

    int k = nstep;
    set_from_column(mu, object.d_explicit_dfda_all, k);
    set_from_column(eta, object.d_explicit_dfdv_all, k);
    set_from_column(rhs, object.d_explicit_dfdu_all, k, -1.0);
    rhs.noalias() += -dHdu * mu - dLdu * eta;
    set_fixed_from_column(rhs, analysis.results.SPCdofs, object.d_explicit_dfdFext_all, k, -1.0);
    if (analysis.results.factorized_K_eff_eigen) {
        rom->set_factorized_solver(Keff, analysis.results.factorized_K_eff_eigen);
    }
    rom->main(Keff, rhs, lambda, make_adjoint_context(analysis, Iter, k));
    lambda_all.col(k) = lambda;

    for (k = nstep - 1; k >= 0; --k) {
        lambda_new = lambda;
        mu_new = mu;
        eta_new = eta;
        Mlambda.noalias() = M * lambda_new;

        set_from_column(mu, object.d_explicit_dfda_all, k);
        mu.noalias() += -(0.5 / beta - 1.0) * Mlambda + dHda_new * mu_new + dLda_new * eta_new;
        set_from_column(eta, object.d_explicit_dfdv_all, k);
        eta.noalias() += -(1.0 / (beta * dt)) * Mlambda + dHdv_new * mu_new + dLdv_new * eta_new;
        zero_fixed(mu, analysis.results.SPCdofs);
        zero_fixed(eta, analysis.results.SPCdofs);

        set_from_column(rhs, object.d_explicit_dfdu_all, k, -1.0);
        rhs.noalias() += a0c * Mlambda - dHdu_new * mu_new - dLdu_new * eta_new;
        if (k != 0) rhs.noalias() += -dHdu * mu - dLdu * eta;
        set_fixed_from_column(rhs, analysis.results.SPCdofs, object.d_explicit_dfdFext_all, k, -1.0);
        if (k == 0) {
            if (analysis.results.factorized_K_initial_eigen) {
                rom->set_factorized_solver(K0, analysis.results.factorized_K_initial_eigen);
            }
            rom->main(K0, rhs, lambda, make_adjoint_context(analysis, Iter, k));
        } else {
            if (analysis.results.factorized_K_eff_eigen) {
                rom->set_factorized_solver(Keff, analysis.results.factorized_K_eff_eigen);
            }
            rom->main(Keff, rhs, lambda, make_adjoint_context(analysis, Iter, k));
        }
        lambda_all.col(k) = lambda;
    }

    Eigen::MatrixXd dfdx_phys = explicit_gradient_dynamic(fem, analysis, filtering_state.xPhys, object, lambda_all);
    return filtering_sensitivity(dfdx_phys, filtering, filtering_state);
}

Eigen::MatrixXd AVM(const FEM& fem,
                    Analysis& analysis,
                    const Eigen::MatrixXd&,
                    const std::vector<int>& x_label,
                    ObjectiveGroup& object,
                    const Filtering& filtering,
                    const FilteringState& filtering_state,
                    int Iter) {
    Eigen::MatrixXd out = dynamic_elastic_base_full_impl(fem, analysis, object, filtering, filtering_state, Iter);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}

Eigen::MatrixXd AVM(const FEM&,
                    Analysis&,
                    const Eigen::MatrixXd&,
                    const std::vector<int>& x_label,
                    Constraint& object,
                    const Filtering& filtering,
                    const FilteringState& filtering_state,
                    int) {
    Eigen::MatrixXd out = filtering_sensitivity(object.d_explicit_df_dx, filtering, filtering_state);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}

double objective_group_value_for_debug(const FEM& fem,
                                       const Analysis& analysis,
                                       const Eigen::MatrixXd& xPhys,
                                       const ObjectiveGroup& object) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) return 0.0;

    double value = 0.0;
    for (const auto& source_term : object.terms) {
        if (lower(source_term.function_name) != "l2_compliance_elastic") throw std::runtime_error("Unknown objective term");
        ObjectiveTerm term = source_term;
        L2_compliance_elastic_for_debug(fem, analysis, xPhys, term);
        const double value_initial = source_term.value_initial != 0.0 ? source_term.value_initial : term.value;
        value += term.weight * term.value / value_initial;
    }
    return value;
}

Eigen::MatrixXd Debug(const FEM& fem,
                      Analysis& analysis,
                      const Eigen::MatrixXd& x,
                      const std::vector<int>& x_label,
                      ObjectiveGroup& object,
                      const Filtering& filtering,
                      const FilteringState& filtering_state,
                      const OutputData& output,
                      double fd_epsilon,
                      int max_checks,
                      int Iter) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    Eigen::MatrixXd avm = AVM(fem, analysis, x, x_label, object, filtering, filtering_state, Iter);

    const int nelem = fem.NElem;
    const int nvar = nvar_of(x);
    std::vector<std::pair<int, int>> candidates;
    candidates.reserve(static_cast<std::size_t>(nelem * nvar));
    for (int v = 0; v < nvar; ++v) {
        for (int e = 0; e < nelem; ++e) {
            if (x_label[static_cast<std::size_t>(e)] == 0) candidates.push_back({e, v});
        }
    }
    if (max_checks <= 0 || max_checks > static_cast<int>(candidates.size())) max_checks = static_cast<int>(candidates.size());
    const int stride = std::max(1, static_cast<int>(candidates.size()) / std::max(1, max_checks));

    std::ofstream out;
    if (rank == 0) {
        std::filesystem::create_directories(output.outDir_mat);
        out.open(output.outDir_mat + "/sensitivity_debug.csv");
        out.precision(17);
        out << "Iter,elem,var,x,avm,fd,abs_err,rel_err,obj_plus,obj_minus\n";
        std::cout << "Debug sensitivity: finite-difference check on " << max_checks
                  << " design variables, eps=" << fd_epsilon << "\n";
    }

    int checked = 0;
    for (int idx = 0; idx < static_cast<int>(candidates.size()) && checked < max_checks; idx += stride) {
        const int e = candidates[static_cast<std::size_t>(idx)].first;
        const int v = candidates[static_cast<std::size_t>(idx)].second;
        const double x0 = x(e, v);
        const double h_plus = std::min(fd_epsilon, 1.0 - x0);
        const double h_minus = std::min(fd_epsilon, x0);
        if (h_plus <= 0.0 && h_minus <= 0.0) continue;

        double obj_plus = object.value;
        double obj_minus = object.value;
        if (h_plus > 0.0) {
            Eigen::MatrixXd xp = x;
            FilteringState state_plus = filtering_state;
            xp(e, v) = x0 + h_plus;
            filtering_calculation(xp, x_label, filtering, state_plus);
            dynamic_elastic_linear(analysis, fem, state_plus.xPhys, Iter);
            obj_plus = objective_group_value_for_debug(fem, analysis, state_plus.xPhys, object);
        }
        if (h_minus > 0.0) {
            Eigen::MatrixXd xm = x;
            FilteringState state_minus = filtering_state;
            xm(e, v) = x0 - h_minus;
            filtering_calculation(xm, x_label, filtering, state_minus);
            dynamic_elastic_linear(analysis, fem, state_minus.xPhys, Iter);
            obj_minus = objective_group_value_for_debug(fem, analysis, state_minus.xPhys, object);
        }

        double fd = 0.0;
        if (h_plus > 0.0 && h_minus > 0.0) fd = (obj_plus - obj_minus) / (h_plus + h_minus);
        else if (h_plus > 0.0) fd = (obj_plus - object.value) / h_plus;
        else fd = (object.value - obj_minus) / h_minus;

        if (rank == 0) {
            const double g = avm(e, v);
            const double abs_err = std::abs(g - fd);
            const double rel_err = abs_err / std::max(1e-14, std::abs(fd));
            out << Iter << "," << e << "," << v << "," << x0 << ","
                << g << "," << fd << "," << abs_err << "," << rel_err << ","
                << obj_plus << "," << obj_minus << "\n";
        }
        ++checked;
    }

    dynamic_elastic_linear(analysis, fem, filtering_state.xPhys, Iter);
    return avm;
}

void save_history(const Optimization& Opt, const OutputData& Output) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) return;
    std::filesystem::create_directories(Output.outDir_mat);
    std::ofstream out(Output.outDir_mat + "/optimization_history.csv");
    out << "Iter,objective,constraint,change,rms_change,forward_time,objective_time,constraint_time,objective_sensitivity_time,constraint_sensitivity_time,mma_time,iteration_time\n";
    for (std::size_t i = 0; i < Opt.history.Iter.size(); ++i) {
        out << Opt.history.Iter[i] << ","
            << Opt.history.objective[i] << ","
            << Opt.history.constraint[i] << ","
            << Opt.history.change[i] << ","
            << Opt.history.rms_change[i] << ","
            << Opt.history.forward_time[i] << ","
            << Opt.history.objective_time[i] << ","
            << Opt.history.constraint_time[i] << ","
            << Opt.history.objective_sensitivity_time[i] << ","
            << Opt.history.constraint_sensitivity_time[i] << ","
            << Opt.history.mma_time[i] << ","
            << Opt.history.iteration_time[i] << "\n";
    }
}

std::vector<double> matlab_like_density_display(const FEM& fem,
                                                const Eigen::MatrixXd& xPhys) {
    if (fem.dim != 2 || xPhys.size() == 0) return {};

    const int mTopo = static_cast<int>(xPhys.cols());
    std::vector<double> x_display(static_cast<std::size_t>(fem.nelx * fem.nely), 0.0);
    for (int i = 0; i < fem.nelx; ++i) {
        for (int j = 0; j < fem.nely; ++j) {
            const int e = i * fem.nely + j;
            std::vector<double> Qs(static_cast<std::size_t>(mTopo), 0.0);
            for (int v = 0; v < mTopo; ++v) Qs[static_cast<std::size_t>(v)] = std::max(1e-14, 1.0 - xPhys(e, v));
            double prod_all = 1.0;
            for (double q : Qs) prod_all *= q;
            double multi_topo = 0.0;
            for (int v = 0; v < mTopo; ++v) {
                const double prod_except = prod_all / Qs[static_cast<std::size_t>(v)];
                multi_topo += xPhys(e, v) * prod_except * static_cast<double>(v + 1);
            }
            x_display[static_cast<std::size_t>(j * fem.nelx + i)] = multi_topo;
        }
    }
    return x_display;
}

void save_density_iteration(const FEM& fem,
                            const Eigen::MatrixXd& xPhys,
                            const OutputData& output,
                            int Iter) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0 || xPhys.size() == 0) return;

    const std::filesystem::path dir = output.densityDir.empty()
        ? std::filesystem::path(output.outDir_mat) / "density"
        : std::filesystem::path(output.densityDir);
    std::filesystem::create_directories(dir);

    char name[64] = {0};
    std::snprintf(name, sizeof(name), "density_iter_%04d.csv", Iter);
    std::ofstream csv(dir / name);
    if (!csv) throw std::runtime_error("eigen_backend: cannot write density csv");
    csv.precision(17);
    csv << "elem,i,j,k,xPhys\n";
    for (int e = 0; e < fem.NElem; ++e) {
        int i = 0, j = 0, k = 0;
        if (fem.dim == 2) {
            i = e / fem.nely;
            j = e % fem.nely;
        } else {
            const int layer = fem.nelx * fem.nely;
            k = e / layer;
            const int rem = e % layer;
            i = rem / fem.nely;
            j = rem % fem.nely;
        }
        csv << e << "," << i << "," << j << "," << k << "," << xPhys(e, 0) << "\n";
    }

    if (fem.dim != 2) return;
    const std::vector<double> x_display = matlab_like_density_display(fem, xPhys);
    std::snprintf(name, sizeof(name), "density_iter_%04d.pgm", Iter);
    std::ofstream pgm(dir / name, std::ios::binary);
    if (!pgm) throw std::runtime_error("eigen_backend: cannot write density pgm");
    pgm << "P5\n" << fem.nelx << " " << fem.nely << "\n255\n";
    const double max_val = std::max(1e-14, *std::max_element(x_display.begin(), x_display.end()));
    for (int j = fem.nely - 1; j >= 0; --j) {
        for (int i = 0; i < fem.nelx; ++i) {
            const double val = x_display[static_cast<std::size_t>(j * fem.nelx + i)] / max_val;
            const unsigned char pixel = static_cast<unsigned char>(std::lround(255.0 * std::clamp(val, 0.0, 1.0)));
            pgm.write(reinterpret_cast<const char*>(&pixel), 1);
        }
    }
}

void write_rom_matrix_csv(const std::string& path,
                          const std::vector<std::vector<double>>& values) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("eigen_backend: cannot write " + path);
    out.precision(17);
    for (const auto& row : values) {
        for (std::size_t j = 0; j < row.size(); ++j) {
            if (j) out << ",";
            out << row[j];
        }
        out << "\n";
    }
}

std::vector<std::vector<double>> rom_info_matrix(const std::vector<std::vector<ROMInfo>>& info,
                                                 const std::string& field) {
    std::vector<std::vector<double>> values(info.size());
    for (std::size_t i = 0; i < info.size(); ++i) {
        values[i].resize(info[i].size(), std::numeric_limits<double>::quiet_NaN());
        for (std::size_t j = 0; j < info[i].size(); ++j) {
            const ROMInfo& r = info[i][j];
            if (field == "rom_used") values[i][j] = r.used_rom ? 1.0 : 0.0;
            else if (field == "fom_used") values[i][j] = r.used_rom ? 0.0 : 1.0;
            else if (field == "accepted") values[i][j] = r.accepted ? 1.0 : 0.0;
            else if (field == "residual") values[i][j] = r.residual_norm;
            else if (field == "ksp_iterations") values[i][j] = static_cast<double>(r.ksp_iterations);
            else if (field == "rank") values[i][j] = static_cast<double>(r.rank);
            else if (field == "threshold") values[i][j] = r.threshold;
            else if (field == "error") values[i][j] = r.error;
            else if (field == "basis_error") values[i][j] = r.basis_error;
            else if (field == "timing_total") values[i][j] = r.timing_total;
            else if (field == "timing_petsc_to_eigen_vec") values[i][j] = r.timing_petsc_to_eigen_vec;
            else if (field == "timing_petsc_to_eigen_mat") values[i][j] = r.timing_petsc_to_eigen_mat;
            else if (field == "timing_projection") values[i][j] = r.timing_projection;
            else if (field == "timing_reduced_rhs") values[i][j] = r.timing_reduced_rhs;
            else if (field == "timing_reduced_solve") values[i][j] = r.timing_reduced_solve;
            else if (field == "timing_reconstruct") values[i][j] = r.timing_reconstruct;
            else if (field == "timing_residual") values[i][j] = r.timing_residual;
            else if (field == "timing_fom_fallback") values[i][j] = r.timing_fom_fallback;
            else if (field == "timing_fom_factorization") values[i][j] = r.timing_fom_factorization;
            else if (field == "timing_fom_solve") values[i][j] = r.timing_fom_solve;
            else if (field == "timing_basis_update") values[i][j] = r.timing_basis_update;
            else if (field == "timing_basis1_update") values[i][j] = r.timing_basis1_update;
            else if (field == "timing_online_basis_update") values[i][j] = r.timing_online_basis_update;
            else if (field == "timing_end_basis_update") values[i][j] = r.timing_end_basis_update;
            else if (field == "timing_basis_copy_reset") values[i][j] = r.timing_basis_copy_reset;
            else if (field == "timing_snapshot") values[i][j] = r.timing_snapshot;
        }
    }
    return values;
}

void save_rom_info_group(const std::string& out_dir,
                         const std::string& prefix,
                         const std::shared_ptr<ROMBase>& rom) {
    if (!rom) return;
    std::vector<std::string> fields = {
        "fom_used", "accepted", "residual", "ksp_iterations",
        "timing_total", "timing_fom_factorization", "timing_fom_solve"};
    if (rom->use_mor) {
        fields = {
            "rom_used", "fom_used", "accepted", "residual",
            "ksp_iterations", "rank", "threshold", "error", "basis_error",
            "timing_total", "timing_petsc_to_eigen_vec", "timing_petsc_to_eigen_mat",
            "timing_projection", "timing_reduced_rhs", "timing_reduced_solve",
            "timing_reconstruct", "timing_residual", "timing_fom_fallback",
            "timing_fom_factorization", "timing_fom_solve",
            "timing_basis_update", "timing_basis1_update", "timing_online_basis_update",
            "timing_end_basis_update", "timing_basis_copy_reset",
            "timing_snapshot"};
    }
    for (const std::string& field : fields) {
        write_rom_matrix_csv(out_dir + "/" + prefix + "_" + field + ".csv",
                             rom_info_matrix(rom->ROM_info, field));
    }
}

void save_rom_histories(const std::vector<Analysis>& analysis_all,
                        const Optimization& Opt,
                        const OutputData& Output) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) return;
    std::filesystem::create_directories(Output.outDir_mat);
    if (!analysis_all.empty()) save_rom_info_group(Output.outDir_mat, "forward", analysis_all.front().backend_ROM);
    if (!Opt.objective_groups.empty()) save_rom_info_group(Output.outDir_mat, "obj_sens", Opt.objective_groups.front().backend_ROM);
}

} // namespace

void optimization11(const FEM& fem,
                    std::vector<Analysis>& analysis_all,
                    Optimization& Opt,
                    const OutputData& Output) {
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size != 1) throw std::runtime_error("eigen_backend::optimization11 requires mpi_size=1");
    FilteringState filtering_state;
    filtering_initilization(fem, Opt.filtering, filtering_state);
    const int nelem = fem.NElem;
    const int nvar = nvar_of(Opt.x);
    MatlabMMA mma(nelem * nvar, static_cast<int>(Opt.constraints.size()), 1.0, 0.0, 1000.0, 0.0);

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

    while (Iter <= Opt.MaxIter && (Iter == 0 || Change > Opt.Tol)) {
        const auto iter_t0 = Clock::now();
        if (Iter > 0) ++loopbeta;
        filtering_calculation(Opt.x, Opt.x_label, Opt.filtering, filtering_state);
        if (Opt.save_density_each_iter && Iter % Opt.save_density_interval == 0) {
            save_density_iteration(fem, filtering_state.xPhys, Output, Iter);
        }

        auto t0 = Clock::now();
        for (auto& A : analysis_all) dynamic_elastic_linear(A, fem, filtering_state.xPhys, Iter);
        auto t1 = Clock::now();
        forward_time = elapsed(t0, t1);

        t0 = Clock::now();
        evaluate_objective(fem, analysis_all, filtering_state.xPhys, Opt.objective_groups, obj_value);
        t1 = Clock::now();
        objective_time = elapsed(t0, t1);

        t0 = Clock::now();
        evaluate_constraint(fem, analysis_all, filtering_state.xPhys, Opt.constraints, constr_value_vec);
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

        dObj_dx_mat = zeros(nelem, nvar);
        t0 = Clock::now();
        for (auto& group : Opt.objective_groups) {
            Analysis& A = analysis_all[static_cast<std::size_t>(group.analysis_id - 1)];
            Eigen::MatrixXd dg = lower(group.sens_type) == "debug"
                ? Debug(fem, A, Opt.x, Opt.x_label, group, Opt.filtering, filtering_state, Output, Opt.debug_fd_epsilon, Opt.debug_fd_max_checks, Iter)
                : AVM(fem, A, Opt.x, Opt.x_label, group, Opt.filtering, filtering_state, Iter);
            add_scaled(dObj_dx_mat, dg, 1.0);
        }
        t1 = Clock::now();
        objective_sensitivity_time = elapsed(t0, t1);
        dObj_dx_vec = flatten(dObj_dx_mat);

        dConstr_dx.clear();
        t0 = Clock::now();
        for (auto& c : Opt.constraints) {
            Analysis& A = analysis_all[static_cast<std::size_t>(c.analysis_id - 1)];
            Eigen::MatrixXd dc = AVM(fem, A, Opt.x, Opt.x_label, c, Opt.filtering, filtering_state, Iter);
            std::vector<double> flat = flatten(dc);
            dConstr_dx.insert(dConstr_dx.end(), flat.begin(), flat.end());
        }
        t1 = Clock::now();
        constraint_sensitivity_time = elapsed(t0, t1);

        Opt.history.Iter.push_back(Iter);
        Opt.history.objective.push_back(obj_value);
        Opt.history.constraint.push_back(constr_value_vec.empty() ? 0.0 : constr_value_vec.front());
        Opt.history.change.push_back(Iter == 0 ? 0.0 : Change);
        Opt.history.rms_change.push_back(0.0);
        Opt.history.forward_time.push_back(forward_time);
        Opt.history.objective_time.push_back(objective_time);
        Opt.history.constraint_time.push_back(constraint_time);
        Opt.history.objective_sensitivity_time.push_back(objective_sensitivity_time);
        Opt.history.constraint_sensitivity_time.push_back(constraint_sensitivity_time);
        Opt.history.mma_time.push_back(0.0);
        Opt.history.iteration_time.push_back(0.0);

        if (Iter >= Opt.MaxIter) {
            Opt.history.iteration_time.back() = elapsed(iter_t0, Clock::now());
            //if (rank == 0) std::cout << " mma=0 iter_total=" << Opt.history.iteration_time.back() << "\n";
            save_history(Opt, Output);
            save_rom_histories(analysis_all, Opt, Output);
            break;
        }

        t0 = Clock::now();
        std::vector<double> xvec = flatten(Opt.x);
        std::vector<double> xmin(xvec.size()), xmax(xvec.size());
        for (std::size_t i = 0; i < xvec.size(); ++i) {
            xmin[i] = std::max(0.0, xvec[i] - Opt.algorithm.move);
            xmax[i] = std::min(1.0, xvec[i] + Opt.algorithm.move);
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
        Opt.history.rms_change.back() = rms;
        Opt.x = unflatten(xvec, nelem, nvar);
        t1 = Clock::now();
        mma_time = elapsed(t0, t1);
        Opt.history.mma_time.back() = mma_time;
        Opt.history.iteration_time.back() = elapsed(iter_t0, Clock::now());
        //if (rank == 0) std::cout << " mma=" << mma_time << " iter_total=" << Opt.history.iteration_time.back() << "\n";

        if (Opt.filtering.beta < Opt.filtering.beta_max && (loopbeta >= Opt.filtering.beta_loop || Change <= Opt.Tol)) {
            Opt.filtering.beta = std::min(Opt.filtering.beta + 2.0, Opt.filtering.beta_max);
            loopbeta = 0;
            Change = 1.0;
            if (rank == 0) std::cout << "Parameter beta increased to " << Opt.filtering.beta << ".\n";
        }
        save_history(Opt, Output);
        save_rom_histories(analysis_all, Opt, Output);
        ++Iter;
    }
    save_history(Opt, Output);
    save_rom_histories(analysis_all, Opt, Output);
}

} // namespace eigen_backend
