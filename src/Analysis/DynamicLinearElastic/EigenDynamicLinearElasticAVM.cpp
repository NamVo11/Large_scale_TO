#include "Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"

#include <petscsys.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <omp.h>

namespace {

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
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
    const auto& dynamic = dynamic_cast<const DynamicLinearElastic&>(analysis);
    ROMContext ctx;
    ctx.analysis_id = analysis.id;
    ctx.opt_iter = Iter;
    ctx.solve_step = time_step + 1;
    ctx.time_step = time_step;
    ctx.time = time_step * dynamic.dt;
    ctx.role = "adjoint";
    ctx.system_type = "newmark_adjoint_effective";
    ctx.is_end_of_analysis = time_step == 0;
    return ctx;
}

// ================= TEMP DEBUG: adjoint/forward RHS at force DOFs =================
void write_temp_adjoint_rhs_debug(const Analysis& analysis,
                                  int Iter,
                                  int k,
                                  const Eigen::VectorXd& rhs_adjoint) {
    constexpr const char* path =
        "/home/votrongnam/PROJECT/Large_scale_TO/case_study/thuuuu/adjoint_rhs_debug.csv";

    if (analysis.results.F_ext.size() == 0 ||
        k < 0 || k >= analysis.results.F_ext.cols()) {
        return;
    }

    const bool write_header = [] {
        std::ifstream in(path);
        return !in.good() || in.peek() == std::ifstream::traits_type::eof();
    }();

    std::ofstream out(path, std::ios::app);
    if (!out) return;

    if (write_header) {
        out << "Iter,step,"
            << "adj_rhs_norm_global,forward_rhs_norm_global,"
            << "adj_force_sum,forward_force_sum,"
            << "adj_force_mean,forward_force_mean,"
            << "adj_force_abs_mean,forward_force_abs_mean,"
            << "adj_force_norm,forward_force_norm,"
            << "adj_force_max_abs,forward_force_max_abs,"
            << "num_force_dofs\n";
    }

    std::vector<int> force_dofs;
    force_dofs.reserve(static_cast<std::size_t>(analysis.results.F_ext.rows()));
    for (Eigen::Index dof = 0; dof < analysis.results.F_ext.rows(); ++dof) {
        const double row_max = analysis.results.F_ext.row(dof).cwiseAbs().maxCoeff();
        if (row_max > 0.0) force_dofs.push_back(static_cast<int>(dof));
    }

    double adj_sum = 0.0;
    double fwd_sum = 0.0;
    double adj_abs_sum = 0.0;
    double fwd_abs_sum = 0.0;
    double adj_sq_sum = 0.0;
    double fwd_sq_sum = 0.0;
    double adj_max_abs = 0.0;
    double fwd_max_abs = 0.0;

    for (int dof : force_dofs) {
        if (dof < 0 || dof >= rhs_adjoint.size() || dof >= analysis.results.F_ext.rows()) continue;
        const double adj_val = rhs_adjoint(dof);
        const double fwd_val = analysis.results.F_ext(dof, k);
        const double adj_abs = std::abs(adj_val);
        const double fwd_abs = std::abs(fwd_val);

        adj_sum += adj_val;
        fwd_sum += fwd_val;
        adj_abs_sum += adj_abs;
        fwd_abs_sum += fwd_abs;
        adj_sq_sum += adj_val * adj_val;
        fwd_sq_sum += fwd_val * fwd_val;
        adj_max_abs = std::max(adj_max_abs, adj_abs);
        fwd_max_abs = std::max(fwd_max_abs, fwd_abs);
    }

    const double n = static_cast<double>(force_dofs.size());
    const Eigen::VectorXd forward_rhs = analysis.results.F_ext.col(k);

    out << std::scientific << std::setprecision(16)
        << Iter << ','
        << k << ','
        << rhs_adjoint.norm() << ','
        << forward_rhs.norm() << ','
        << adj_sum << ','
        << fwd_sum << ','
        << (n > 0.0 ? adj_sum / n : 0.0) << ','
        << (n > 0.0 ? fwd_sum / n : 0.0) << ','
        << (n > 0.0 ? adj_abs_sum / n : 0.0) << ','
        << (n > 0.0 ? fwd_abs_sum / n : 0.0) << ','
        << std::sqrt(adj_sq_sum) << ','
        << std::sqrt(fwd_sq_sum) << ','
        << adj_max_abs << ','
        << fwd_max_abs << ','
        << force_dofs.size()
        << '\n';
}
// ================= END TEMP DEBUG: adjoint/forward RHS at force DOFs =================

Eigen::MatrixXd explicit_gradient_dynamic(const FEM& fem,
                                          const Analysis& analysis,
                                          const Eigen::MatrixXd& xPhys,
                                          const DynamicLinearElasticAVM& object,
                                          const Eigen::MatrixXd& lambda_all) {
    using Clock = std::chrono::steady_clock;

    const auto timer_begin = Clock::now();
    const auto elapsed = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const int nstep = analysis.Nstep;
    const int nvar = nvar_of(xPhys);

    if (xPhys.rows() != fem.NElem) throw std::runtime_error("explicit_gradient_dynamic: xPhys row mismatch");
    if (analysis.results.U.cols() < nstep + 1 ||
        analysis.results.A.cols() < nstep + 1 ||
        lambda_all.cols() < nstep + 1) {
        throw std::runtime_error("explicit_gradient_dynamic: insufficient time-step columns");
    }
    if (analysis.results.U.rows() != analysis.results.A.rows() ||
        analysis.results.U.rows() != lambda_all.rows()) {
        throw std::runtime_error("explicit_gradient_dynamic: U, A and lambda_all row mismatch");
    }

    int sens_element_threads = 1;
    if (const char* env = std::getenv("SENS_ELEMENT_THREADS")) {
        sens_element_threads = std::max(1, std::atoi(env));
    }
    if (size != 1) sens_element_threads = 1;

    Eigen::MatrixXd df_dx = Eigen::MatrixXd::Zero(fem.NElem, nvar);

    if (rank == 0) {
        int actual_threads = 1;
        #pragma omp parallel num_threads(sens_element_threads) if(sens_element_threads > 1)
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }
        std::cout << "  thread_debug explicit_gradient_dynamic:"
                  << " requested_threads=" << sens_element_threads
                  << " actual_threads=" << actual_threads
                  << " omp_get_max_threads=" << omp_get_max_threads()
                  << '\n';
    }

    const auto t0 = Clock::now();

    for (const Region& region : analysis.regions) {
        if (region.status != "design_val") continue;
        if (region.elem_ids.empty()) continue;
        if (region.mat_paras.val_ids.empty()) throw std::runtime_error("explicit_gradient_dynamic: no val_ids");
        if (region.mat_paras.E_inf.empty()) throw std::runtime_error("explicit_gradient_dynamic: no elastic parameters");
        if (region.mat_paras.rho.empty()) throw std::runtime_error("explicit_gradient_dynamic: no density parameters");

        for (int val_id : region.mat_paras.val_ids) {
            if (val_id <= 0 || val_id > nvar) throw std::runtime_error("explicit_gradient_dynamic: val_id outside xPhys");
        }

        for (int elem_id : region.elem_ids) {
            if (elem_id < 0 || elem_id >= fem.NElem) throw std::runtime_error("explicit_gradient_dynamic: invalid elem id");
            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            if (dofs.size() > 24) throw std::runtime_error("explicit_gradient_dynamic: element ndof > 24");
        }

        const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            uniform_em = fem.compute_element_matrices(region.elem_ids.front(), region_nu, fem.Thickness);
            if (uniform_em.ndof > 24) throw std::runtime_error("explicit_gradient_dynamic: uniform element ndof > 24");
        }

        const int region_nelem = static_cast<int>(region.elem_ids.size());
        #pragma omp parallel for schedule(static) num_threads(sens_element_threads) if(sens_element_threads > 1)
        for (int local_e = 0; local_e < region_nelem; ++local_e) {
            const int elem_id = region.elem_ids[static_cast<std::size_t>(local_e)];

            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                local_em = fem.compute_element_matrices(elem_id, region_nu, fem.Thickness);
                em_ptr = &local_em;
            }

            const ElementMatrix& em = *em_ptr;
            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));

            Eigen::Matrix<double, 24, 1> u_elem_buffer;
            Eigen::Matrix<double, 24, 1> acc_elem_buffer;
            Eigen::Matrix<double, 24, 1> lambda_elem_buffer;
            Eigen::Matrix<double, 24, 1> Ku_elem_buffer;
            Eigen::Matrix<double, 24, 1> Ma_elem_buffer;

            double sum_lambda_dot_Ku = 0.0;
            double sum_lambda_dot_Ma = 0.0;

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

                const auto u_elem = u_elem_buffer.head(em.ndof);
                const auto acc_elem = acc_elem_buffer.head(em.ndof);
                const auto lambda_elem = lambda_elem_buffer.head(em.ndof);
                auto Ku_elem = Ku_elem_buffer.head(em.ndof);
                auto Ma_elem = Ma_elem_buffer.head(em.ndof);

                Ku_elem.noalias() = em.ke * u_elem;
                if (k == 0) Ma_elem.setZero();
                else Ma_elem.noalias() = em.me * acc_elem;

                sum_lambda_dot_Ku += lambda_elem.dot(Ku_elem);
                sum_lambda_dot_Ma += lambda_elem.dot(Ma_elem);
            }

            for (int val_id : region.mat_paras.val_ids) {
                const double x = xPhys(elem_id, val_id - 1);
                const double dE = ramp_derivative(region.mat_paras.E_inf, x, region.interp.p_E_elastic);
                const double drho = ramp_derivative(region.mat_paras.rho, x, region.interp.p_rho);
                df_dx(elem_id, val_id - 1) += dE * sum_lambda_dot_Ku + drho * sum_lambda_dot_Ma;
            }
        }
    }

    const auto t1 = Clock::now();
    const double timing_element_loop = elapsed(t0, t1);

    const auto add_begin = Clock::now();
    add_scaled(df_dx, object.d_explicit_df_dx, 1.0);
    const auto add_end = Clock::now();

    if (rank == 0) {
        const double timing_total = elapsed(timer_begin, Clock::now());
        std::cout << "  explicit_gradient_dynamic[s]:"
                  << " threads=" << sens_element_threads
                  << " element_loop=" << timing_element_loop
                  << " unflatten_add=" << elapsed(add_begin, add_end)
                  << " total=" << timing_total
                  << '\n';
    }
    return df_dx;
}

Eigen::MatrixXd dynamic_elastic_base_full_impl(const FEM& fem,
                                               Analysis& analysis,
                                               DynamicLinearElasticAVM& object,
                                               const Filtering& filtering,
                                               const FilteringState& filtering_state,
                                               int Iter) {
    auto& dynamic = dynamic_cast<DynamicLinearElastic&>(analysis);
    const int nstep = analysis.Nstep;
    const double beta = 0.25;
    const double gamma = 0.5;
    const double dt = dynamic.dt;
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
    if (!object.ROM) throw std::runtime_error("EigenDynamicLinearElasticAVM requires ROM initialized");

    std::shared_ptr<ROMBase>& rom = object.ROM;
    if (analysis.results.factorized_K_eff_eigen) {
        rom->set_factorized_solver(Keff, analysis.results.factorized_K_eff_eigen);
    }

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
    write_temp_adjoint_rhs_debug(analysis, Iter, k, rhs);
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
            rom->set_factorized_solver(Keff, nullptr);
            analysis.results.factorized_K_eff_eigen.reset();
            if (analysis.results.factorized_K_initial_eigen) {
                rom->set_factorized_solver(K0, analysis.results.factorized_K_initial_eigen);
            }
            write_temp_adjoint_rhs_debug(analysis, Iter, k, rhs);
            rom->main(K0, rhs, lambda, make_adjoint_context(analysis, Iter, k));
            rom->set_factorized_solver(K0, nullptr);
            analysis.results.factorized_K_initial_eigen.reset();
        } else {
            write_temp_adjoint_rhs_debug(analysis, Iter, k, rhs);
            rom->main(Keff, rhs, lambda, make_adjoint_context(analysis, Iter, k));
        }
        lambda_all.col(k) = lambda;
    }

    analysis.results.Lambda = lambda_all;
    Eigen::MatrixXd dfdx_phys = explicit_gradient_dynamic(fem, analysis, filtering_state.xPhys, object, lambda_all);
    return filtering.sensitivity(dfdx_phys, filtering_state);
}

} // namespace

Eigen::MatrixXd EigenDynamicLinearElasticAVM::AVM(const FEM& fem,
                                                  Analysis& analysis,
                                                  const Eigen::MatrixXd&,
                                                  const std::vector<int>& x_label,
                                                  const Filtering& filtering,
                                                  const FilteringState& filtering_state,
                                                  int Iter) {
    Eigen::MatrixXd out = dynamic_elastic_base_full_impl(fem, analysis, *this, filtering, filtering_state, Iter);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}
