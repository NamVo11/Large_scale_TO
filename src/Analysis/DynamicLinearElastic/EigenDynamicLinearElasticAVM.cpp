#include "Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/StructuralElement/StructuralElement.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"

#include <petscsys.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <omp.h>

namespace {

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
}

void set_from_vector(Eigen::VectorXd& v, const Eigen::VectorXd& values, double scale = 1.0) {
    v.setZero();
    if (values.size() == 0) return;
    if (values.size() != v.size()) {
        throw std::runtime_error("eigen_backend set_from_vector: size mismatch");
    }
    v.noalias() = scale * values;
}

void set_fixed_from_vector(Eigen::VectorXd& v,
                           const std::vector<int>& fixed,
                           const Eigen::VectorXd& values,
                           double scale = 1.0) {
    if (values.size() == 0) return;
    if (values.size() != v.size()) {
        throw std::runtime_error("eigen_backend set_fixed_from_vector: size mismatch");
    }
    for (int id : fixed) v(id) = scale * values(id);
}

void zero_fixed(Eigen::VectorXd& v, const std::vector<int>& fixed) {
    for (int id : fixed) v(id) = 0.0;
}

Eigen::MatrixXd explicit_gradient_dynamic(const FEM& fem,
                                          DynamicLinearElastic& analysis,
                                          const Eigen::MatrixXd& xPhys,
                                          DynamicLinearElasticAVM& object,
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
    const bool has_damping = analysis.alpha_M != 0.0 || analysis.alpha_K != 0.0;

    if (xPhys.rows() != fem.NElem) throw std::runtime_error("explicit_gradient_dynamic: xPhys row mismatch");
    if (analysis.U.cols() < nstep + 1 ||
        analysis.A.cols() < nstep + 1 ||
        lambda_all.cols() < nstep + 1) {
        throw std::runtime_error("explicit_gradient_dynamic: insufficient time-step columns");
    }
    if (has_damping && analysis.V.cols() < nstep + 1) {
        throw std::runtime_error("explicit_gradient_dynamic: damping sensitivity requires velocity history");
    }
    if (analysis.U.rows() != analysis.A.rows() ||
        analysis.U.rows() != lambda_all.rows()) {
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

    for (const auto& region_ptr : fem.Regions) {
        const Region& region = *region_ptr;
        if (region.status != "design_val") continue;
        if (region.elem_ids.empty()) continue;
        const auto region_element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front())));
        if (!region_element || !region_element->material_paras) {
            throw std::runtime_error("Element material is not initialized");
        }
        const auto& region_material = dynamic_cast<const ElasticParas&>(*region_element->material_paras);
        if (region_material.val_ids.empty()) throw std::runtime_error("explicit_gradient_dynamic: no val_ids");
        if (region_material.E_values.empty()) throw std::runtime_error("explicit_gradient_dynamic: no elastic parameters");
        if (region_material.rho_values.empty()) throw std::runtime_error("explicit_gradient_dynamic: no density parameters");
        if (!analysis.interp) throw std::runtime_error("Analysis interpolation is not initialized");

        for (int val_id : region_material.val_ids) {
            if (val_id <= 0 || val_id > nvar) throw std::runtime_error("explicit_gradient_dynamic: val_id outside xPhys");
        }

        for (int elem_id : region.elem_ids) {
            if (elem_id < 0 || elem_id >= fem.NElem) throw std::runtime_error("explicit_gradient_dynamic: invalid elem id");
            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            if (dofs.size() > 24) throw std::runtime_error("explicit_gradient_dynamic: element ndof > 24");
        }

        const double region_nu = region_material.nu_values.front();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            uniform_em = fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front()))->compute_element_matrices(fem, region_nu, fem.thickness);
            if (uniform_em.ndof > 24) throw std::runtime_error("explicit_gradient_dynamic: uniform element ndof > 24");
        }

        const int region_nelem = static_cast<int>(region.elem_ids.size());
        #pragma omp parallel for schedule(static) num_threads(sens_element_threads) if(sens_element_threads > 1)
        for (int local_e = 0; local_e < region_nelem; ++local_e) {
            const int elem_id = region.elem_ids[static_cast<std::size_t>(local_e)];

            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                local_em = fem.ElementObjects.at(static_cast<std::size_t>(elem_id))->compute_element_matrices(fem, region_nu, fem.thickness);
                em_ptr = &local_em;
            }

            const ElementMatrix& em = *em_ptr;
            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));

            Eigen::Matrix<double, 24, 1> u_elem_buffer;
            Eigen::Matrix<double, 24, 1> v_elem_buffer;
            Eigen::Matrix<double, 24, 1> acc_elem_buffer;
            Eigen::Matrix<double, 24, 1> lambda_elem_buffer;
            Eigen::Matrix<double, 24, 1> Ku_elem_buffer;
            Eigen::Matrix<double, 24, 1> Ma_elem_buffer;

            double sum_lambda_dot_Ku = 0.0;
            double sum_lambda_dot_Ma = 0.0;

            for (int k = 0; k <= nstep; ++k) {
                const auto u = analysis.U.col(k);
                const auto acc = analysis.A.col(k);
                const auto lambda = lambda_all.col(k);

                for (int i = 0; i < em.ndof; ++i) {
                    const int dof_i = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                    u_elem_buffer(i) = u(dof_i);
                    if (has_damping) v_elem_buffer(i) = analysis.V(dof_i, k);
                    acc_elem_buffer(i) = acc(dof_i);
                    lambda_elem_buffer(i) = lambda(dof_i);
                }

                const auto u_elem = u_elem_buffer.head(em.ndof);
                const auto acc_elem = acc_elem_buffer.head(em.ndof);
                const auto lambda_elem = lambda_elem_buffer.head(em.ndof);
                auto Ku_elem = Ku_elem_buffer.head(em.ndof);
                auto Ma_elem = Ma_elem_buffer.head(em.ndof);

                if (!has_damping) {
                    Ku_elem.noalias() = em.ke * u_elem;
                    if (k == 0) Ma_elem.setZero();
                    else Ma_elem.noalias() = em.me * acc_elem;
                } else if (k == 0) {
                    Ku_elem.noalias() = em.ke * u_elem;
                    Ma_elem.setZero();
                } else {
                    const auto v_elem = v_elem_buffer.head(em.ndof);
                    Ku_elem.noalias() = em.ke * (u_elem + analysis.alpha_K * v_elem);
                    Ma_elem.noalias() = em.me * (acc_elem + analysis.alpha_M * v_elem);
                }

                sum_lambda_dot_Ku += lambda_elem.dot(Ku_elem);
                sum_lambda_dot_Ma += lambda_elem.dot(Ma_elem);
            }

            const auto element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(elem_id)));
            if (!element || !element->material_paras) throw std::runtime_error("Element material is not initialized");
            const auto& material = dynamic_cast<const ElasticParas&>(*element->material_paras);
            for (int val_id : material.val_ids) {
                const double x = xPhys(elem_id, val_id - 1);
                const bool is_design = myjson::lower_ascii(element->status) == "design_val";
                const double dE = analysis.interp->sensitivity(material.E_values, x, is_design, val_id, analysis.interp->modulus_penalty());
                const double drho = analysis.interp->sensitivity(material.rho_values, x, is_design, val_id, analysis.interp->mass_density_penalty());
                df_dx(elem_id, val_id - 1) += dE * sum_lambda_dot_Ku + drho * sum_lambda_dot_Ma;
            }
        }
    }

    const auto t1 = Clock::now();
    const double timing_element_loop = elapsed(t0, t1);

    const auto add_begin = Clock::now();
    for (int val_id = 0; val_id < nvar; ++val_id) {
        const Eigen::VectorXd explicit_dfdx = object.dfdx(fem,
                                                          analysis,
                                                          xPhys,
                                                          val_id);
        if (explicit_dfdx.size() == 0) continue;
        if (explicit_dfdx.size() != df_dx.rows()) {
            throw std::runtime_error("explicit_gradient_dynamic: explicit dfdx size mismatch");
        }
        df_dx.col(val_id).noalias() += explicit_dfdx;
    }
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

} // namespace

Eigen::MatrixXd EigenDynamicLinearElasticAVM::AVM(const FEM& fem,
                                                  Analysis& analysis_base,
                                                  const Eigen::MatrixXd&,
                                                  const std::vector<int>& x_label,
                                                  const Filtering& filtering,
                                                  int Iter) {
    auto& analysis = dynamic_cast<EigenDynamicLinearElastic&>(analysis_base);
    const int nstep = analysis.Nstep;
    const double beta = 0.25;
    const double gamma = 0.5;
    const double dt = analysis.dt;
    const double a0c = 1.0 / (beta * dt * dt);
    const double a1c = gamma / (beta * dt);
    const double c_eta = gamma / beta - 1.0;
    const double c_mu = (gamma / (2.0 * beta) - 1.0) * dt;
    const double dHdu = a0c;
    const double dLdu = a1c;
    const double dHda_new = -(0.5 / beta - 1.0);
    const double dHdv_new = -1.0 / (beta * dt);
    const double dHdu_new = -a0c;
    const double dLda_new = (1.0 - 0.5 * gamma / beta) * dt;
    const double dLdv_new = 1.0 - gamma / beta;
    const double dLdu_new = -gamma / (beta * dt);

    const auto& K0 = analysis.K_inf;
    const auto& Keff = analysis.K_eff;
    const auto& M = analysis.M;
    const auto& C = analysis.C;
    const bool has_damping = analysis.alpha_M != 0.0 || analysis.alpha_K != 0.0;
    if (K0.rows() == 0 || Keff.rows() == 0 || M.rows() == 0) {
        throw std::runtime_error("eigen_backend sensitivity requires Eigen forward matrices");
    }
    if (has_damping && C.rows() == 0) {
        throw std::runtime_error("EigenDynamicLinearElasticAVM requires damping matrix");
    }
    if (!ROM) throw std::runtime_error("EigenDynamicLinearElasticAVM requires ROM initialized");

    std::shared_ptr<ROMBase>& rom = ROM;
    if (analysis.factorized_K_eff) {
        rom->set_factorized_solver(Keff, analysis.factorized_K_eff);
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
    Eigen::VectorXd Clambda;
    if (has_damping) Clambda = Eigen::VectorXd::Zero(ndof);
    Eigen::MatrixXd lambda_all(ndof, nstep + 1);

    int k = nstep;
    set_from_vector(mu, dfda(fem, analysis, filtering.xPhys, k));
    set_from_vector(eta, dfdv(fem, analysis, filtering.xPhys, k));
    set_from_vector(rhs, dfdu(fem, analysis, filtering.xPhys, k), -1.0);
    rhs.noalias() += -dHdu * mu - dLdu * eta;
    set_fixed_from_vector(rhs, analysis.SPCdofs, dfdFext(fem, analysis, filtering.xPhys, k), -1.0);
    rom->InitializeROMContext(analysis, Iter, k + 1, k,
                              k * analysis.dt, "adjoint",
                              "newmark_adjoint_effective", k == 0);
    rom->main(Keff, rhs, lambda);
    lambda_all.col(k) = lambda;

    for (k = nstep - 1; k >= 0; --k) {
        lambda_new = lambda;
        mu_new = mu;
        eta_new = eta;
        Mlambda.noalias() = M * lambda_new;
        if (has_damping) Clambda.noalias() = C * lambda_new;

        set_from_vector(mu, dfda(fem, analysis, filtering.xPhys, k));
        mu.noalias() += -(0.5 / beta - 1.0) * Mlambda + dHda_new * mu_new + dLda_new * eta_new;
        if (has_damping) mu.noalias() += -c_mu * Clambda;
        set_from_vector(eta, dfdv(fem, analysis, filtering.xPhys, k));
        eta.noalias() += -(1.0 / (beta * dt)) * Mlambda + dHdv_new * mu_new + dLdv_new * eta_new;
        if (has_damping) eta.noalias() += -c_eta * Clambda;
        zero_fixed(mu, analysis.SPCdofs);
        zero_fixed(eta, analysis.SPCdofs);

        set_from_vector(rhs, dfdu(fem, analysis, filtering.xPhys, k), -1.0);
        rhs.noalias() += a0c * Mlambda - dHdu_new * mu_new - dLdu_new * eta_new;
        if (has_damping) rhs.noalias() += a1c * Clambda;
        if (k != 0) rhs.noalias() += -dHdu * mu - dLdu * eta;
        set_fixed_from_vector(rhs, analysis.SPCdofs, dfdFext(fem, analysis, filtering.xPhys, k), -1.0);
        if (k == 0) {
            rom->set_factorized_solver(Keff, nullptr);
            analysis.factorized_K_eff.reset();
            if (analysis.factorized_K_initial) {
                rom->set_factorized_solver(K0, analysis.factorized_K_initial);
            }
           rom->InitializeROMContext(analysis, Iter, k + 1, k, k * analysis.dt,
                                    "adjoint", "newmark_adjoint_effective", k == 0);
            rom->main(K0, rhs, lambda);
            rom->set_factorized_solver(K0, nullptr);
            analysis.factorized_K_initial.reset();
        } else {
            rom->InitializeROMContext(analysis, Iter, k + 1, k, k * analysis.dt,
                                    "adjoint", "newmark_adjoint_effective", k == 0);
            rom->main(Keff, rhs, lambda);
        }
        lambda_all.col(k) = lambda;
    }

    Eigen::MatrixXd dfdx_phys = explicit_gradient_dynamic(fem, analysis, filtering.xPhys, *this, lambda_all);
    Eigen::MatrixXd out = filtering.sensitivity(dfdx_phys);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}
