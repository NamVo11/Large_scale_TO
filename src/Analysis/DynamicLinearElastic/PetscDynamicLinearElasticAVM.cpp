#include "Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/StructuralElement/StructuralElement.hpp"
#include "FEM/Element/StructuralElement/StructuralElementSmallStrain.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"
#include "helper/MemoryDebug.hpp"
#include "helper/PetscEigenBridge.hpp"
#include "helper/PetscEigenHelper.hpp"

#include <petscksp.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
}

Eigen::MatrixXd unflatten(const std::vector<double>& x, int nelem, int nvar) {
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
        x.data(), nelem, nvar);
}

void set_vec_from_eigen_vector(Vec v, const Eigen::VectorXd& values, double scale = 1.0) {
    if (values.size() == 0) {
        VecSet(v, 0.0);
        return;
    }

    PetscMPIInt size = 1;
    MPI_Comm_size(PetscObjectComm(reinterpret_cast<PetscObject>(v)), &size);
    if (size == 1) {
        PetscInt n = 0;
        VecGetLocalSize(v, &n);
        if (values.size() != static_cast<Eigen::Index>(n)) {
            throw std::runtime_error("set_vec_from_eigen_vector: vector size mismatch");
        }
        PetscScalar* a = nullptr;
        VecGetArray(v, &a);
        for (PetscInt i = 0; i < n; ++i) a[i] = scale * values(static_cast<Eigen::Index>(i));
        VecRestoreArray(v, &a);
        return;
    }

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PetscObjectComm(reinterpret_cast<PetscObject>(v)), &rank);
    VecSet(v, 0.0);
    if (rank == 0) {
        std::vector<PetscInt> ids(static_cast<std::size_t>(values.size()));
        std::vector<PetscScalar> vals(static_cast<std::size_t>(values.size()));
        std::iota(ids.begin(), ids.end(), 0);
        for (Eigen::Index i = 0; i < values.size(); ++i) {
            vals[static_cast<std::size_t>(i)] = scale * values(i);
        }
        VecSetValues(v, static_cast<PetscInt>(ids.size()), ids.data(), vals.data(), INSERT_VALUES);
    }
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}

void set_vec_fixed_from_eigen_vector(Vec v,
                                     const std::vector<int>& fixed,
                                     const Eigen::VectorXd& values,
                                     double scale = 1.0) {
    if (fixed.empty() || values.size() == 0) return;

    PetscMPIInt size = 1;
    MPI_Comm_size(PetscObjectComm(reinterpret_cast<PetscObject>(v)), &size);
    if (size == 1) {
        PetscInt n = 0;
        VecGetLocalSize(v, &n);
        if (values.size() != static_cast<Eigen::Index>(n)) {
            throw std::runtime_error("set_vec_fixed_from_eigen_vector: vector size mismatch");
        }
        PetscScalar* a = nullptr;
        VecGetArray(v, &a);
        for (int id : fixed) {
            if (id < 0 || id >= n) {
                VecRestoreArray(v, &a);
                throw std::runtime_error("set_vec_fixed_from_eigen_vector: fixed dof out of range");
            }
            a[id] = scale * values(static_cast<Eigen::Index>(id));
        }
        VecRestoreArray(v, &a);
        return;
    }

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PetscObjectComm(reinterpret_cast<PetscObject>(v)), &rank);
    if (rank == 0) {
        std::vector<PetscInt> ids(fixed.begin(), fixed.end());
        std::vector<PetscScalar> vals(fixed.size());
        for (std::size_t i = 0; i < fixed.size(); ++i) {
            vals[i] = scale * values(static_cast<Eigen::Index>(fixed[i]));
        }
        VecSetValues(v, static_cast<PetscInt>(ids.size()), ids.data(), vals.data(), INSERT_VALUES);
    }
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}

void zero_fixed(Vec v, const std::vector<int>& fixed) {
    if (fixed.empty()) return;

    PetscMPIInt size = 1;
    MPI_Comm_size(PetscObjectComm(reinterpret_cast<PetscObject>(v)), &size);
    if (size == 1) {
        PetscInt n = 0;
        VecGetLocalSize(v, &n);
        PetscScalar* a = nullptr;
        VecGetArray(v, &a);
        for (int id : fixed) {
            if (id < 0 || id >= n) throw std::runtime_error("zero_fixed: fixed dof out of range");
            a[id] = 0.0;
        }
        VecRestoreArray(v, &a);
        return;
    }

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
        std::vector<PetscInt> ids(fixed.begin(), fixed.end());
        std::vector<PetscScalar> vals(ids.size(), 0.0);
        VecSetValues(v, static_cast<PetscInt>(ids.size()), ids.data(), vals.data(), INSERT_VALUES);
    }
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
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
    const int elem_start = (fem.NElem * rank) / size;
    const int elem_end = (fem.NElem * (rank + 1)) / size;

    if (xPhys.rows() != fem.NElem) {
        throw std::runtime_error("explicit_gradient_dynamic: xPhys row count does not match fem.NElem");
    }
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

    auto t0 = Clock::now();
    std::vector<double> df_dx_local(static_cast<std::size_t>(fem.NElem) * static_cast<std::size_t>(nvar), 0.0);
    auto t1 = Clock::now();
    const double timing_alloc_gradient = elapsed(t0, t1);

    t0 = Clock::now();
    for (const auto& region_ptr : fem.Regions) {
        const Region& region = *region_ptr;
        if (region.status != "design_val") continue;
        if (region.elem_ids.empty()) continue;
        const auto region_element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front())));
        if (!region_element || !region_element->material_paras) {
            throw std::runtime_error("Element material is not initialized");
        }
        const auto& region_material = dynamic_cast<const ElasticParas&>(*region_element->material_paras);
        if (region_material.val_ids.empty()) {
            throw std::runtime_error("explicit_gradient_dynamic: design_val region has no val_ids");
        }
        if (region_material.E_values.empty()) {
            throw std::runtime_error("explicit_gradient_dynamic: design_val region has no elastic parameters");
        }
        if (region_material.rho_values.empty()) {
            throw std::runtime_error("explicit_gradient_dynamic: design_val region has no density parameters");
        }
        if (!analysis.interp) throw std::runtime_error("Analysis interpolation is not initialized");

        for (int val_id : region_material.val_ids) {
            if (val_id <= 0 || val_id > nvar) {
                throw std::runtime_error("explicit_gradient_dynamic: region val_id is outside xPhys columns");
            }
        }

        for (int elem_id : region.elem_ids) {
            if (elem_id < 0 || elem_id >= fem.NElem) {
                throw std::runtime_error("explicit_gradient_dynamic: invalid element id in region");
            }
            if (elem_id < elem_start || elem_id >= elem_end) continue;
            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            if (dofs.size() > 24) {
                throw std::runtime_error("explicit_gradient_dynamic: element ndof > 24 is not supported");
            }
        }

        const double region_nu = region_material.nu_values.front();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            region_element->nu_interp = region_nu;
            uniform_em = region_element->create_local_matrix(fem, Eigen::VectorXd::Zero(region_element->dof_count()), true, true);
            if (uniform_em.ndof > 24) {
                throw std::runtime_error("explicit_gradient_dynamic: uniform element ndof > 24");
            }
        }

        const int region_nelem = static_cast<int>(region.elem_ids.size());
        #pragma omp parallel for schedule(static) num_threads(sens_element_threads) if(sens_element_threads > 1)
        for (int local_e = 0; local_e < region_nelem; ++local_e) {
            const int elem_id = region.elem_ids[static_cast<std::size_t>(local_e)];
            if (elem_id < elem_start || elem_id >= elem_end) continue;

            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                auto& sensitivity_element = dynamic_cast<StructuralElementSmallStrain&>(*fem.ElementObjects.at(static_cast<std::size_t>(elem_id)));
                sensitivity_element.nu_interp = region_nu;
                local_em = sensitivity_element.create_unit_local_matrix(fem, Eigen::VectorXd::Zero(sensitivity_element.dof_count()), true, true);
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
                const std::size_t ev =
                    static_cast<std::size_t>(elem_id) * static_cast<std::size_t>(nvar) +
                    static_cast<std::size_t>(val_id - 1);
                df_dx_local[ev] += dE * sum_lambda_dot_Ku + drho * sum_lambda_dot_Ma;
            }
        }
    }
    t1 = Clock::now();
    const double timing_element_loop = elapsed(t0, t1);

    t0 = Clock::now();
    std::vector<double> global_flat(df_dx_local.size(), 0.0);
    t1 = Clock::now();
    const double timing_alloc_global = elapsed(t0, t1);

    if (df_dx_local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("explicit_gradient_dynamic: MPI_Allreduce count exceeds INT_MAX");
    }

    t0 = Clock::now();
    MPI_Allreduce(df_dx_local.data(),
                  global_flat.data(),
                  static_cast<int>(df_dx_local.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  PETSC_COMM_WORLD);
    t1 = Clock::now();
    const double timing_allreduce = elapsed(t0, t1);

    t0 = Clock::now();
    Eigen::MatrixXd df_dx = unflatten(global_flat, fem.NElem, nvar);
    for (int val_id = 0; val_id < nvar; ++val_id) {
        const Eigen::VectorXd explicit_dfdx = object.dfdx(fem, analysis, xPhys, val_id);
        if (explicit_dfdx.size() == 0) continue;
        if (explicit_dfdx.size() != df_dx.rows()) {
            throw std::runtime_error("explicit_gradient_dynamic: explicit dfdx size mismatch");
        }
        df_dx.col(val_id).noalias() += explicit_dfdx;
    }
    t1 = Clock::now();
    const double timing_unflatten_add = elapsed(t0, t1);

    if (rank == 0) {
        const double timing_total = elapsed(timer_begin, Clock::now());
        std::cout << "  explicit_gradient_dynamic[s]:"
                  << " ranks=" << size
                  << " threads=" << sens_element_threads
                  << " alloc_gradient=" << timing_alloc_gradient
                  << " element_loop=" << timing_element_loop
                  << " alloc_global=" << timing_alloc_global
                  << " allreduce=" << timing_allreduce
                  << " unflatten_add=" << timing_unflatten_add
                  << " total=" << timing_total
                  << '\n';
    }
    return df_dx;
}

} // namespace

Eigen::MatrixXd PetscDynamicLinearElasticAVM::AVM(const FEM& fem,
                                                  Analysis& analysis_base,
                                                  const Eigen::MatrixXd&,
                                                  const std::vector<int>& x_label,
                                                  const Filtering& filtering,
                                                  int Iter) {
    auto& analysis = dynamic_cast<PetscDynamicLinearElastic&>(analysis_base);
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

    if (!ROM) throw std::runtime_error("PetscDynamicLinearElasticAVM requires ROM initialized");
    MemoryDebug::print("petsc_sens_begin");

    Mat K0 = analysis.K_inf;
    MemoryDebug::print("petsc_sens_after_get_K0");
    Mat Keff = analysis.K_eff;
    MemoryDebug::print("petsc_sens_after_get_Keff");
    Mat C = analysis.C;
    const bool has_damping = analysis.alpha_M != 0.0 || analysis.alpha_K != 0.0;
    if (has_damping && !C) {
        throw std::runtime_error("PetscDynamicLinearElasticAVM requires damping matrix");
    }

    Eigen::SparseMatrix<double> K0_eigen_cache;
    Eigen::SparseMatrix<double> Keff_eigen_cache;

    Vec lambda = nullptr;
    Vec lambda_new = nullptr;
    Vec mu = nullptr;
    Vec mu_new = nullptr;
    Vec eta = nullptr;
    Vec eta_new = nullptr;
    Vec rhs = nullptr;
    Vec Mlambda = nullptr;
    Vec Clambda = nullptr;

    MatCreateVecs(Keff, &lambda, &rhs);
    VecDuplicate(lambda, &lambda_new);
    VecDuplicate(lambda, &mu);
    VecDuplicate(lambda, &mu_new);
    VecDuplicate(lambda, &eta);
    VecDuplicate(lambda, &eta_new);
    VecDuplicate(lambda, &Mlambda);
    if (has_damping) VecDuplicate(lambda, &Clambda);
    MemoryDebug::print("petsc_sens_after_vec_create");

    Eigen::MatrixXd lambda_all(fem.ndof, nstep + 1);
    MemoryDebug::print("petsc_sens_after_lambda_all_alloc");

    ROM->set_petsc_factorized_solver(Keff, analysis.factorized_K_eff);

    int k = nstep;
    set_vec_from_eigen_vector(mu, dfda(fem, analysis, filtering.xPhys, k));
    set_vec_from_eigen_vector(eta, dfdv(fem, analysis, filtering.xPhys, k));
    set_vec_from_eigen_vector(rhs, dfdu(fem, analysis, filtering.xPhys, k), -1.0);
    VecAXPY(rhs, -dHdu, mu);
    VecAXPY(rhs, -dLdu, eta);
    set_vec_fixed_from_eigen_vector(rhs, analysis.SPCdofs, dfdFext(fem, analysis, filtering.xPhys, k), -1.0);
    ROM->InitializeROMContext(analysis, Iter, k + 1, k, k * analysis.dt,
                            "adjoint", "newmark_adjoint_effective", k == 0);
    solve_petsc_with_rom(*ROM, Keff, rhs, lambda, &Keff_eigen_cache);
    lambda_all.col(k) = gather_vec_to_all_local(lambda);
    MemoryDebug::print("petsc_sens_after_step_end");

    for (k = nstep - 1; k >= 0; --k) {
        VecCopy(lambda, lambda_new);
        VecCopy(mu, mu_new);
        VecCopy(eta, eta_new);
        MatMult(analysis.M, lambda_new, Mlambda);
        if (has_damping) MatMult(C, lambda_new, Clambda);

        set_vec_from_eigen_vector(mu, dfda(fem, analysis, filtering.xPhys, k));
        VecAXPY(mu, -(0.5 / beta - 1.0), Mlambda);
        if (has_damping) VecAXPY(mu, -c_mu, Clambda);
        VecAXPY(mu, dHda_new, mu_new);
        VecAXPY(mu, dLda_new, eta_new);
        set_vec_from_eigen_vector(eta, dfdv(fem, analysis, filtering.xPhys, k));
        VecAXPY(eta, -(1.0 / (beta * dt)), Mlambda);
        if (has_damping) VecAXPY(eta, -c_eta, Clambda);
        VecAXPY(eta, dHdv_new, mu_new);
        VecAXPY(eta, dLdv_new, eta_new);
        zero_fixed(mu, analysis.SPCdofs);
        zero_fixed(eta, analysis.SPCdofs);

        set_vec_from_eigen_vector(rhs, dfdu(fem, analysis, filtering.xPhys, k), -1.0);
        VecAXPY(rhs, a0c, Mlambda);
        if (has_damping) VecAXPY(rhs, a1c, Clambda);
        VecAXPY(rhs, -dHdu_new, mu_new);
        VecAXPY(rhs, -dLdu_new, eta_new);
        if (k != 0) {
            VecAXPY(rhs, -dHdu, mu);
            VecAXPY(rhs, -dLdu, eta);
        }
        set_vec_fixed_from_eigen_vector(rhs, analysis.SPCdofs, dfdFext(fem, analysis, filtering.xPhys, k), -1.0);

        if (k == 0) {
            ROM->set_petsc_factorized_solver(nullptr, nullptr);
            if (analysis.factorized_K_eff) KSPDestroy(&analysis.factorized_K_eff);

            ROM->set_petsc_factorized_solver(K0, analysis.factorized_K_initial);
            ROM->InitializeROMContext(analysis, Iter, k + 1, k, k * analysis.dt,
                                    "adjoint", "newmark_adjoint_effective", k == 0);
            solve_petsc_with_rom(*ROM, K0, rhs, lambda, &K0_eigen_cache);

            ROM->set_petsc_factorized_solver(nullptr, nullptr);
            if (analysis.factorized_K_initial) KSPDestroy(&analysis.factorized_K_initial);
        } else {
            ROM->InitializeROMContext(analysis, Iter, k + 1, k, k * analysis.dt,
                                    "adjoint", "newmark_adjoint_effective", k == 0);
            solve_petsc_with_rom(*ROM, Keff, rhs, lambda, &Keff_eigen_cache);
        }

        lambda_all.col(k) = gather_vec_to_all_local(lambda);
        if (k == nstep - 1 || k == nstep / 2 || k == 0) {
            MemoryDebug::print("petsc_sens_after_step_" + std::to_string(k));
        }
    }

    MemoryDebug::print("petsc_sens_before_explicit_gradient");
    Eigen::MatrixXd dfdx_phys = explicit_gradient_dynamic(fem, analysis, filtering.xPhys, *this, lambda_all);
    MemoryDebug::print("petsc_sens_after_explicit_gradient");

    VecDestroy(&lambda);
    VecDestroy(&lambda_new);
    VecDestroy(&mu);
    VecDestroy(&mu_new);
    VecDestroy(&eta);
    VecDestroy(&eta_new);
    VecDestroy(&rhs);
    VecDestroy(&Mlambda);
    if (Clambda) VecDestroy(&Clambda);
    MemoryDebug::print("petsc_sens_after_destroy_temp");

    Eigen::MatrixXd out = filtering.sensitivity(dfdx_phys);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}
