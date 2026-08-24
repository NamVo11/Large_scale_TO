#include "Analysis/StaticLinearElastic/StaticLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"

#include <petscsys.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <omp.h>

namespace {

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
}

double ramp_derivative(const std::vector<double>& para, double x, double p) {
    const double denom = 1.0 + p * (1.0 - x);
    const double dg = (1.0 + p) / (denom * denom);
    return (para[1] - para[0]) * dg;
}

void zero_fixed(Eigen::VectorXd& v, const std::vector<int>& fixed) {
    for (int id : fixed) v(id) = 0.0;
}

ROMContext make_adjoint_context(const Analysis& analysis, int Iter) {
    ROMContext ctx;
    ctx.analysis_id = analysis.id;
    ctx.opt_iter = Iter;
    ctx.solve_step = 1;
    ctx.time_step = analysis.Nstep;
    ctx.time = 1.0;
    ctx.role = "adjoint";
    ctx.system_type = "static_adjoint";
    ctx.is_end_of_analysis = true;
    return ctx;
}

Eigen::MatrixXd explicit_gradient_static(const FEM& fem,
                                         const Analysis& analysis,
                                         const Eigen::MatrixXd& xPhys,
                                         const StaticLinearElasticAVM& object,
                                         const Eigen::VectorXd& lambda) {
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    int sens_element_threads = 1;
    if (const char* env = std::getenv("SENS_ELEMENT_THREADS")) {
        sens_element_threads = std::max(1, std::atoi(env));
    }
    if (size != 1) sens_element_threads = 1;

    const int nvar = nvar_of(xPhys);
    Eigen::MatrixXd df_dx = Eigen::MatrixXd::Zero(fem.NElem, nvar);
    const Eigen::VectorXd u = analysis.results.U.col(analysis.Nstep);

    for (const Region& region : analysis.regions) {
        if (region.status != "design_val") continue;
        if (region.mat_paras.val_ids.empty()) throw std::runtime_error("explicit_gradient_static: no val_ids");
        if (region.mat_paras.E_inf.empty()) throw std::runtime_error("explicit_gradient_static: no elastic parameters");

        const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            if (region.elem_ids.empty()) throw std::runtime_error("explicit_gradient_static: region has no elements");
            uniform_em = fem.compute_element_matrices(region.elem_ids.front(), region_nu, fem.Thickness);
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
            Eigen::Matrix<double, 24, 1> lambda_elem_buffer;
            Eigen::Matrix<double, 24, 1> Ku_elem_buffer;
            for (int i = 0; i < em.ndof; ++i) {
                const int dof_i = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                u_elem_buffer(i) = u(dof_i);
                lambda_elem_buffer(i) = lambda(dof_i);
            }
            const auto u_elem = u_elem_buffer.head(em.ndof);
            const auto lambda_elem = lambda_elem_buffer.head(em.ndof);
            auto Ku_elem = Ku_elem_buffer.head(em.ndof);
            Ku_elem.noalias() = em.ke * u_elem;
            const double lambda_dot_Ku = lambda_elem.dot(Ku_elem);

            for (int val_id : region.mat_paras.val_ids) {
                if (val_id <= 0 || val_id > nvar) throw std::runtime_error("explicit_gradient_static: val_id outside xPhys");
                const double x = xPhys(elem_id, val_id - 1);
                const double dE = ramp_derivative(region.mat_paras.E_inf, x, region.interp.p_E_elastic);
                df_dx(elem_id, val_id - 1) -= dE * lambda_dot_Ku;
            }
        }
    }

    if (object.d_explicit_df_dx.size() != 0) df_dx.noalias() += object.d_explicit_df_dx;
    if (rank == 0) {
        std::cout << "  explicit_gradient_static: threads=" << sens_element_threads << "\n";
    }
    return df_dx;
}

Eigen::MatrixXd static_elastic_base_full_impl(const FEM& fem,
                                              Analysis& analysis,
                                              StaticLinearElasticAVM& object,
                                              const Filtering& filtering,
                                              const FilteringState& filtering_state,
                                              int Iter) {
    if (!object.ROM) throw std::runtime_error("StaticLinearElasticAVM requires ROM initialized");
    const auto& K = analysis.results.K_inf_eigen;
    if (K.rows() == 0) throw std::runtime_error("StaticLinearElasticAVM requires Eigen forward stiffness matrix");
    if (object.d_explicit_dfdu_all.cols() <= analysis.Nstep) {
        throw std::runtime_error("StaticLinearElasticAVM: dfdu_all has insufficient columns");
    }

    Eigen::VectorXd rhs = object.d_explicit_dfdu_all.col(analysis.Nstep);
    zero_fixed(rhs, analysis.results.SPCdofs);
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(rhs.size());
    const SolveStats stats = object.ROM->main(K, rhs, lambda, make_adjoint_context(analysis, Iter));
    (void)stats;
    return filtering.sensitivity(explicit_gradient_static(fem, analysis, filtering_state.xPhys, object, lambda),
                                 filtering_state);
}

} // namespace

Eigen::MatrixXd EigenStaticLinearElasticAVM::AVM(const FEM& fem,
                                                Analysis& analysis,
                                                const Eigen::MatrixXd&,
                                                const std::vector<int>& x_label,
                                                const Filtering& filtering,
                                                const FilteringState& filtering_state,
                                                int Iter) {
    Eigen::MatrixXd out = static_elastic_base_full_impl(fem, analysis, *this, filtering, filtering_state, Iter);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}

Eigen::MatrixXd PetscStaticLinearElasticAVM::AVM(const FEM& fem,
                                                Analysis& analysis,
                                                const Eigen::MatrixXd&,
                                                const std::vector<int>& x_label,
                                                const Filtering& filtering,
                                                const FilteringState& filtering_state,
                                                int Iter) {
    Eigen::MatrixXd out = static_elastic_base_full_impl(fem, analysis, *this, filtering, filtering_state, Iter);
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}
