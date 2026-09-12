#include "Analysis/DynamicLinearElastic/DynamicLinearElasticFunctional.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"

#include <stdexcept>

namespace {

double ramp_derivative(const std::vector<double>& para, double x, double p) {
    const double denom = 1.0 + p * (1.0 - x);
    const double dg = (1.0 + p) / (denom * denom);
    return (para[1] - para[0]) * dg;
}

} // namespace

void DynamicLinearElasticFunctionalBase::initialize(const FEM& fem,
                                                   DynamicLinearElastic& analysis,
                                                   const Eigen::MatrixXd& xPhys) {
    calculate(fem, analysis, xPhys);
    if (initial_value == 0.0) initial_value = value;
}

void L2ComplianceElastic::calculate(const FEM& fem,
                                    DynamicLinearElastic& analysis,
                                    const Eigen::MatrixXd& xPhys) {
    const int ndof = fem.ndof;
    const int nstep = analysis.Nstep;
    const int nvar = static_cast<int>(xPhys.cols());

    value = 0.0;
    dfdu_all = Eigen::MatrixXd::Zero(ndof, nstep + 1);
    dfdv_all.resize(0, 0);
    dfda_all.resize(0, 0);
    dfdFext_all.resize(0, 0);
    df_dx = Eigen::MatrixXd::Zero(fem.NElem, nvar);

    /*
    // Original L2 compliance objective, kept for temporary objective testing.
    for (int k = 0; k <= nstep; ++k) {
        const auto u = analysis.results.U.col(k);
        const auto f = analysis.results.F_ext.col(k);
        const double s = f.dot(u);
        value += s * s;
        dfdu_all.col(k).noalias() = 2.0 * s * f;
        dfdFext_all.col(k).noalias() = 2.0 * s * u;
        for (int d : analysis.results.SPCdofs) {
            dfdu_all(d, k) = 0.0;
        }
        for (int d : analysis.results.free) {
            dfdFext_all(d, k) = 0.0;
        }
    }
    */

    // Temporary test objective: 0.5 * sum_k U(k)^T K_initial U(k).
    // Explicit density derivative: 0.5 * sum_k Ue(k)^T dKe/dx Ue(k).
    const auto& K_initial = analysis.results.K_inf_eigen;
    if (K_initial.rows() != ndof || K_initial.cols() != ndof) {
        throw std::runtime_error("L2ComplianceElastic strain-energy test requires Eigen K_initial");
    }
    Eigen::VectorXd Ku(ndof);
    for (int k = 0; k <= nstep; ++k) {
        const auto u = analysis.results.U.col(k);
        Ku.noalias() = K_initial * u;
        value += 0.5 * u.dot(Ku);
        dfdu_all.col(k).noalias() = Ku;
        for (int d : analysis.results.SPCdofs) {
            dfdu_all(d, k) = 0.0;
        }
    }

    for (const Region& region : analysis.regions) {
        if (region.status != "design_val") continue;
        if (region.elem_ids.empty()) continue;
        if (region.mat_paras.val_ids.empty()) throw std::runtime_error("L2ComplianceElastic: no val_ids");
        if (region.mat_paras.E_inf.empty()) throw std::runtime_error("L2ComplianceElastic: no elastic parameters");

        const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            uniform_em = fem.compute_element_matrices(region.elem_ids.front(), region_nu, fem.Thickness);
        }

        for (int elem_id : region.elem_ids) {
            if (elem_id < 0 || elem_id >= fem.NElem) throw std::runtime_error("L2ComplianceElastic: invalid elem id");

            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                local_em = fem.compute_element_matrices(elem_id, region_nu, fem.Thickness);
                em_ptr = &local_em;
            }
            const ElementMatrix& em = *em_ptr;
            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            if (dofs.size() > 24) throw std::runtime_error("L2ComplianceElastic: element ndof > 24");

            Eigen::Matrix<double, 24, 1> u_elem_buffer;
            Eigen::Matrix<double, 24, 1> Ku_elem_buffer;
            double sum_u_dot_Ku = 0.0;
            for (int k = 0; k <= nstep; ++k) {
                const auto u = analysis.results.U.col(k);
                for (int i = 0; i < em.ndof; ++i) {
                    const int dof_i = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                    u_elem_buffer(i) = u(dof_i);
                }
                const auto u_elem = u_elem_buffer.head(em.ndof);
                auto Ku_elem = Ku_elem_buffer.head(em.ndof);
                Ku_elem.noalias() = em.ke * u_elem;
                sum_u_dot_Ku += u_elem.dot(Ku_elem);
            }

            for (int val_id : region.mat_paras.val_ids) {
                if (val_id <= 0 || val_id > nvar) throw std::runtime_error("L2ComplianceElastic: val_id outside xPhys");
                const double x = xPhys(elem_id, val_id - 1);
                const double dE = ramp_derivative(region.mat_paras.E_inf, x, region.interp.p_E_elastic);
                df_dx(elem_id, val_id - 1) += 0.5 * dE * sum_u_dot_Ku;
            }
        }
    }
}
