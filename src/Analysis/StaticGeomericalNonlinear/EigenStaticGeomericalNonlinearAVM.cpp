#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearAVM.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/Element/StructuralElement/StructuralElementTotalLagrange.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"

#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <stdexcept>

int EigenStaticGeomericalNonlinearAVM::sens_threads() {
    const char* v = std::getenv("SENS_ELEMENT_THREADS");
    return v ? std::max(1, std::atoi(v)) : 1;
}
// ===========================================

// Analog of explicit_gradient_static() in EigenStaticLinearElasticAVM.cpp.
// The caller supplies the CURRENT load step's adjoint vector.
Eigen::MatrixXd EigenStaticGeomericalNonlinearAVM::explicit_gradient(
    const FEM& fem,
    StaticGeomericalNonlinear& A,
    const Eigen::MatrixXd& xPhys,
    const Eigen::VectorXd& lambda) {

    Eigen::MatrixXd df_dx = Eigen::MatrixXd::Zero(fem.NElem, xPhys.cols());
    const int threads = sens_threads();
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(threads));

#pragma omp parallel for schedule(static) num_threads(threads)
    for (int e = 0; e < fem.NElem; ++e) {
        auto& error = errors[static_cast<std::size_t>(omp_get_thread_num())];
        if (error) continue;
        try {
            auto* element = dynamic_cast<StructuralElementTotalLagrange*>(
                fem.ElementObjects.at(static_cast<std::size_t>(e)).get());
            if (!element || !element->material_paras)
                throw std::runtime_error("Invalid nonlinear element in AVM");
            if (myjson::lower_ascii(element->status) != "design_val") continue;

            const auto* paras = dynamic_cast<const ElasticParas*>(element->material_paras.get());
            if (!paras || paras->val_ids.size() != 1)
                throw std::runtime_error("Nonlinear AVM currently requires one val_id per element");

            const int val_id = paras->val_ids.front();
            const int col = val_id - 1;
            if (col < 0 || col >= xPhys.cols())
                throw std::runtime_error("Nonlinear AVM: invalid val_id");

            const double dE = A.interp->sensitivity(
                paras->E_values, xPhys(e, col), true, val_id,
                A.interp->modulus_penalty());
            if (!(element->E_interp > 0.0))
                throw std::runtime_error("Nonlinear AVM: E_interp must be positive");

            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(e));
            Eigen::VectorXd u_e(static_cast<Eigen::Index>(dofs.size()));
            Eigen::VectorXd lambda_e(static_cast<Eigen::Index>(dofs.size()));
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                const int global = static_cast<int>(dofs[i]);
                u_e[static_cast<Eigen::Index>(i)] = A.U(global, 0);
                lambda_e[static_cast<Eigen::Index>(i)] = lambda[global];
            }

            // StVK, constant Poisson ratio:
            // df_int^e / d(rho_e) = (dE_interp/d(rho_e) / E_interp) * f_int^e
            // f_int^e already includes Gauss-point integration.
            const ElementMatrix em = element->create_local_matrix(fem, u_e, false, false);
            df_dx(e, col) = -(dE / element->E_interp) * lambda_e.dot(em.f_int);
        } catch (...) {
            if (!error) error = std::current_exception();
        }
    }
    for (const auto& error : errors)
        if (error) std::rethrow_exception(error);

    // Explicit density derivative of objective. For compliance with
    // design-independent external forces, this is zero.
    for (int v = 0; v < xPhys.cols(); ++v) {
        const Eigen::VectorXd explicit_dfdx = dfdx(fem, A, xPhys, v);
        if (!explicit_dfdx.size()) continue;
        if (explicit_dfdx.size() != fem.NElem)
            throw std::runtime_error("Nonlinear AVM: explicit dfdx size mismatch");
        df_dx.col(v).noalias() += explicit_dfdx;
    }
    return df_dx;
}
// ===========================================

EigenStaticGeomericalNonlinearAVM::EigenStaticGeomericalNonlinearAVM(
    const myjson::json& j,
    const std::vector<std::shared_ptr<Analysis>>& analyses,
    const myjson::json& analyses_json, const myjson::json& run_json)
    : StaticGeomericalNonlinearAVM(j, analyses, analyses_json, run_json) {
    if (!std::dynamic_pointer_cast<StaticGeomericalNonlinear>(analysis))
        throw std::runtime_error("Eigen nonlinear AVM requires nonlinear analysis");
}
// ===========================================

Eigen::MatrixXd EigenStaticGeomericalNonlinearAVM::AVM(
    const FEM& fem,
    Analysis& analysis_base,
    const Eigen::MatrixXd& /*x*/,
    const std::vector<int>& x_label,
    const Filtering& filtering,
    int Iter) {

    auto& A = dynamic_cast<StaticGeomericalNonlinear&>(analysis_base);
    if (!ROM || A.K_T.rows() != static_cast<int>(A.free.size()))
        throw std::runtime_error("AVM needs ROM solver and converged nonlinear tangent K_T");

    // 1. Adjoint RHS, following the SAME group->dfdu() interface as linear.
    // term->scale is already included in dfdu().
    const Eigen::VectorXd rhs_full = dfdu(fem, A, filtering.xPhys, A.current_step);
    if (rhs_full.size() != fem.ndof)
        throw std::runtime_error("Nonlinear AVM: dfdu size mismatch");

    const int nfree = static_cast<int>(A.free.size());
    Eigen::VectorXd rhs(nfree);
    for (int i = 0; i < nfree; ++i) rhs[i] = rhs_full[A.free[i]];

    // 2. StVK tangent is symmetric: solve the free-DOF adjoint through the shared ROM interface.
    ROM->set_factorized_solver(A.K_T, nullptr);
    ROM->InitializeROMContext(A, Iter, A.current_step, A.current_step,
                              static_cast<double>(A.current_step) / A.Nstep,
                              "adjoint", "nonlinear_adjoint", A.current_step == A.Nstep);
    Eigen::VectorXd lambda_free = Eigen::VectorXd::Zero(nfree);
    ROM->main(A.K_T, rhs, lambda_free);
    if (lambda_free.size() != nfree || !lambda_free.allFinite())
        throw std::runtime_error("Nonlinear AVM: adjoint solve failed");

    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(fem.ndof);
    for (int i = 0; i < nfree; ++i) lambda[A.free[i]] = lambda_free[i];

    // 3. Element sensitivity for ONE load step, same place as in linear AVM.
    Eigen::MatrixXd out = filtering.sensitivity(
        explicit_gradient(fem, A, filtering.xPhys, lambda));

    // 4. Preserve the existing fixed-design-variable rule.
    for (Eigen::Index e = 0; e < out.rows(); ++e)
        if (x_label.at(static_cast<std::size_t>(e)) != 0) out.row(e).setZero();
    return out;
}
// ===========================================
