#include "Analysis/StaticGeomericalNonlinear/StaticGeomericalNonlinearAVM.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/Element/StructuralElement/StructuralElementTotalLagrange.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"

#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <stdexcept>

namespace {
int sens_threads() {
    const char* v = std::getenv("SENS_ELEMENT_THREADS");
    return v ? std::max(1, std::atoi(v)) : 1;
}

Eigen::MatrixXd one_term_gradient(const FEM& fem,
                                  StaticGeomericalNonlinear& A,
                                  StaticGeomericalNonlinearFunctionalBase& term,
                                  const Eigen::MatrixXd& xPhys) {
    if (!A.factorized_K_T || A.K_T.rows() != static_cast<int>(A.free.size()))
        throw std::runtime_error("AVM requires factorized converged K_T,ff");

    // Adjoint: K_T,ff^T lambda_f = df/du_f.
    // StVK Total Lagrangian elastic tangent is symmetric, so the SAME SparseLU
    // factors can be used for K_T^T in exact arithmetic.
    const Eigen::VectorXd dfdu = term.dfdu(fem, A, xPhys, A.current_step);
    if (dfdu.size() != fem.ndof) throw std::runtime_error("Nonlinear dfdu size mismatch");
    Eigen::VectorXd rhs_f(static_cast<Eigen::Index>(A.free.size()));
    for (std::size_t i = 0; i < A.free.size(); ++i)
        rhs_f[static_cast<Eigen::Index>(i)] = dfdu[A.free[i]];
    const Eigen::VectorXd lambda_f = A.factorized_K_T->solve(rhs_f);
    if (A.factorized_K_T->info() != Eigen::Success || !lambda_f.allFinite())
        throw std::runtime_error("Nonlinear AVM adjoint solve failed");
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(fem.ndof);
    for (std::size_t i = 0; i < A.free.size(); ++i)
        lambda[A.free[i]] = lambda_f[static_cast<Eigen::Index>(i)];

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(fem.NElem, xPhys.cols());
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(sens_threads()));
    const int threads = sens_threads();
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int e = 0; e < fem.NElem; ++e) {
        const int tid = omp_get_thread_num();
        auto& error = errors[static_cast<std::size_t>(tid)];
        if (error) continue;
        try {
            auto* element = dynamic_cast<StructuralElementTotalLagrange*>(
                fem.ElementObjects.at(static_cast<std::size_t>(e)).get());
            if (!element || !element->material_paras)
                throw std::runtime_error("Invalid nonlinear element in AVM");
            if (myjson::lower_ascii(element->status) != "design_val") continue;
            const auto* paras = dynamic_cast<const ElasticParas*>(element->material_paras.get());
            if (!paras || paras->val_ids.size() != 1)
                throw std::runtime_error("Nonlinear AVM requires exactly one design val_id");
            const int val_id = paras->val_ids.front();
            const int col = val_id - 1;
            if (col < 0 || col >= xPhys.cols())
                throw std::runtime_error("Nonlinear AVM val_id outside xPhys");
            const double dE = A.interp->sensitivity(
                paras->E_values, xPhys(e, col), true, val_id,
                A.interp->modulus_penalty());
            if (!(element->E_interp > 0.0))
                throw std::runtime_error("Nonlinear AVM requires E_interp > 0");

            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(e));
            Eigen::VectorXd u_e(static_cast<Eigen::Index>(dofs.size()));
            Eigen::VectorXd lambda_e(static_cast<Eigen::Index>(dofs.size()));
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                const int global = static_cast<int>(dofs[i]);
                u_e[static_cast<Eigen::Index>(i)] = A.U(global, 1);
                lambda_e[static_cast<Eigen::Index>(i)] = lambda[global];
            }

            // StVK with constant nu: S=C(E_interp,nu):GreenStrain and
            // dS/d(rho) = [dE_interp/d(rho) / E_interp] S.
            // Consequently df_int^e/d(rho) = (dE/E_interp) * f_int^e.
            // create_local_matrix integrates all Gauss points for current u_e.
            // No mass, K_T, nor extra E_interp multiplication is required.
            const ElementMatrix em = element->create_local_matrix(fem, u_e, false, false);
            grad(e, col) = -(dE / element->E_interp) * lambda_e.dot(em.f_int);
        } catch (...) {
            if (!error) error = std::current_exception();
        }
    }
    for (const auto& error : errors) if (error) std::rethrow_exception(error);

    // Explicit derivative of term w.r.t. xPhys (zero for force compliance here).
    for (int v = 0; v < xPhys.cols(); ++v) {
        const Eigen::VectorXd explicit_term = term.dfdx(fem, A, xPhys, v);
        if (explicit_term.size() == 0) continue;
        if (explicit_term.size() != fem.NElem)
            throw std::runtime_error("Nonlinear explicit dfdx size mismatch");
        grad.col(v).noalias() += explicit_term;
    }
    return grad;
}
} // namespace

EigenStaticGeomericalNonlinearAVM::EigenStaticGeomericalNonlinearAVM(
    const myjson::json& j,
    const std::vector<std::shared_ptr<Analysis>>& analyses,
    const myjson::json& analyses_json, const myjson::json& run_json)
    : StaticGeomericalNonlinearAVM(j, analyses, analyses_json, run_json) {
    if (!std::dynamic_pointer_cast<StaticGeomericalNonlinear>(analysis))
        throw std::runtime_error("Eigen nonlinear AVM requires Eigen nonlinear analysis");
}

std::vector<Eigen::MatrixXd> EigenStaticGeomericalNonlinearAVM::calculate_raw_gradients(
    const FEM& fem, StaticGeomericalNonlinear& A, const Eigen::MatrixXd& xPhys) {
    std::vector<Eigen::MatrixXd> g;
    g.reserve(terms.size());
    for (auto& term : terms) g.push_back(one_term_gradient(fem, A, *term, xPhys));
    return g;
}
Eigen::MatrixXd EigenStaticGeomericalNonlinearAVM::AVM(
    const FEM& fem, Analysis& A_base, const Eigen::MatrixXd&,
    const std::vector<int>& x_label, const Filtering& filtering, int) {
    auto& A = dynamic_cast<StaticGeomericalNonlinear&>(A_base);
    const auto raw = calculate_raw_gradients(fem, A, filtering.xPhys);
    Eigen::MatrixXd phys = Eigen::MatrixXd::Zero(fem.NElem, filtering.xPhys.cols());
    for (std::size_t i = 0; i < terms.size(); ++i) phys += terms[i]->scale * raw[i];
    Eigen::MatrixXd result = filtering.sensitivity(phys);
    for (int e = 0; e < result.rows(); ++e)
        if (x_label.at(static_cast<std::size_t>(e)) != 0) result.row(e).setZero();
    return result;
}
