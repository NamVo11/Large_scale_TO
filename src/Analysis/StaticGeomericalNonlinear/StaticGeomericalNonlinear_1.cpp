#include "Analysis/Analysis.hpp"
#include "FEM/Element/StructuralElement/StructuralElementTotalLagrange.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"

#include <Eigen/SparseLU>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Triplet = Eigen::Triplet<double>;
using Sparse = Eigen::SparseMatrix<double>;

int omp_threads_from_env() {
    const char* v = std::getenv("OMP_NUM_THREADS");
    if (!v || !*v) return 1;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    return end == v || n < 1 ? 1 : static_cast<int>(n);
}

StructuralElementTotalLagrange& nl_element(const FEM& fem, int e) {
    // No persistent typed-pointer vector: FEM owns each element.
    auto* p = dynamic_cast<StructuralElementTotalLagrange*>(
        fem.ElementObjects.at(static_cast<std::size_t>(e)).get());
    if (!p) throw std::runtime_error(
        "StaticGeomericalNonlinear: expected finite_strain element at index " + std::to_string(e));
    return *p;
}

void prepare_elements(StaticGeomericalNonlinear& A, const FEM& fem,
                      const Eigen::MatrixXd& xPhys) {
    if (!A.interp) throw std::runtime_error("Nonlinear analysis requires interpolation");
    if (xPhys.rows() != fem.NElem || xPhys.cols() < 1)
        throw std::invalid_argument("Nonlinear xPhys dimensions mismatch");

    for (int e = 0; e < fem.NElem; ++e) {
        auto& element = nl_element(fem, e);
        auto* paras = element.material_paras
                          ? dynamic_cast<const ElasticParas*>(element.material_paras.get())
                          : nullptr;
        if (!paras || paras->E_values.empty() || paras->nu_values.empty() || paras->rho_values.empty())
            throw std::runtime_error("Nonlinear element requires complete ElasticParas");
        if (paras->val_ids.size() > 1)
            throw std::runtime_error("Initial nonlinear implementation supports one material val_id per element");
        const int val_id = paras->val_ids.empty() ? 0 : paras->val_ids.front();
        const int col = val_id > 0 ? val_id - 1 : 0;
        if (col < 0 || col >= xPhys.cols())
            throw std::runtime_error("Nonlinear element val_id is outside xPhys columns");
        const bool is_design = myjson::lower_ascii(element.status) == "design_val";
        const double x = xPhys(e, col);
        element.E_interp = A.interp->calculate(paras->E_values, x, is_design,
                                               val_id, A.interp->modulus_penalty());
        element.nu_interp = paras->nu_values.front();
        element.rho_interp = A.interp->calculate(paras->rho_values, x, is_design,
                                                 val_id, A.interp->mass_density_penalty());
    }
}

void apply_loads(const StaticGeomericalNonlinear& A, int step,
                 Eigen::VectorXd& u, Eigen::VectorXd& f_ext) {
    f_ext.setZero();
    for (const auto& bc : A.Neum) {
        if (bc->nDof <= 0) throw std::runtime_error("Neumann BC has no DOFs");
        const double v = bc->value(A, step, static_cast<double>(bc->nDof));
        for (int d : bc->dof_ids) f_ext[d] += v;
    }
    // Prescribed u_c is imposed BEFORE Newton and stays fixed through its iterations.
    for (const auto& bc : A.Dirich) {
        const double v = bc->value(A, step, 1.0);
        for (int d : bc->dof_ids) u[d] = v;
    }
}

// Equivalent to EigenStaticLinearElastic::assemble_K() with two thread-local
// contributions: K_T,ff triplets and full f_int. No mass / uniform-element branch.
void assemble_system(StaticGeomericalNonlinear& A, const FEM& fem,
                     const Eigen::VectorXd& u, Eigen::VectorXd& f_int,
                     Sparse& K_ff) {
    const auto t0 = Clock::now();
    const int num_threads = omp_threads_from_env();
    const int ndofe_max = fem.dim == 2 ? 8 : 24;
    const std::size_t reserve_nnz = static_cast<std::size_t>(fem.NElem) *
                                     static_cast<std::size_t>(ndofe_max) * ndofe_max;
    std::vector<std::vector<Triplet>> thread_k_triplets(static_cast<std::size_t>(num_threads));
    for (auto& t : thread_k_triplets)
        t.reserve(reserve_nnz / static_cast<std::size_t>(num_threads) + 1);
    std::vector<Eigen::VectorXd> thread_fint(static_cast<std::size_t>(num_threads),
                                             Eigen::VectorXd::Zero(fem.ndof));
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(num_threads));

#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        auto& triplets = thread_k_triplets[static_cast<std::size_t>(tid)];
        auto& local_fint = thread_fint[static_cast<std::size_t>(tid)];
        auto& error = errors[static_cast<std::size_t>(tid)];

#pragma omp for schedule(static)
        for (int e = 0; e < fem.NElem; ++e) {
            // Exceptions may not escape an OpenMP region.
            if (error) continue;
            try {
                auto& element = nl_element(fem, e);
                element.rollback_materials();
                const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(e));
                Eigen::VectorXd u_e(static_cast<Eigen::Index>(dofs.size()));
                for (Eigen::Index i = 0; i < u_e.size(); ++i)
                    u_e[i] = u[static_cast<int>(dofs[static_cast<std::size_t>(i)])];

                const ElementMatrix em = element.create_local_matrix(fem, u_e, true, false);
                for (int i = 0; i < em.ndof; ++i) {
                    const int gi = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                    local_fint[gi] += em.f_int[i];
                    const int row = A.free_index[static_cast<std::size_t>(gi)];
                    if (row < 0) continue;
                    for (int j = 0; j < em.ndof; ++j) {
                        const int gj = static_cast<int>(dofs[static_cast<std::size_t>(j)]);
                        const int col = A.free_index[static_cast<std::size_t>(gj)];
                        if (col >= 0) triplets.emplace_back(row, col, em.ke(i, j));
                    }
                }
            } catch (...) {
                if (!error) error = std::current_exception();
            }
        }
    }
    for (const auto& err : errors) if (err) std::rethrow_exception(err);

    f_int.setZero(fem.ndof);
    for (const auto& v : thread_fint) f_int += v;
    std::size_t total = 0;
    for (const auto& t : thread_k_triplets) total += t.size();
    std::vector<Triplet> k_triplets;
    k_triplets.reserve(total);
    for (auto& t : thread_k_triplets)
        k_triplets.insert(k_triplets.end(), std::make_move_iterator(t.begin()),
                           std::make_move_iterator(t.end()));
    const int nf = static_cast<int>(A.free.size());
    K_ff.resize(nf, nf);
    K_ff.setFromTriplets(k_triplets.begin(), k_triplets.end());
    K_ff.makeCompressed();
    A.assembly_time += std::chrono::duration<double>(Clock::now() - t0).count();
}

void rollback_all(const FEM& fem) {
    for (int e = 0; e < fem.NElem; ++e) nl_element(fem, e).rollback_materials();
}
void commit_all(const FEM& fem) {
    for (int e = 0; e < fem.NElem; ++e) nl_element(fem, e).commit_materials();
}
} // anonymous namespace

void StaticGeomericalNonlinear::initialize_solver(const FEM& fem,
                                                   const Eigen::MatrixXd& xPhys, int /*Iter*/) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        int size = 1;
        MPI_Comm_size(PETSC_COMM_WORLD, &size);
        if (size != 1) throw std::runtime_error("Nonlinear Eigen FOM requires one MPI rank");
    }
    prepare_elements(*this, fem, xPhys);
    SPCdofs = SPC();
    free = free_dofs(fem.ndof);
    free_index.assign(static_cast<std::size_t>(fem.ndof), -1);
    for (std::size_t i = 0; i < free.size(); ++i)
        free_index[static_cast<std::size_t>(free[i])] = static_cast<int>(i);

    // Two columns only: 0=previous converged response, 1=current response.
    U = Eigen::MatrixXd::Zero(fem.ndof, 2);
    F_ext = Eigen::MatrixXd::Zero(fem.ndof, 2);
    current_step = 0;
    K_T.resize(0, 0);
    factorized_K_T.reset();
    assembly_time = solve_time = 0.0;
    newton_iterations.clear();
    equilibrium_residuals.clear();
    ksp_iterations.clear();
    ksp_residuals.clear();
}

void StaticGeomericalNonlinear::solver(const FEM& fem,
                                      const Eigen::MatrixXd& /*xPhys*/, int /*Iter*/) {
    if (current_step < 1 || current_step > Nstep)
        throw std::runtime_error("Set A.current_step in [1,Nstep] before A.solver()");
    if (U.rows() != fem.ndof || U.cols() != 2)
        throw std::runtime_error("A.initialize_solver() must precede A.solver()");
    // Maintain the previous equilibrium as a warm start for the new load step.
    U.col(0) = U.col(1);
    F_ext.col(0) = F_ext.col(1);
    Eigen::VectorXd u = U.col(0);
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(fem.ndof);
    apply_loads(*this, current_step, u, f_ext);
    factorized_K_T.reset(); // Invalidate old load step's factorization.
    double initial_norm = 0.0;
    double final_norm = 0.0;
    int updates = 0;
    bool converged = false;

    try {
        for (int iter = 0; iter <= max_newton_iterations; ++iter) {
            Eigen::VectorXd f_int;
            assemble_system(*this, fem, u, f_int, K_T);
            Eigen::VectorXd residual(static_cast<Eigen::Index>(free.size()));
            for (std::size_t i = 0; i < free.size(); ++i)
                residual[static_cast<Eigen::Index>(i)] = f_ext[free[i]] - f_int[free[i]];
            final_norm = residual.norm();
            if (iter == 0) initial_norm = final_norm;
            const double tolerance_scale = std::max({1.0, f_ext.norm(), initial_norm});
            if (final_norm <= newton_tolerance * tolerance_scale) {
                converged = true;
                break;
            }
            if (iter == max_newton_iterations) break;

            const auto t0 = Clock::now();
            Eigen::SparseLU<Sparse> direct;
            direct.compute(K_T);
            if (direct.info() != Eigen::Success)
                throw std::runtime_error("Nonlinear Newton K_T factorization failed");
            const Eigen::VectorXd du_f = direct.solve(residual);
            if (direct.info() != Eigen::Success || !du_f.allFinite())
                throw std::runtime_error("Nonlinear Newton K_T solve failed");
            solve_time += std::chrono::duration<double>(Clock::now() - t0).count();
            // Full Newton: update free DOFs; u_c stays fixed during iterations.
            for (std::size_t i = 0; i < free.size(); ++i)
                u[free[i]] += du_f[static_cast<Eigen::Index>(i)];
            ++updates;
        }
        if (!converged)
            throw std::runtime_error("Newton failed at step " + std::to_string(current_step) +
                                     " residual=" + std::to_string(final_norm));

        // CRITICAL: K_T is assembled at the CONVERGED displacement.
        // Factorization from the preceding correction is invalid for the adjoint.
        const auto t0 = Clock::now();
        factorized_K_T = std::make_shared<Eigen::SparseLU<Sparse>>();
        factorized_K_T->compute(K_T);
        if (factorized_K_T->info() != Eigen::Success)
            throw std::runtime_error("Converged K_T factorization for AVM failed");
        solve_time += std::chrono::duration<double>(Clock::now() - t0).count();
        commit_all(fem);
    } catch (...) {
        rollback_all(fem);
        factorized_K_T.reset();
        throw;
    }

    U.col(1) = u;
    F_ext.col(1) = f_ext;
    newton_iterations.push_back(updates);
    equilibrium_residuals.push_back(final_norm);
    std::cout << "nonlinear step=" << current_step << " newton=" << updates
              << " residual=" << final_norm << "\n";
}
