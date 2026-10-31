#include "Analysis/Analysis.hpp"
#include "FEM/Element/StructuralElement/StructuralElementTotalLagrange.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"
#include "helper/MyAlgebra.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <mpi.h>
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

void StaticGeomericalNonlinear::validate_input(const FEM& fem, const Eigen::MatrixXd& y_elem) const {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        int ranks = 1;
        MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
        if (ranks != 1) throw std::runtime_error("Nonlinear Eigen FOM requires one MPI rank");
    }

    if (!interp) throw std::runtime_error("Nonlinear analysis requires interpolation");
    if (Nstep < 1) throw std::invalid_argument("Nonlinear analysis requires Nstep >= 1");
    if (fem.ndof <= 0 || fem.NElem <= 0)
        throw std::invalid_argument("Nonlinear analysis: invalid FEM dimensions");
    if (y_elem.rows() != fem.NElem || y_elem.cols() < 1)
        throw std::invalid_argument("Nonlinear analysis: y_elem dimensions mismatch");
    if (fem.ElementObjects.size() != static_cast<std::size_t>(fem.NElem))
        throw std::runtime_error("Nonlinear analysis: ElementObjects size mismatch");
}
// ===========================================

std::vector<StructuralElementTotalLagrange*>
StaticGeomericalNonlinear::prepare_elements(const FEM& fem, const Eigen::MatrixXd& y_elem) const {
    std::vector<StructuralElementTotalLagrange*> elements;
    elements.reserve(static_cast<std::size_t>(fem.NElem));

    for (int e = 0; e < fem.NElem; ++e) {
        auto* element = dynamic_cast<StructuralElementTotalLagrange*>(fem.ElementObjects.at(e).get());
        if (!element)
            throw std::invalid_argument("StaticGeomericalNonlinear requires FEM.element_type = finite_strain");
        if (!element->material_paras)
            throw std::runtime_error("Element material parameters are missing");

        const auto* paras = dynamic_cast<const ElasticParas*>(element->material_paras.get());
        if (!paras)
            throw std::invalid_argument("StaticGeomericalNonlinear requires elastic material parameters");
        if (paras->nu_values.empty())
            throw std::runtime_error("Element Poisson ratio is missing");

        const bool is_design = myjson::lower_ascii(element->status) == "design_val";
        const int val_id = paras->val_ids.empty() ? 0 : paras->val_ids.front();
        const double density = y_elem(e, 0);

        element->E_interp = interp->calculate(paras->E_values, density, is_design, val_id,
                                              interp->modulus_penalty());
        element->nu_interp = paras->nu_values.front();
        element->rho_interp = interp->calculate(paras->rho_values, density, is_design, val_id,
                                                interp->mass_density_penalty());
        elements.push_back(element);
    }

    return elements;
}
// ===========================================

std::vector<int> StaticGeomericalNonlinear::make_free_index(int ndof, const std::vector<int>& free) {
    std::vector<int> free_index(static_cast<std::size_t>(ndof), -1);
    for (std::size_t i = 0; i < free.size(); ++i) free_index[free[i]] = static_cast<int>(i);
    return free_index;
}
// ===========================================

void StaticGeomericalNonlinear::set_neum(Eigen::VectorXd& f_ext, int step) const {
    f_ext.setZero();
    for (const auto& bc : Neum) {
        const double value = bc->value(*this, step, static_cast<double>(bc->nDof));
        for (int dof : bc->dof_ids) f_ext[dof] += value;
    }
}
// ===========================================

void StaticGeomericalNonlinear::set_dirichlet_u(Eigen::VectorXd& u, int step) const {
    // Newton only updates free DOFs; prescribed displacements remain fixed during a load step.
    for (const auto& bc : Dirich) {
        const double value = bc->value(*this, step, 1.0);
        for (int dof : bc->dof_ids) u[dof] = value;
    }
}
// ===========================================

int StaticGeomericalNonlinear::omp_threads_from_env() {
    const char* value = std::getenv("OMP_NUM_THREADS");
    if (!value || !*value) return 1;
    char* end = nullptr;
    const long n = std::strtol(value, &end, 10);
    if (end == value || n < 1) return 1;
    return static_cast<int>(n);
}
// ===========================================

void StaticGeomericalNonlinear::assemble_system(const FEM& fem,
                                                const std::vector<StructuralElementTotalLagrange*>& elements,
                                                const std::vector<int>& free_index, const Eigen::VectorXd& u,
                                                Eigen::VectorXd& f_int, Eigen::SparseMatrix<double>* K_ff)
{
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const int ndof = fem.ndof;
    const int nfree = static_cast<int>(free.size());
    const bool need_tangent = (K_ff != nullptr);
    const int ndofe_max = (fem.dim == 2) ? 8 : 24;
    const std::size_t reserve_nnz = static_cast<std::size_t>(fem.NElem) *
                                    static_cast<std::size_t>(ndofe_max) *
                                    static_cast<std::size_t>(ndofe_max);
    const int num_threads = omp_threads_from_env();

    std::vector<std::vector<MyAlgebra::Triplet>> thread_k_triplets(static_cast<std::size_t>(num_threads));
    const std::size_t reserve_per_thread = reserve_nnz / static_cast<std::size_t>(num_threads) + 1;
    if (need_tangent)
        for (auto& triplets : thread_k_triplets) triplets.reserve(reserve_per_thread);

    std::vector<Eigen::VectorXd> thread_fint(static_cast<std::size_t>(num_threads), Eigen::VectorXd::Zero(ndof));
    std::vector<std::exception_ptr> thread_errors(static_cast<std::size_t>(num_threads));

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        auto& local_k_triplets = thread_k_triplets[static_cast<std::size_t>(tid)];
        auto& local_fint = thread_fint[static_cast<std::size_t>(tid)];
        auto& local_error = thread_errors[static_cast<std::size_t>(tid)];

        #pragma omp for schedule(static)
        for (int e = 0; e < static_cast<int>(elements.size()); ++e) {
            try {
                auto* element = elements[e];
                const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(e));
                const int ndofe = static_cast<int>(dofs.size());

                // Evaluate each trial from the last committed material state.
                element->rollback_materials();
                Eigen::VectorXd u_e(ndofe);
                for (int i = 0; i < ndofe; ++i)
                    u_e[i] = u[static_cast<int>(dofs[static_cast<std::size_t>(i)])];

                const ElementMatrix em = element->create_local_matrix(fem, u_e, need_tangent, false);
                for (int i = 0; i < em.ndof; ++i) {
                    const int gi = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                    local_fint[gi] += em.f_int[i];
                }

                if (!need_tangent) continue;
                for (int i = 0; i < em.ndof; ++i) {
                    const int gi = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                    const int row = free_index[gi];
                    if (row < 0) continue;

                    for (int j = 0; j < em.ndof; ++j) {
                        const int gj = static_cast<int>(dofs[static_cast<std::size_t>(j)]);
                        const int col = free_index[gj];
                        if (col < 0) continue;
                        local_k_triplets.emplace_back(row, col, em.ke(i, j));
                    }
                }
            } catch (...) {
                // OpenMP workers cannot propagate exceptions directly.
                if (!local_error) local_error = std::current_exception();
            }
        }
    }

    for (const auto& error : thread_errors)
        if (error) std::rethrow_exception(error);

    f_int = Eigen::VectorXd::Zero(ndof);
    for (const auto& local_fint : thread_fint) f_int += local_fint;

    if (need_tangent) {
        std::size_t total_triplets = 0;
        for (const auto& local : thread_k_triplets) total_triplets += local.size();

        std::vector<MyAlgebra::Triplet> k_triplets;
        k_triplets.reserve(total_triplets);
        for (auto& local : thread_k_triplets)
            k_triplets.insert(k_triplets.end(), std::make_move_iterator(local.begin()),
                              std::make_move_iterator(local.end()));

        K_ff->resize(nfree, nfree);
        K_ff->setFromTriplets(k_triplets.begin(), k_triplets.end());
        K_ff->makeCompressed();
    }

    assembly_time += std::chrono::duration<double>(Clock::now() - t0).count();
}
// ===========================================

StaticGeomericalNonlinear::NewtonResult
StaticGeomericalNonlinear::solve_newton_step(const FEM& fem,
                                             const std::vector<StructuralElementTotalLagrange*>& elements,
                                             const std::vector<int>& free_index, const Eigen::VectorXd& f_ext,
                                             Eigen::VectorXd& u, int step)
{
    using Clock = std::chrono::steady_clock;
    NewtonResult result;
    double initial_norm = 0.0;
    double final_norm = 0.0;
    int updates = 0;

    try {
        // With max_newton_iterations updates, evaluate the residual up to one extra time.
        for (int iter = 0; iter <= max_newton_iterations; ++iter) {
            Eigen::VectorXd f_int;
            Eigen::SparseMatrix<double> K_ff;
            assemble_system(fem, elements, free_index, u, f_int, &K_ff);

            const int nfree = static_cast<int>(free.size());
            Eigen::VectorXd residual(nfree);
            for (int i = 0; i < nfree; ++i) {
                const int gi = free[i];
                residual[i] = f_ext[gi] - f_int[gi];
            }

            final_norm = residual.norm();
            if (!std::isfinite(final_norm)) throw std::runtime_error("Newton residual is not finite");
            if (iter == 0) initial_norm = final_norm;

            const double scale = std::max({1.0, f_ext.norm(), initial_norm});
            const double threshold = newton_tolerance * scale;
            if (final_norm <= threshold) {
                result.iterations = updates;
                result.residual_norm = final_norm;
                return result;
            }
            if (iter == max_newton_iterations) break;

            const auto solve_start = Clock::now();
            Eigen::SparseLU<Eigen::SparseMatrix<double>> linear_solver;
            linear_solver.compute(K_ff);
            if (linear_solver.info() != Eigen::Success)
                throw std::runtime_error("Newton tangent factorization failed");

            const Eigen::VectorXd delta_u = linear_solver.solve(residual);
            if (linear_solver.info() != Eigen::Success || !delta_u.allFinite())
                throw std::runtime_error("Newton tangent solve failed");
            solve_time += std::chrono::duration<double>(Clock::now() - solve_start).count();

            // Full Newton update; line search will be added separately.
            for (int i = 0; i < nfree; ++i) u[free[i]] += delta_u[i];
            ++updates;
        }

        throw std::runtime_error("Nonlinear Newton failed at load step " + std::to_string(step) +
                                 ", iterations=" + std::to_string(updates) +
                                 ", residual=" + std::to_string(final_norm));
    } catch (...) {
        for (auto* element : elements) element->rollback_materials();
        throw;
    }
}
// ===========================================

void StaticGeomericalNonlinear::solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter) {
    (void)Iter;
    validate_input(fem, y_elem);

    assembly_time = 0.0;
    solve_time = 0.0;
    newton_iterations.clear();
    equilibrium_residuals.clear();
    // SparseLU is direct: Newton iterations are not KSP iterations.
    ksp_iterations.clear();
    ksp_residuals.clear();

    // FEM owns the elements; the typed vector only avoids repeated casts.
    const std::vector<StructuralElementTotalLagrange*> elements = prepare_elements(fem, y_elem);
    const int ndof = fem.ndof;
    SPCdofs = SPC();
    free = free_dofs(ndof);
    const std::vector<int> free_index = make_free_index(ndof, free);

    U = Eigen::MatrixXd::Zero(ndof, Nstep + 1);
    F_ext = Eigen::MatrixXd::Zero(ndof, Nstep + 1);
    Eigen::VectorXd u = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(ndof);

    for (int step = 1; step <= Nstep; ++step) {
        set_neum(f_ext, step);
        set_dirichlet_u(u, step);
        const NewtonResult stats = solve_newton_step(fem, elements, free_index, f_ext, u, step);

        // Commit only after convergence; keep the converged state for the next load step.
        for (auto* element : elements) element->commit_materials();
        U.col(step) = u;
        F_ext.col(step) = f_ext;
        newton_iterations.push_back(stats.iterations);
        equilibrium_residuals.push_back(stats.residual_norm);
        std::cout << "nonlinear step=" << step << ", newton=" << stats.iterations
                  << ", residual=" << stats.residual_norm << '\n';
    }

    std::cout << "nonlinear assembly[s] = " << assembly_time
              << ", solve[s] = " << solve_time << '\n';
}
// ===========================================
