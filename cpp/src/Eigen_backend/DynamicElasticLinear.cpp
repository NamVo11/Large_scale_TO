#include "Eigen_backend/DynamicElasticLinear.hpp"

#include "Eigen_backend/ROM/ROM_method/ROM_Base.hpp"
#include "FEMAssembly.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <petscsys.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace eigen_backend {
namespace {

using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

std::vector<int> fixed_dofs(const Analysis& analysis) {
    std::vector<int> dofs;
    for (const auto& bc : analysis.Dirich) dofs.insert(dofs.end(), bc.dof_ids.begin(), bc.dof_ids.end());
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    return dofs;
}

std::vector<int> free_dofs(int ndof, const std::vector<int>& fixed) {
    std::vector<char> is_fixed(static_cast<std::size_t>(ndof), 0);
    for (int d : fixed) {
        if (d >= 0 && d < ndof) is_fixed[static_cast<std::size_t>(d)] = 1;
    }
    std::vector<int> free;
    free.reserve(static_cast<std::size_t>(ndof) - fixed.size());
    for (int d = 0; d < ndof; ++d) {
        if (!is_fixed[static_cast<std::size_t>(d)]) free.push_back(d);
    }
    return free;
}

std::pair<double, double> interp_region(const Region& region, double x) {
    const double Eeff = region.interp.elastic_interp
        ? region.interp.elastic_interp(region.mat_paras.E_inf, x)
        : RAMP(region.mat_paras.E_inf, x, region.interp.p_E_elastic);
    const double rhoeff = region.interp.mass_density_interp
        ? region.interp.mass_density_interp(region.mat_paras.rho, x)
        : RAMP(region.mat_paras.rho, x, region.interp.p_rho);
    return {Eeff, rhoeff};
}

void assemble_K_M(const FEM& fem, const Analysis& analysis, const Eigen::MatrixXd& y_elem, Sparse& K, Sparse& M) {
    std::vector<Triplet> k_triplets;
    std::vector<Triplet> m_triplets;
    const int ndofe_max = fem.dim == 2 ? 8 : 24;
    const std::size_t reserve_nnz = static_cast<std::size_t>(fem.NElem * ndofe_max * ndofe_max);
    k_triplets.reserve(reserve_nnz);
    m_triplets.reserve(reserve_nnz);

    for (const Region& region : analysis.regions) {
        const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            if (region.elem_ids.empty()) throw std::runtime_error("eigen assemble_K_M: region has no elements");
            uniform_em = compute_element_matrices(fem, region.elem_ids.front(), region_nu, fem.Thickness);
        }
        for (int elem_id : region.elem_ids) {
            const double x = y_elem(elem_id, 0);
            const auto [Eeff, rhoeff] = interp_region(region, x);
            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                local_em = compute_element_matrices(fem, elem_id, region_nu, fem.Thickness);
                em_ptr = &local_em;
            }
            const ElementMatrix& em = *em_ptr;
            const auto& rows = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            for (int i = 0; i < em.ndof; ++i) {
                const int row = static_cast<int>(rows[static_cast<std::size_t>(i)]);
                for (int j = 0; j < em.ndof; ++j) {
                    const int col = static_cast<int>(rows[static_cast<std::size_t>(j)]);
                    k_triplets.emplace_back(row, col, Eeff * em.ke(i, j));
                    m_triplets.emplace_back(row, col, rhoeff * em.me(i, j));
                }
            }
        }
    }
    K.resize(fem.ndof, fem.ndof);
    M.resize(fem.ndof, fem.ndof);
    K.setFromTriplets(k_triplets.begin(), k_triplets.end());
    M.setFromTriplets(m_triplets.begin(), m_triplets.end());
    K.makeCompressed();
    M.makeCompressed();
}

void apply_dirichlet_to_matrix(Sparse& A, const std::vector<int>& fixed, double diag) {
    if (fixed.empty()) return;
    A.makeCompressed();
    for (int d : fixed) {
        for (Sparse::InnerIterator it(A, d); it; ++it) {
            if (it.row() != d) it.valueRef() = 0.0;
        }
    }
    for (int col = 0; col < A.outerSize(); ++col) {
        for (Sparse::InnerIterator it(A, col); it; ++it) {
            if (it.row() != col && std::binary_search(fixed.begin(), fixed.end(), static_cast<int>(it.row()))) {
                it.valueRef() = 0.0;
            }
        }
    }
    for (int d : fixed) A.coeffRef(d, d) = diag;
    A.prune(0.0);
    A.makeCompressed();
}

void set_dirich_zero(Eigen::VectorXd& rhs, const std::vector<int>& fixed) {
    for (int d : fixed) rhs(d) = 0.0;
}

void set_neum(const Analysis& analysis, Eigen::VectorXd& f, int istep) {
    f.setZero();
    for (const auto& bc : analysis.Neum) {
        const double value = bc.fun_value ? bc.fun_value(istep) : bc_value(bc, analysis, istep);
        for (int dof : bc.dof_ids) f(dof) += value;
    }
}

ROMContext make_context(const Analysis& analysis,
                        int Iter,
                        int solve_step,
                        int time_step,
                        const std::string& system_type,
                        bool is_end) {
    ROMContext ctx;
    ctx.analysis_id = analysis.id;
    ctx.opt_iter = Iter;
    ctx.solve_step = solve_step;
    ctx.time_step = time_step;
    ctx.time = time_step * analysis.dt;
    ctx.role = "primal";
    ctx.system_type = system_type;
    ctx.is_end_of_analysis = is_end;
    return ctx;
}

} // namespace

void dynamic_elastic_linear(Analysis& analysis, const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter) {
    (void)Iter;
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size != 1) throw std::runtime_error("eigen_backend::dynamic_elastic_linear requires mpi_size=1");

    const int ndof = fem.ndof;
    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    Sparse K;
    Sparse M;
    assemble_K_M(fem, analysis, y_elem, K, M);
    PetscTime(&t1);
    analysis.results.assembly_time = static_cast<double>(t1 - t0);
    if (rank == 0) std::cout << "  assembly[s]: eigen K/M = " << analysis.results.assembly_time << "\n";

    const std::vector<int> fixed = fixed_dofs(analysis);
    Sparse K_initial = K;
    apply_dirichlet_to_matrix(K_initial, fixed, 1.0);
    analysis.results.factorized_K_initial_eigen.reset();
    analysis.results.factorized_K_eff_eigen.reset();

    Eigen::VectorXd u = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd u_pred = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd v_pred = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(ndof);

    if (!analysis.backend_ROM) analysis.backend_ROM = make_eigen_rom_from_analysis(analysis);
    std::shared_ptr<ROMBase>& rom = analysis.backend_ROM;
    rom->set_factorized_solver(K_initial, nullptr);

    set_neum(analysis, f_ext, 0);
    rhs = f_ext;
    set_dirich_zero(rhs, fixed);
    const SolveStats initial = rom->main(
        K_initial, rhs, u, make_context(analysis, Iter, 1, 0, "initial_equilibrium", analysis.Nstep == 0));
    if (initial.factorization_time > 0.0) {
        analysis.results.factorized_K_initial_eigen = rom->get_factorized_solver(K_initial);
    }
    if (rank == 0) std::cout << "step   0: solve_res = " << initial.residual_norm << "\n";

    const double beta = 0.25;
    const double gamma = 0.5;
    const double a0c = 1.0 / (beta * analysis.dt * analysis.dt);
    Sparse K_eff = K + a0c * M;
    apply_dirichlet_to_matrix(K_eff, fixed, 1.0);
    analysis.results.K_inf_eigen = K_initial;
    analysis.results.M_eigen = M;
    analysis.results.K_eff_eigen = K_eff;

    const int nstore = analysis.Nstep + 1;
    analysis.results.U.resize(ndof, nstore);
    analysis.results.V.resize(ndof, nstore);
    analysis.results.A.resize(ndof, nstore);
    analysis.results.F_ext.resize(ndof, nstore);
    analysis.results.ksp_iterations.clear();
    analysis.results.ksp_residuals.clear();
    analysis.results.SPCdofs = fixed;
    analysis.results.free = free_dofs(ndof, fixed);
    analysis.results.U.col(0) = u;
    analysis.results.V.col(0) = v;
    analysis.results.A.col(0) = acc;
    analysis.results.F_ext.col(0) = f_ext;

    PetscTime(&t0);
    for (int istep = 1; istep <= analysis.Nstep; ++istep) {
        u_pred = u + analysis.dt * v + analysis.dt * analysis.dt * (0.5 - beta) * acc;
        v_pred = v + analysis.dt * (1.0 - gamma) * acc;

        set_neum(analysis, f_ext, istep);
        rhs.noalias() = f_ext + a0c * (M * u_pred);
        set_dirich_zero(rhs, fixed);

        const SolveStats step = rom->main(
            K_eff, rhs, u, make_context(analysis, Iter, istep + 1, istep, "newmark_effective", istep == analysis.Nstep));
        if (!analysis.results.factorized_K_eff_eigen && step.factorization_time > 0.0) {
            analysis.results.factorized_K_eff_eigen = rom->get_factorized_solver(K_eff);
        }
        analysis.results.ksp_iterations.push_back(step.ksp_iterations);
        analysis.results.ksp_residuals.push_back(step.residual_norm);

        acc = a0c * (u - u_pred);
        v = v_pred + gamma * analysis.dt * acc;
/*
        if (rank == 0) {
            std::cout << "step " << istep << ": solve_res = "
                      << step.residual_norm << ", ksp_it = " << step.ksp_iterations
                      << ", ksp_res = " << step.residual_norm << "\n";
        }
*/
        analysis.results.U.col(istep) = u;
        analysis.results.V.col(istep) = v;
        analysis.results.A.col(istep) = acc;
        analysis.results.F_ext.col(istep) = f_ext;
    }
    PetscTime(&t1);
    analysis.results.solve_time = static_cast<double>(t1 - t0);
}

} // namespace eigen_backend
