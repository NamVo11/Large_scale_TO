#include "Analysis/Analysis.hpp"

#include "ROM/ROM_method/ROM_Base.hpp"
#include "helper/MemoryDebug.hpp"
#include "helper/MyAlgebra.hpp"
#include "helper/PetscEigenBridge.hpp"
#include "helper/PetscEigenHelper.hpp"

#include <petscksp.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

struct ElementContribution {
    PetscInt ndof = 0;
    std::array<PetscInt, 24> rows{};
    std::array<PetscScalar, 24 * 24> ke{};
};

double interp_elastic(const Region& region, double x) {
    return region.interp.elastic_interp(region.mat_paras.E_inf, x);
}

int omp_threads_from_env() {
    const char* value = std::getenv("OMP_NUM_THREADS");
    if (!value || !*value) return 1;
    char* end = nullptr;
    const long n = std::strtol(value, &end, 10);
    if (end == value || n < 1) return 1;
    return static_cast<int>(n);
}

void clear_stored_petsc(Analysis& analysis) {
    if (analysis.ROM) analysis.ROM->set_petsc_factorized_solver(nullptr, nullptr);
    AnalysisResult& r = analysis.results;
    if (r.K_inf) MatDestroy(&r.K_inf);
    if (r.M) MatDestroy(&r.M);
    if (r.K_eff) MatDestroy(&r.K_eff);
    if (r.factorized_K_initial) KSPDestroy(&r.factorized_K_initial);
    if (r.factorized_K_eff) KSPDestroy(&r.factorized_K_eff);
}

void store_factorized_solver(KSP& target, KSP source) {
    if (!source) return;
    PetscObjectReference((PetscObject)source);
    if (target) KSPDestroy(&target);
    target = source;
}

void set_dirichlet_rhs(const Analysis& analysis, Vec rhs, int istep) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
        for (const auto& bc_ptr : analysis.Dirich) {
            const BoundaryCondition& bc = *bc_ptr;
            const PetscScalar value = bc.value(analysis, istep, 1.0);
            for (int dof0 : bc.dof_ids) {
                const PetscInt dof = static_cast<PetscInt>(dof0);
                VecSetValues(rhs, 1, &dof, &value, INSERT_VALUES);
            }
        }
    }
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);
}

void set_neum(const Analysis& analysis, Vec f, int istep) {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    VecSet(f, 0.0);
    if (rank == 0) {
        for (const auto& bc_ptr : analysis.Neum) {
            const BoundaryCondition& bc = *bc_ptr;
            const PetscScalar value = bc.value(analysis, istep, static_cast<double>(bc.nDof));
            for (int dof0 : bc.dof_ids) {
                const PetscInt dof = static_cast<PetscInt>(dof0);
                VecSetValues(f, 1, &dof, &value, ADD_VALUES);
            }
        }
    }
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);
}

ROMContext make_context(const Analysis& analysis, int Iter) {
    ROMContext ctx;
    ctx.analysis_id = analysis.id;
    ctx.opt_iter = Iter;
    ctx.solve_step = 1;
    ctx.time_step = analysis.Nstep;
    ctx.time = 1.0;
    ctx.role = "primal";
    ctx.system_type = "static_equilibrium";
    ctx.is_end_of_analysis = true;
    return ctx;
}

SolveStats solve_with_common_rom(ROMBase& rom,
                                 Mat lhs,
                                 Vec rhs,
                                 Vec u,
                                 const ROMContext& context,
                                 Eigen::SparseMatrix<double>* lhs_e_cache) {
    rom.set_petsc_system(lhs, rhs, u);
    if (!rom.use_mor) {
        Eigen::SparseMatrix<double> empty_lhs;
        Eigen::VectorXd empty_rhs;
        Eigen::VectorXd empty_u;
        return rom.main(empty_lhs, empty_rhs, empty_u, context);
    }

    if (lhs_e_cache->rows() == 0 || lhs_e_cache->cols() == 0) {
        *lhs_e_cache = petsc_mat_to_eigen(lhs);
    }
    const Eigen::VectorXd rhs_e = petsc_vec_to_eigen(rhs);
    Eigen::VectorXd u_e = Eigen::VectorXd::Zero(rhs_e.size());
    const SolveStats stats = rom.main(*lhs_e_cache, rhs_e, u_e, context);
    eigen_to_petsc_vec(u_e, u);
    return stats;
}

} // namespace

void PetscStaticLinearElastic::assemble_K(const FEM& fem,
                                          const Eigen::MatrixXd& y_elem,
                                          Mat K) const {
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    const int elem_start = (fem.NElem * rank) / size;
    const int elem_end = (fem.NElem * (rank + 1)) / size;

    if (size == 1) {
        const int num_threads = omp_threads_from_env();
        std::vector<std::vector<ElementContribution>> thread_contribs(static_cast<std::size_t>(num_threads));

        for (const Region& region : regions) {
            const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
            ElementMatrix uniform_em;
            if (fem.uniform_element_matrices) {
                if (region.elem_ids.empty()) throw std::runtime_error("petsc assemble_K: region has no elements");
                uniform_em = fem.compute_element_matrices(region.elem_ids.front(), region_nu, fem.Thickness);
            }

            #pragma omp parallel num_threads(num_threads)
            {
                const int tid = omp_get_thread_num();
                auto& local_contribs = thread_contribs[static_cast<std::size_t>(tid)];

                #pragma omp for schedule(static)
                for (std::size_t e = 0; e < region.elem_ids.size(); ++e) {
                    const int elem_id = region.elem_ids[e];
                    const double Eeff = interp_elastic(region, y_elem(elem_id, 0));
                    ElementMatrix local_em;
                    const ElementMatrix* em_ptr = &uniform_em;
                    if (!fem.uniform_element_matrices) {
                        local_em = fem.compute_element_matrices(elem_id, region_nu, fem.Thickness);
                        em_ptr = &local_em;
                    }
                    const ElementMatrix& em = *em_ptr;
                    const auto& rows = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
                    ElementContribution c;
                    c.ndof = static_cast<PetscInt>(em.ndof);
                    for (int i = 0; i < em.ndof; ++i) {
                        c.rows[static_cast<std::size_t>(i)] = rows[static_cast<std::size_t>(i)];
                        for (int j = 0; j < em.ndof; ++j) {
                            const std::size_t ij = static_cast<std::size_t>(i * em.ndof + j);
                            c.ke[ij] = static_cast<PetscScalar>(Eeff * em.ke(i, j));
                        }
                    }
                    local_contribs.push_back(std::move(c));
                }
            }
        }

        for (const auto& local_contribs : thread_contribs) {
            for (const ElementContribution& c : local_contribs) {
                MatSetValues(K, c.ndof, c.rows.data(), c.ndof, c.rows.data(), c.ke.data(), ADD_VALUES);
            }
        }
    } else {
        for (const Region& region : regions) {
            const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
            ElementMatrix uniform_em;
            if (fem.uniform_element_matrices) {
                if (region.elem_ids.empty()) throw std::runtime_error("petsc assemble_K: region has no elements");
                uniform_em = fem.compute_element_matrices(region.elem_ids.front(), region_nu, fem.Thickness);
            }
            for (int elem_id : region.elem_ids) {
                if (elem_id < elem_start || elem_id >= elem_end) continue;
                const double Eeff = interp_elastic(region, y_elem(elem_id, 0));
                ElementMatrix local_em;
                const ElementMatrix* em_ptr = &uniform_em;
                if (!fem.uniform_element_matrices) {
                    local_em = fem.compute_element_matrices(elem_id, region_nu, fem.Thickness);
                    em_ptr = &local_em;
                }
                const ElementMatrix& em = *em_ptr;
                const auto& rows = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
                const LocalDenseMatrix ke = Eeff * em.ke;
                MatSetValues(K, em.ndof, rows.data(), em.ndof, rows.data(), ke.data(), ADD_VALUES);
            }
        }
    }

    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
}

void PetscStaticLinearElastic::solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter) {
    if (!ROM) throw std::runtime_error("PetscStaticLinearElastic::solver requires analysis.ROM initialized");
    if (Nstep != 1) throw std::runtime_error("PetscStaticLinearElastic requires Nstep = 1");

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    clear_stored_petsc(*this);

    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    MemoryDebug::print("static_forward_begin");
    const PetscInt ndof = fem.ndof;
    const PetscInt d_nz = fem.dim == 2 ? 64 : 192;
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ndof, ndof,
                 d_nz, nullptr, d_nz, nullptr, &results.K_inf);
    Mat K = results.K_inf;
    assemble_K(fem, y_elem, K);
    PetscTime(&t1);
    results.assembly_time = t1 - t0;
    if (rank == 0) std::cout << "  assembly[s]: petsc static K = " << results.assembly_time << "\n";

    const auto fixed = SPC();
    MyAlgebra::Petsc::apply_dirichlet_to_matrix(K, fixed, 1.0);

    Vec u = nullptr;
    Vec rhs = nullptr;
    Vec f_ext = nullptr;
    MatCreateVecs(K, &u, &rhs);
    VecDuplicate(u, &f_ext);
    VecSet(u, 0.0);
    set_neum(*this, f_ext, Nstep);
    VecCopy(f_ext, rhs);
    set_dirichlet_rhs(*this, rhs, Nstep);

    Eigen::SparseMatrix<double> K_eigen_cache;
    PetscTime(&t0);
    const SolveStats stats = solve_with_common_rom(*ROM, K, rhs, u, make_context(*this, Iter), &K_eigen_cache);
    PetscTime(&t1);
    results.solve_time = t1 - t0;
    results.K_inf_eigen = K_eigen_cache;
    store_factorized_solver(results.factorized_K_initial, ROM->get_petsc_factorized_solver(K));

    results.U.resize(static_cast<int>(ndof), Nstep + 1);
    results.V.resize(0, 0);
    results.A.resize(0, 0);
    results.F_ext.resize(static_cast<int>(ndof), Nstep + 1);
    results.U.setZero();
    results.F_ext.setZero();
    results.ksp_iterations.clear();
    results.ksp_residuals.clear();
    results.ksp_iterations.push_back(stats.ksp_iterations);
    results.ksp_residuals.push_back(stats.residual_norm);
    results.SPCdofs = fixed;
    results.free = free_dofs(static_cast<int>(ndof));
    gather_vec_to_all_into_column(u, results.U, Nstep);
    gather_vec_to_all_into_column(f_ext, results.F_ext, Nstep);
    if (rank == 0) {
        std::cout << "static solve_res = " << stats.residual_norm
                  << ", ksp_it = " << stats.ksp_iterations << "\n";
    }

    VecDestroy(&f_ext);
    VecDestroy(&rhs);
    VecDestroy(&u);
}
