#include "Analysis/Analysis.hpp"
#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/StructuralElement/StructuralElement.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"

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

int omp_threads_from_env() {
    const char* value = std::getenv("OMP_NUM_THREADS");
    if (!value || !*value) return 1;
    char* end = nullptr;
    const long n = std::strtol(value, &end, 10);
    if (end == value || n < 1) return 1;
    return static_cast<int>(n);
}

void clear_stored_petsc(PetscStaticLinearElastic& analysis) {
    if (analysis.ROM) analysis.ROM->set_petsc_factorized_solver(nullptr, nullptr);
    if (analysis.K_inf) MatDestroy(&analysis.K_inf);
    if (analysis.factorized_K_initial) KSPDestroy(&analysis.factorized_K_initial);
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

} // namespace

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
                 d_nz, nullptr, d_nz, nullptr, &K_inf);
    Mat K = K_inf;
    assemble_K(fem, y_elem, K);
    PetscTime(&t1);
    assembly_time = t1 - t0;
    if (rank == 0) std::cout << "  assembly[s]: petsc static K = " << assembly_time << "\n";

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
    ROM->InitializeROMContext(*this, Iter, 1, Nstep, 1.0, "primal", "static_equilibrium", true);
    const SolveStats stats = solve_petsc_with_rom(*ROM, K, rhs, u, &K_eigen_cache);
    PetscTime(&t1);
    solve_time = t1 - t0;
    store_factorized_solver(factorized_K_initial, ROM->get_petsc_factorized_solver(K));

    U.resize(static_cast<int>(ndof), Nstep + 1);
    F_ext.resize(static_cast<int>(ndof), Nstep + 1);
    U.setZero();
    F_ext.setZero();
    ksp_iterations.clear();
    ksp_residuals.clear();
    ksp_iterations.push_back(stats.ksp_iterations);
    ksp_residuals.push_back(stats.residual_norm);
    SPCdofs = fixed;
    free = free_dofs(static_cast<int>(ndof));
    gather_vec_to_all_into_column(u, U, Nstep);
    gather_vec_to_all_into_column(f_ext, F_ext, Nstep);
    if (rank == 0) {
        std::cout << "static solve_res = " << stats.residual_norm
                  << ", ksp_it = " << stats.ksp_iterations << "\n";
    }

    VecDestroy(&f_ext);
    VecDestroy(&rhs);
    VecDestroy(&u);
}


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

        for (const auto& region_ptr : fem.Regions) {
            const Region& region = *region_ptr;
            const auto region_element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front())));
            if (!region_element || !region_element->material_paras) {
                throw std::runtime_error("Element material is not initialized");
            }
            const double region_nu = dynamic_cast<const ElasticParas&>(*region_element->material_paras).nu_values.front();
            ElementMatrix uniform_em;
            if (fem.uniform_element_matrices) {
                if (region.elem_ids.empty()) throw std::runtime_error("petsc assemble_K: region has no elements");
                region_element->nu_interp = region_nu;
                uniform_em = region_element->create_local_matrix(fem, Eigen::VectorXd::Zero(region_element->dof_count()), true, false);
            }

            #pragma omp parallel num_threads(num_threads)
            {
                const int tid = omp_get_thread_num();
                auto& local_contribs = thread_contribs[static_cast<std::size_t>(tid)];

                #pragma omp for schedule(static)
                for (std::size_t e = 0; e < region.elem_ids.size(); ++e) {
                    const int elem_id = region.elem_ids[e];
                    const auto element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(elem_id)));
                    if (!element || !element->material_paras) throw std::runtime_error("Element material is not initialized");
                    if (!interp) throw std::runtime_error("Analysis interpolation is not initialized");
                    const auto& material = dynamic_cast<const ElasticParas&>(*element->material_paras);
                    const int val_id = material.val_ids.empty() ? 0 : material.val_ids.front();
                    element->E_interp = interp->calculate(material.E_values, y_elem(elem_id, 0),
                        myjson::lower_ascii(element->status) == "design_val", val_id, interp->modulus_penalty());
                    element->nu_interp = material.nu_values.front();
                    element->rho_interp = interp->calculate(material.rho_values, y_elem(elem_id, 0),
                        myjson::lower_ascii(element->status) == "design_val", val_id, interp->mass_density_penalty());
                    const double Eeff = element->E_interp;
                    ElementMatrix local_em;
                    const ElementMatrix* em_ptr = &uniform_em;
                    if (!fem.uniform_element_matrices) {
                        local_em = element->create_local_matrix(fem, Eigen::VectorXd::Zero(element->dof_count()), true, false);
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
                            c.ke[ij] = static_cast<PetscScalar>((fem.uniform_element_matrices ? Eeff : 1.0) * em.ke(i, j));
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
        for (const auto& region_ptr : fem.Regions) {
            const Region& region = *region_ptr;
            const auto region_element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front())));
            if (!region_element || !region_element->material_paras) {
                throw std::runtime_error("Element material is not initialized");
            }
            const double region_nu = dynamic_cast<const ElasticParas&>(*region_element->material_paras).nu_values.front();
            ElementMatrix uniform_em;
            if (fem.uniform_element_matrices) {
                if (region.elem_ids.empty()) throw std::runtime_error("petsc assemble_K: region has no elements");
                region_element->nu_interp = region_nu;
                uniform_em = region_element->create_local_matrix(fem, Eigen::VectorXd::Zero(region_element->dof_count()), true, false);
            }
            for (int elem_id : region.elem_ids) {
                if (elem_id < elem_start || elem_id >= elem_end) continue;
                const auto element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(elem_id)));
                if (!element || !element->material_paras) throw std::runtime_error("Element material is not initialized");
                if (!interp) throw std::runtime_error("Analysis interpolation is not initialized");
                const auto& material = dynamic_cast<const ElasticParas&>(*element->material_paras);
                const int val_id = material.val_ids.empty() ? 0 : material.val_ids.front();
                element->E_interp = interp->calculate(material.E_values, y_elem(elem_id, 0),
                    myjson::lower_ascii(element->status) == "design_val", val_id, interp->modulus_penalty());
                element->nu_interp = material.nu_values.front();
                element->rho_interp = interp->calculate(material.rho_values, y_elem(elem_id, 0),
                    myjson::lower_ascii(element->status) == "design_val", val_id, interp->mass_density_penalty());
                const double Eeff = element->E_interp;
                ElementMatrix local_em;
                const ElementMatrix* em_ptr = &uniform_em;
                if (!fem.uniform_element_matrices) {
                    local_em = element->create_local_matrix(fem, Eigen::VectorXd::Zero(element->dof_count()), true, false);
                    em_ptr = &local_em;
                }
                const ElementMatrix& em = *em_ptr;
                const auto& rows = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
                const LocalDenseMatrix ke = (fem.uniform_element_matrices ? Eeff : 1.0) * em.ke;
                MatSetValues(K, em.ndof, rows.data(), em.ndof, rows.data(), ke.data(), ADD_VALUES);
            }
        }
    }

    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
}
