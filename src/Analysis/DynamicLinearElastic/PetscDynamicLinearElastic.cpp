#include "Analysis/Analysis.hpp"
#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/StructuralElement/StructuralElement.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"
#include "helper/MyAlgebra.hpp"
#include "helper/MemoryDebug.hpp"
#include "helper/PetscEigenBridge.hpp"
#include "helper/PetscEigenHelper.hpp"

#include <petscksp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

struct ElementContribution {
    PetscInt ndof = 0;
    std::array<PetscInt, 24> rows{};
    std::array<PetscScalar, 24 * 24> ke{};
    std::array<PetscScalar, 24 * 24> me{};
};

int omp_threads_from_env() {
    const char* value = std::getenv("OMP_NUM_THREADS");
    if (!value || !*value) return 1;

    char* end = nullptr;
    const long n = std::strtol(value, &end, 10);
    if (end == value || n < 1) return 1;
    return static_cast<int>(n);
}

void clear_stored_petsc(PetscDynamicLinearElastic& analysis) {
    if (analysis.ROM) analysis.ROM->set_petsc_factorized_solver(nullptr, nullptr);
    if (analysis.K_inf) MatDestroy(&analysis.K_inf);
    if (analysis.M) MatDestroy(&analysis.M);
    if (analysis.C) MatDestroy(&analysis.C);
    if (analysis.K_eff) MatDestroy(&analysis.K_eff);
    if (analysis.factorized_K_initial) KSPDestroy(&analysis.factorized_K_initial);
    if (analysis.factorized_K_eff) KSPDestroy(&analysis.factorized_K_eff);
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
                PetscInt dof = dof0;
                VecSetValues(f, 1, &dof, &value, ADD_VALUES);
            }
        }
    }
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);
}

} // namespace

void PetscDynamicLinearElastic::solver(const FEM& fem, const Eigen::MatrixXd& y_elem,int Iter)
{
    PetscDynamicLinearElastic& analysis = *this;
    if (!analysis.ROM) throw std::runtime_error("PetscDynamicLinearElastic::solver requires analysis.ROM initialized");
    if (analysis.Nstep < 0 || analysis.dt <= 0.0) throw std::runtime_error("Invalid time data");

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    const PetscInt ndof = fem.ndof;
    clear_stored_petsc(analysis);

    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscLogDouble ts0 = 0.0;
    PetscLogDouble ts1 = 0.0;
    PetscLogDouble rhs_t0 = 0.0;
    PetscLogDouble rhs_t1 = 0.0;
    double timing_bc_and_initial_matrix = 0.0;
    double timing_vec_setup = 0.0;
    double timing_initial_rhs = 0.0;
    double timing_initial_solve = 0.0;
    double timing_keff_prepare = 0.0;
    double timing_predictor = 0.0;
    double timing_rhs_build = 0.0;
    double timing_rhs_set_neum = 0.0;
    double timing_rhs_matmult_mass = 0.0;
    double timing_rhs_matmult_damping = 0.0;
    double timing_rhs_vec_ops = 0.0;
    double timing_rhs_dirich = 0.0;
    double timing_solve_wall = 0.0;
    double timing_newmark_update = 0.0;
    double timing_history_store = 0.0;
    double timing_final_matrix_store = 0.0;
    Eigen::SparseMatrix<double> K_initial_eigen_cache;
    Eigen::SparseMatrix<double> K_eff_eigen_cache;
    PetscTime(&t0);
    MemoryDebug::print("forward_begin");

    const PetscInt d_nz = fem.dim == 2 ? 64 : 192; // approximate nonzeros per row
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ndof, ndof, d_nz, nullptr, d_nz, nullptr, &analysis.K_inf);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ndof, ndof,d_nz, nullptr, d_nz, nullptr, &analysis.M);

    // Memory-saving flow: K_inf is assembled as raw K first. After K_eff is
    // formed, K_inf is converted in-place to K_initial with SPC.
    Mat K = analysis.K_inf;
    Mat M = analysis.M;
    assemble_K_M(fem, y_elem, K, M);
    MemoryDebug::print("forward_after_assemble_K_M");
    PetscTime(&t1);
    analysis.assembly_time = t1 - t0;
    if (rank == 0) {
        std::cout << "  assembly[s]: forward K/M = "
                  << analysis.assembly_time << "\n";
    }

    PetscTime(&ts0);
    const auto SPCdofs = analysis.SPC();

    const double beta = 0.25;
    const double gamma = 0.5;
    const double a0c = 1.0 / (beta * analysis.dt * analysis.dt);
    const double a1c = gamma / (beta * analysis.dt);
    const bool has_damping = analysis.alpha_M != 0.0 || analysis.alpha_K != 0.0;

    PetscTime(&ts0);
    if (has_damping) {
        MatDuplicate(M, MAT_COPY_VALUES, &analysis.C);
        MatScale(analysis.C, analysis.alpha_M);
        MatAXPY(analysis.C, analysis.alpha_K, K, DIFFERENT_NONZERO_PATTERN);
    }
    Mat C = analysis.C;

    MatDuplicate(K, MAT_COPY_VALUES, &analysis.K_eff);
    MatAXPY(analysis.K_eff, a0c, M, SAME_NONZERO_PATTERN);
    if (has_damping) MatAXPY(analysis.K_eff, a1c, C, DIFFERENT_NONZERO_PATTERN);
    Mat K_eff = analysis.K_eff;
    MemoryDebug::print("forward_after_prepare_Keff");
    PetscTime(&ts1);
    timing_keff_prepare += static_cast<double>(ts1 - ts0);

    //MyAlgebra::Petsc::check_matrix_diagonal(K,"K");

    PetscTime(&ts0);
    // From here, K is K_initial with SPC, not raw K.
    MyAlgebra::Petsc::apply_dirichlet_to_matrix(K, SPCdofs, 1.0);
    MyAlgebra::Petsc::apply_dirichlet_to_matrix(K_eff, SPCdofs, 1.0);
    MemoryDebug::print("forward_after_dirichlet_K_initial");
    PetscTime(&ts1);
    timing_bc_and_initial_matrix += static_cast<double>(ts1 - ts0);

    PetscTime(&ts0);
    Vec u = nullptr;
    Vec v = nullptr;
    Vec acc = nullptr;
    Vec u_pred = nullptr;
    Vec v_pred = nullptr;
    Vec rhs = nullptr;
    Vec f_ext = nullptr;
    Vec tmp = nullptr;
    Vec damping_predictor = nullptr;
    Vec tmp_C = nullptr;
    MatCreateVecs(K, &u, &rhs);
    VecDuplicate(u, &v);
    VecDuplicate(u, &acc);
    VecDuplicate(u, &u_pred);
    VecDuplicate(u, &v_pred);
    VecDuplicate(u, &f_ext);
    VecDuplicate(u, &tmp);
    if (has_damping) {
        VecDuplicate(u, &damping_predictor);
        VecDuplicate(u, &tmp_C);
    }
    MemoryDebug::print("forward_after_vec_create");
    VecSet(u, 0.0);
    VecSet(v, 0.0);
    VecSet(acc, 0.0);
    PetscTime(&ts1);
    timing_vec_setup += static_cast<double>(ts1 - ts0);

    PetscTime(&ts0);
    set_neum(analysis, f_ext, 0);
    VecCopy(f_ext, rhs);
    set_dirichlet_rhs(analysis, rhs, 0);
    PetscTime(&ts1);
    timing_initial_rhs += static_cast<double>(ts1 - ts0);

    PetscTime(&ts0);
    analysis.ROM->InitializeROMContext(analysis, Iter, 1, 0, 0.0, "primal", 
                                        "initial_equilibrium", analysis.Nstep == 0);
    
    const SolveStats initial_stats = solve_petsc_with_rom(  *analysis.ROM,
                                                            K,
                                                            rhs,
                                                            u,
                                                            &K_initial_eigen_cache);
    PetscTime(&ts1);
    timing_initial_solve += static_cast<double>(ts1 - ts0);
    timing_solve_wall += timing_initial_solve;

    if (analysis.save_factorized_K_initial) {
        store_factorized_solver(analysis.factorized_K_initial, analysis.ROM->get_petsc_factorized_solver(K));
    } else {
        analysis.ROM->set_petsc_factorized_solver(nullptr, nullptr);
        if (analysis.factorized_K_initial) KSPDestroy(&analysis.factorized_K_initial);
    }

    MemoryDebug::print("forward_after_initial_solve_store_factor");

    const double initial_residual = initial_stats.residual_norm;
    if (rank == 0) {
        std::cout << "step   0: solve_res = " << initial_residual << "\n";
    }

    const std::size_t nstore = static_cast<std::size_t>(analysis.Nstep + 1);
    analysis.U.resize(ndof, static_cast<int>(nstore));
    analysis.V.resize(has_damping ? ndof : 0,
                      has_damping ? static_cast<int>(nstore) : 0);
    analysis.A.resize(ndof, static_cast<int>(nstore));
    analysis.F_ext.resize(ndof, static_cast<int>(nstore));
    MemoryDebug::print("forward_after_history_alloc");
    analysis.ksp_iterations.clear();
    analysis.ksp_residuals.clear();
    analysis.SPCdofs = SPCdofs;
    analysis.free = analysis.free_dofs(static_cast<int>(ndof));
    PetscTime(&ts0);
    gather_vec_to_all_into_column(u, analysis.U, 0);
    if (has_damping) gather_vec_to_all_into_column(v, analysis.V, 0);
    gather_vec_to_all_into_column(acc, analysis.A, 0);
    gather_vec_to_all_into_column(f_ext, analysis.F_ext, 0);
    PetscTime(&ts1);
    timing_history_store += static_cast<double>(ts1 - ts0);

    PetscTime(&t0);
    for (int istep = 1; istep <= analysis.Nstep; ++istep) {
        //if (rank == 0) std::cout << "step " << istep << "/" << analysis.Nstep << "\n";

        PetscTime(&ts0);
        VecCopy(u, u_pred);
        VecAXPY(u_pred, analysis.dt, v);
        VecAXPY(u_pred, analysis.dt * analysis.dt * (0.5 - beta), acc);

        VecCopy(v, v_pred);
        VecAXPY(v_pred, analysis.dt * (1.0 - gamma), acc);
        PetscTime(&ts1);
        timing_predictor += static_cast<double>(ts1 - ts0);

        PetscTime(&rhs_t0);

        PetscTime(&ts0);
        set_neum(analysis, f_ext, istep);
        PetscTime(&ts1);
        timing_rhs_set_neum += static_cast<double>(ts1 - ts0);

        PetscTime(&ts0);
        MatMult(M, u_pred, tmp);
        PetscTime(&ts1);
        timing_rhs_matmult_mass += static_cast<double>(ts1 - ts0);

        if (has_damping) {
            PetscTime(&ts0);
            VecCopy(u_pred, damping_predictor);
            VecScale(damping_predictor, a1c);
            VecAXPY(damping_predictor, -1.0, v_pred);
            MatMult(C, damping_predictor, tmp_C);
            PetscTime(&ts1);
            timing_rhs_matmult_damping += static_cast<double>(ts1 - ts0);
        }

        PetscTime(&ts0);
        VecCopy(f_ext, rhs);
        VecAXPY(rhs, a0c, tmp);
        if (has_damping) VecAXPY(rhs, 1.0, tmp_C);
        PetscTime(&ts1);
        timing_rhs_vec_ops += static_cast<double>(ts1 - ts0);

        PetscTime(&ts0);
        set_dirichlet_rhs(analysis, rhs, istep);
        PetscTime(&ts1);
        timing_rhs_dirich += static_cast<double>(ts1 - ts0);

        PetscTime(&rhs_t1);
        timing_rhs_build += static_cast<double>(rhs_t1 - rhs_t0);

        PetscTime(&ts0);

        analysis.ROM->InitializeROMContext( analysis, Iter, istep + 1, istep, istep * analysis.dt,
                                            "primal", "newmark_effective", istep == analysis.Nstep);

        const SolveStats stats = solve_petsc_with_rom(*analysis.ROM, K_eff, rhs, u,
                                                     &K_eff_eigen_cache);
                                            
        PetscTime(&ts1);
        timing_solve_wall += static_cast<double>(ts1 - ts0);

        if (!analysis.factorized_K_eff) {
            store_factorized_solver(analysis.factorized_K_eff,
                                    analysis.ROM->get_petsc_factorized_solver(K_eff));
        }

        if (istep == 1 || istep == analysis.Nstep) {
            MemoryDebug::print("forward_after_step_" + std::to_string(istep));
        }
      
        analysis.ksp_iterations.push_back(stats.ksp_iterations);
        analysis.ksp_residuals.push_back(stats.residual_norm);
        const double step_residual = stats.residual_norm;

        PetscTime(&ts0);
        VecCopy(u, tmp);
        VecAXPY(tmp, -1.0, u_pred);
        VecCopy(tmp, acc);
        VecScale(acc, a0c);

        VecCopy(v_pred, v);
        VecAXPY(v, gamma * analysis.dt, acc);
        PetscTime(&ts1);
        timing_newmark_update += static_cast<double>(ts1 - ts0);

        if (rank == 0) {
            std::cout << "step " << istep << ": solve_res = "
                      << step_residual
                      << ", ksp_it = " << stats.ksp_iterations
                      << ", ksp_res = " << stats.residual_norm << "\n";
        }

        const std::size_t store_id = static_cast<std::size_t>(istep);
        PetscTime(&ts0);
        gather_vec_to_all_into_column(u, analysis.U, static_cast<int>(store_id));
        if (has_damping) gather_vec_to_all_into_column(v, analysis.V, static_cast<int>(store_id));
        gather_vec_to_all_into_column(acc, analysis.A, static_cast<int>(store_id));
        gather_vec_to_all_into_column(f_ext, analysis.F_ext, static_cast<int>(store_id));
        PetscTime(&ts1);
        timing_history_store += static_cast<double>(ts1 - ts0);
    }
    PetscTime(&t1);
    analysis.solve_time = t1 - t0;

    MemoryDebug::print("forward_after_store_matrices");

    if (rank == 0) {
        std::cout << "  forward breakdown[s]:"
                  << " bc_initial_matrix=" << timing_bc_and_initial_matrix
                  << " vec_setup=" << timing_vec_setup
                  << " initial_rhs=" << timing_initial_rhs
                  << " initial_solve=" << timing_initial_solve
                  << " keff_prepare=" << timing_keff_prepare
                  << " predictor=" << timing_predictor
                  << " rhs_build=" << timing_rhs_build
                  << " rhs_set_neum=" << timing_rhs_set_neum
                  << " rhs_matmult_mass=" << timing_rhs_matmult_mass
                  << " rhs_matmult_damping=" << timing_rhs_matmult_damping
                  << " rhs_vec_ops=" << timing_rhs_vec_ops
                  << " rhs_dirich=" << timing_rhs_dirich
                  << " solve_wall=" << timing_solve_wall
                  << " newmark_update=" << timing_newmark_update
                  << " history_store=" << timing_history_store
                  << " final_matrix_store=" << timing_final_matrix_store
                  << "\n";
    }

    VecDestroy(&u);
    VecDestroy(&v);
    VecDestroy(&acc);
    VecDestroy(&u_pred);
    VecDestroy(&v_pred);
    VecDestroy(&rhs);
    VecDestroy(&f_ext);
    VecDestroy(&tmp);
    if (damping_predictor) VecDestroy(&damping_predictor);
    if (tmp_C) VecDestroy(&tmp_C);
    // K, M, C and K_eff are owned by analysis and reused by sensitivity.
    MemoryDebug::print("forward_after_destroy_temp");
}


void PetscDynamicLinearElastic::assemble_K_M(const FEM& fem,
                                             const Eigen::MatrixXd& y_elem,
                                             Mat K,
                                             Mat M) const {
    const Analysis& analysis = *this;
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
                if (region.elem_ids.empty()) throw std::runtime_error("assemble_K_M: region has no elements");
                uniform_em = fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front()))->compute_element_matrices(fem, region_nu, fem.thickness);
            }

                #pragma omp parallel num_threads(num_threads)
            {
                const int tid = omp_get_thread_num();
                auto& local_contribs = thread_contribs[static_cast<std::size_t>(tid)];

                #pragma omp for schedule(static)
                
                for (std::size_t e = 0; e < region.elem_ids.size(); ++e) {
                    const int elem_id = region.elem_ids[e];
                    const double x = y_elem(elem_id, 0);
                    const auto element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(elem_id)));
                    if (!element || !element->material_paras) throw std::runtime_error("Element material is not initialized");
                    if (!analysis.interp) throw std::runtime_error("Analysis interpolation is not initialized");
                    const auto& material = dynamic_cast<const ElasticParas&>(*element->material_paras);
                    const bool is_design = myjson::lower_ascii(element->status) == "design_val";
                    const int val_id = material.val_ids.empty() ? 0 : material.val_ids.front();
                    element->E_interp = analysis.interp->calculate(material.E_values, x, is_design,
                        val_id, analysis.interp->modulus_penalty());
                    element->nu_interp = material.nu_values.front();
                    element->rho_interp = analysis.interp->calculate(material.rho_values, x, is_design,
                        val_id, analysis.interp->mass_density_penalty());
                    const double Eeff = element->E_interp;
                    const double rhoeff = element->rho_interp;

                    ElementMatrix local_em;
                    const ElementMatrix* em_ptr = &uniform_em;
                    if (!fem.uniform_element_matrices) {
                        local_em = fem.ElementObjects.at(static_cast<std::size_t>(elem_id))->compute_element_matrices(fem, region_nu, fem.thickness);
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
                            c.me[ij] = static_cast<PetscScalar>(rhoeff * em.me(i, j));
                        }
                    }
                    local_contribs.push_back(std::move(c));
                }
            }
        }

        for (const auto& local_contribs : thread_contribs) {
            for (const ElementContribution& c : local_contribs) {
                MatSetValues(K, c.ndof, c.rows.data(), c.ndof, c.rows.data(), c.ke.data(), ADD_VALUES);
                MatSetValues(M, c.ndof, c.rows.data(), c.ndof, c.rows.data(), c.me.data(), ADD_VALUES);
            }
        }

        MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
        return;
    }

    for (const auto& region_ptr : fem.Regions) {
        const Region& region = *region_ptr;
        const auto region_element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front())));
        if (!region_element || !region_element->material_paras) {
            throw std::runtime_error("Element material is not initialized");
        }
        const double region_nu = dynamic_cast<const ElasticParas&>(*region_element->material_paras).nu_values.front();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            if (region.elem_ids.empty()) throw std::runtime_error("assemble_K_M: region has no elements");
            uniform_em = fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front()))->compute_element_matrices(fem, region_nu, fem.thickness);
        }

        for (int elem_id : region.elem_ids) {
            if (elem_id < elem_start || elem_id >= elem_end) continue;
            const double x = y_elem(elem_id, 0);
            const auto element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(elem_id)));
            if (!element || !element->material_paras) throw std::runtime_error("Element material is not initialized");
            if (!analysis.interp) throw std::runtime_error("Analysis interpolation is not initialized");
            const auto& material = dynamic_cast<const ElasticParas&>(*element->material_paras);
            const bool is_design = myjson::lower_ascii(element->status) == "design_val";
            const int val_id = material.val_ids.empty() ? 0 : material.val_ids.front();
            element->E_interp = analysis.interp->calculate(material.E_values, x, is_design,
                val_id, analysis.interp->modulus_penalty());
            element->nu_interp = material.nu_values.front();
            element->rho_interp = analysis.interp->calculate(material.rho_values, x, is_design,
                val_id, analysis.interp->mass_density_penalty());
            const double Eeff = element->E_interp;
            const double rhoeff = element->rho_interp;
            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                local_em = fem.ElementObjects.at(static_cast<std::size_t>(elem_id))->compute_element_matrices(fem, region_nu, fem.thickness);
                em_ptr = &local_em;
            }
            const ElementMatrix& em = *em_ptr;
            const auto& rows = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            const LocalDenseMatrix ke = Eeff * em.ke;
            const LocalDenseMatrix me = rhoeff * em.me;
            MatSetValues(K, em.ndof, rows.data(), em.ndof, rows.data(), ke.data(), ADD_VALUES);
            MatSetValues(M, em.ndof, rows.data(), em.ndof, rows.data(), me.data(), ADD_VALUES);
        }
    }
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}
