#include "Analysis/Analysis.hpp"
#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/StructuralElement/StructuralElement.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"

#include "ROM/ROM_method/ROM_Base.hpp"
#include "helper/MemoryDebug.hpp"
#include "helper/MyAlgebra.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

int omp_threads_from_env() {
    const char* value = std::getenv("OMP_NUM_THREADS");
    if (!value || !*value) return 1;
    char* end = nullptr;
    const long n = std::strtol(value, &end, 10);
    if (end == value || n < 1) return 1;
    return static_cast<int>(n);
}

void set_neum(const Analysis& analysis, Eigen::VectorXd& f, int istep) {
    f.setZero();
    for (const auto& bc_ptr : analysis.Neum) {
        const BoundaryCondition& bc = *bc_ptr;
        const double value = bc.value(analysis, istep, static_cast<double>(bc.nDof));
        for (int dof : bc.dof_ids) f(dof) += value;
    }
}

void set_dirichlet_rhs(const Analysis& analysis, Eigen::VectorXd& rhs, int istep) {
    for (const auto& bc_ptr : analysis.Dirich) {
        const BoundaryCondition& bc = *bc_ptr;
        const double value = bc.value(analysis, istep, 1.0);
        for (int dof : bc.dof_ids) rhs(dof) = value;
    }
}

void clear_stored_eigen(EigenStaticLinearElastic& analysis) {
    if (analysis.ROM) analysis.ROM->set_factorized_solver(analysis.K_inf, nullptr);
    analysis.K_inf.resize(0, 0);
    analysis.factorized_K_initial.reset();
}

} // namespace


void EigenStaticLinearElastic::solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter) {
    if (!ROM) throw std::runtime_error("EigenStaticLinearElastic::solver requires analysis.ROM initialized");
    if (Nstep != 1) throw std::runtime_error("EigenStaticLinearElastic requires Nstep = 1");
    clear_stored_eigen(*this);

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    MemoryDebug::print("static_forward_begin");
    MyAlgebra::Sparse& K = K_inf;
    assemble_K(fem, y_elem, K);
    assembly_time = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "  assembly[s]: eigen static K = " << assembly_time << "\n";

    const std::vector<int> fixed = SPC();
    MyAlgebra::EigenOps::apply_dirichlet_to_matrix(K, fixed, 1.0);

    const int ndof = fem.ndof;
    Eigen::VectorXd u = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(ndof);
    set_neum(*this, f_ext, Nstep);
    rhs = f_ext;
    set_dirichlet_rhs(*this, rhs, Nstep);

    const auto ts = Clock::now();
    ROM->InitializeROMContext(*this, Iter, 1, Nstep, 1.0, "primal", "static_equilibrium", true);
    const SolveStats stats = ROM->main(K, rhs, u);
    solve_time = std::chrono::duration<double>(Clock::now() - ts).count();
    if (save_factorized_K_initial) {
        factorized_K_initial = ROM->get_factorized_solver(K);
    } else {
        factorized_K_initial.reset();
    }

    U = Eigen::MatrixXd::Zero(ndof, Nstep + 1);
    F_ext = Eigen::MatrixXd::Zero(ndof, Nstep + 1);
    ksp_iterations.clear();
    ksp_residuals.clear();
    ksp_iterations.push_back(stats.ksp_iterations);
    ksp_residuals.push_back(stats.residual_norm);
    SPCdofs = fixed;
    free = free_dofs(ndof);
    U.col(Nstep) = u;
    F_ext.col(Nstep) = f_ext;
    std::cout << "static solve_res = " << stats.residual_norm
              << ", ksp_it = " << stats.ksp_iterations << "\n";
}


void EigenStaticLinearElastic::assemble_K(const FEM& fem,
                                          const Eigen::MatrixXd& y_elem,
                                          Eigen::SparseMatrix<double>& K) const {
    const int ndofe_max = (fem.dim == 2) ? 8 : 24;
    const std::size_t reserve_nnz = static_cast<std::size_t>(fem.NElem) *
                                    static_cast<std::size_t>(ndofe_max) *
                                    static_cast<std::size_t>(ndofe_max);
    const int num_threads = omp_threads_from_env();
    std::vector<std::vector<MyAlgebra::Triplet>> thread_triplets(static_cast<std::size_t>(num_threads));
    const std::size_t reserve_per_thread = reserve_nnz / static_cast<std::size_t>(num_threads) + 1;
    for (auto& triplets : thread_triplets) triplets.reserve(reserve_per_thread);

    for (const auto& region_ptr : fem.Regions) {
        const Region& region = *region_ptr;
        const auto region_element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front())));
        if (!region_element || !region_element->material_paras) {
            throw std::runtime_error("Element material is not initialized");
        }
        const double region_nu = dynamic_cast<const ElasticParas&>(*region_element->material_paras).nu_values.front();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            if (region.elem_ids.empty()) throw std::runtime_error("eigen assemble_K: region has no elements");
            uniform_em = fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front()))->compute_element_matrices(fem, region_nu, fem.thickness);
        }

        #pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            auto& local_triplets = thread_triplets[static_cast<std::size_t>(tid)];

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
                    local_em = fem.ElementObjects.at(static_cast<std::size_t>(elem_id))->compute_element_matrices(fem, region_nu, fem.thickness);
                    em_ptr = &local_em;
                }
                const ElementMatrix& em = *em_ptr;
                const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
                for (int i = 0; i < em.ndof; ++i) {
                    const int row = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                    for (int j = 0; j < em.ndof; ++j) {
                        const int col = static_cast<int>(dofs[static_cast<std::size_t>(j)]);
                        local_triplets.emplace_back(row, col, Eeff * em.ke(i, j));
                    }
                }
            }
        }
    }

    std::size_t total_triplets = 0;
    for (const auto& local : thread_triplets) total_triplets += local.size();
    std::vector<MyAlgebra::Triplet> triplets;
    triplets.reserve(total_triplets);
    for (auto& local : thread_triplets) {
        triplets.insert(triplets.end(),
                        std::make_move_iterator(local.begin()),
                        std::make_move_iterator(local.end()));
    }
    K.resize(fem.ndof, fem.ndof);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
}
