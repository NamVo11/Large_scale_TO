#include "Analysis/Analysis.hpp"
#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/StructuralElement/StructuralElement.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"
#include "helper/MemoryDebug.hpp"
#include "helper/MyAlgebra.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include <iterator>


namespace
{
    int omp_threads_from_env()
    {
        const char *value = std::getenv("OMP_NUM_THREADS");
        if (!value || !*value) return 1;

        char *end = nullptr;
        const long n = std::strtol(value, &end, 10);
        if (end == value || n < 1) return 1;
        return static_cast<int>(n);
    }

    void set_neum(const Analysis &analysis, Eigen::VectorXd &f, int istep)
    {
        f.setZero();
        for (const auto &bc_ptr : analysis.Neum)
        {
            const BoundaryCondition& bc = *bc_ptr;
            const double value = bc.value(analysis, istep, static_cast<double>(bc.nDof));
            for (int dof : bc.dof_ids)
                f(dof) += value;
        }
    }

    void set_dirichlet_rhs(const Analysis &analysis, Eigen::VectorXd &rhs, int istep)
    {
        for (const auto &bc_ptr : analysis.Dirich)
        {
            const BoundaryCondition& bc = *bc_ptr;
            const double value = bc.value(analysis, istep, 1.0);
            for (int dof : bc.dof_ids)
                rhs(dof) = value;
        }
    }

    void clear_stored_eigen(EigenDynamicLinearElastic& analysis)
    {
        if (analysis.ROM) analysis.ROM->set_factorized_solver(analysis.K_inf, nullptr);
        analysis.K_inf.resize(0, 0);
        analysis.M.resize(0, 0);
        analysis.C.resize(0, 0);
        analysis.K_eff.resize(0, 0);
        analysis.factorized_K_initial.reset();
        analysis.factorized_K_eff.reset();
    }

} // namespace

void EigenDynamicLinearElastic::solver(const FEM &fem, const Eigen::MatrixXd &y_elem, int Iter)
{
    EigenDynamicLinearElastic &analysis = *this;
    (void)Iter;
    if (!analysis.ROM)
        throw std::runtime_error("EigenDynamicLinearElastic::solver requires analysis.ROM initialized");
    clear_stored_eigen(analysis);

    const int ndof = fem.ndof;
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    MemoryDebug::print("forward_begin");
    // Memory-saving flow: K_inf is assembled as raw K first. After
    // K_eff is formed, K_inf is converted in-place to K_initial with SPC.
    MyAlgebra::Sparse& K = analysis.K_inf;
    MyAlgebra::Sparse& M = analysis.M;
    assemble_K_M(fem, y_elem, K, M);
    MemoryDebug::print("forward_after_assemble_K_M");
    analysis.assembly_time = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "  assembly[s]: eigen K/M = " << analysis.assembly_time << "\n";

    const std::vector<int> fixed = analysis.SPC();

    const double beta = 0.25;
    const double gamma = 0.5;
    const double a0c = 1.0 / (beta * analysis.dt * analysis.dt);
    const double a1c = gamma / (beta * analysis.dt);
    const bool has_damping = analysis.alpha_M != 0.0 || analysis.alpha_K != 0.0;
    if (has_damping) {
        analysis.C = analysis.alpha_M * M + analysis.alpha_K * K;
        analysis.K_eff = K + a0c * M + a1c * analysis.C;
    } else {
        analysis.C.resize(0, 0);
        analysis.K_eff = K + a0c * M;
    }
    MyAlgebra::Sparse& K_initial = analysis.K_inf;
    MyAlgebra::Sparse& C = analysis.C;
    MyAlgebra::Sparse& K_eff = analysis.K_eff;
    MemoryDebug::print("forward_after_prepare_Keff");
    // From here, K_initial stores the SPC-applied initial stiffness, not raw K.
    MyAlgebra::EigenOps::apply_dirichlet_to_matrix(K_initial, fixed, 1.0);
    MyAlgebra::EigenOps::apply_dirichlet_to_matrix(K_eff, fixed, 1.0);
    MemoryDebug::print("forward_after_dirichlet_K_initial");

    Eigen::VectorXd u = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd u_pred = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd v_pred = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd damping_predictor;
    if (has_damping) damping_predictor = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(ndof);
    MemoryDebug::print("forward_after_vec_create");

    std::shared_ptr<ROMBase> &rom = analysis.ROM;

    set_neum(analysis, f_ext, 0);
    rhs = f_ext;
    set_dirichlet_rhs(analysis, rhs, 0);
    rom->InitializeROMContext(analysis, Iter, 1, 0, 0.0, "primal", "initial_equilibrium", analysis.Nstep == 0);
    const SolveStats initial = rom->main(K_initial, rhs, u);

    if (analysis.save_factorized_K_initial) {
        analysis.factorized_K_initial = rom->get_factorized_solver(K_initial);
    } else {
        rom->set_factorized_solver(K_initial, nullptr);
        analysis.factorized_K_initial.reset();
    }

    MemoryDebug::print("forward_after_initial_solve_store_factor");
    std::cout << "step   0: solve_res = " << initial.residual_norm << "\n";

    const int nstore = analysis.Nstep + 1;
    analysis.U.resize(ndof, nstore);
    analysis.V.resize(has_damping ? ndof : 0, has_damping ? nstore : 0);
    analysis.A.resize(ndof, nstore);
    analysis.F_ext.resize(ndof, nstore);
    MemoryDebug::print("forward_after_history_alloc");
    analysis.ksp_iterations.clear();
    analysis.ksp_residuals.clear();
    analysis.SPCdofs = fixed;
    analysis.free = analysis.free_dofs(ndof);
    analysis.U.col(0) = u;
    if (has_damping) analysis.V.col(0) = v;
    analysis.A.col(0) = acc;
    analysis.F_ext.col(0) = f_ext;

    t0 = Clock::now();
    for (int istep = 1; istep <= analysis.Nstep; ++istep)
    {
        u_pred = u + analysis.dt * v + analysis.dt * analysis.dt * (0.5 - beta) * acc;
        v_pred = v + analysis.dt * (1.0 - gamma) * acc;

        set_neum(analysis, f_ext, istep);
        rhs.noalias() = f_ext + a0c * (M * u_pred);
        if (has_damping) {
            damping_predictor.noalias() = a1c * u_pred - v_pred;
            rhs.noalias() += C * damping_predictor;
        }
        set_dirichlet_rhs(analysis, rhs, istep);

        rom->InitializeROMContext(analysis,
                                  Iter,
                                  istep + 1,
                                  istep,
                                  istep * analysis.dt,
                                  "primal",
                                  "newmark_effective",
                                  istep == analysis.Nstep);
        const SolveStats step = rom->main(K_eff, rhs, u);

        if (!analysis.factorized_K_eff)
            analysis.factorized_K_eff = rom->get_factorized_solver(K_eff);
        if (istep == 1 || istep == analysis.Nstep) {
            MemoryDebug::print("forward_after_step_" + std::to_string(istep));
        }
        analysis.ksp_iterations.push_back(step.ksp_iterations);
        analysis.ksp_residuals.push_back(step.residual_norm);

        acc = a0c * (u - u_pred);
        v = v_pred + gamma * analysis.dt * acc;

        std::cout << "step " << istep << ": solve_res = "
                  << step.residual_norm << ", ksp_it = " << step.ksp_iterations
                  << ", ksp_res = " << step.residual_norm << "\n";

        analysis.U.col(istep) = u;
        if (has_damping) analysis.V.col(istep) = v;
        analysis.A.col(istep) = acc;
        analysis.F_ext.col(istep) = f_ext;
    }
    analysis.solve_time = std::chrono::duration<double>(Clock::now() - t0).count();
    MemoryDebug::print("forward_after_store_matrices");
    MemoryDebug::print("forward_after_destroy_temp");
}

// Parallel assembly version using OpenMP. Each thread has its own vector of triplets to avoid data races
void EigenDynamicLinearElastic::assemble_K_M(const FEM &fem,
                                            const Eigen::MatrixXd &y_elem,
                                            Eigen::SparseMatrix<double> &K,
                                            Eigen::SparseMatrix<double> &M) const
                                            
{
    const Analysis &analysis = *this;

    const int ndofe_max = (fem.dim == 2) ? 8 : 24;

    const std::size_t reserve_nnz =
        static_cast<std::size_t>(fem.NElem) *
        static_cast<std::size_t>(ndofe_max) *
        static_cast<std::size_t>(ndofe_max);

    const int num_threads = omp_threads_from_env();

    // Mỗi thread có vector riêng để tránh data race khi emplace_back().
    std::vector<std::vector<MyAlgebra::Triplet>> thread_k_triplets(
        static_cast<std::size_t>(num_threads));

    std::vector<std::vector<MyAlgebra::Triplet>> thread_m_triplets(
        static_cast<std::size_t>(num_threads));

    const std::size_t reserve_per_thread =
        reserve_nnz / static_cast<std::size_t>(num_threads) + 1;

    for (int tid = 0; tid < num_threads; ++tid)
    {
        thread_k_triplets[static_cast<std::size_t>(tid)]
            .reserve(reserve_per_thread);

        thread_m_triplets[static_cast<std::size_t>(tid)]
            .reserve(reserve_per_thread);
    }

    for (const auto& region_ptr : fem.Regions)
    {
        const Region& region = *region_ptr;
        const auto region_element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front())));
        if (!region_element || !region_element->material_paras) {
            throw std::runtime_error("Element material is not initialized");
        }
        const double region_nu = dynamic_cast<const ElasticParas&>(*region_element->material_paras).nu_values.front();

        ElementMatrix uniform_em;

        if (fem.uniform_element_matrices)
        {
            if (region.elem_ids.empty())
            {
                throw std::runtime_error( "eigen assemble_K_M: region has no elements");
            }

            region_element->nu_interp = region_nu;
            uniform_em = region_element->create_local_matrix(fem, Eigen::VectorXd::Zero(region_element->dof_count()), true, true);
        }

        /*
         * uniform_em được tạo trước vùng song song và sau đó chỉ được đọc.
         * Mỗi thread ghi vào vector triplet riêng.
         */
            #pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();

            auto &local_k_triplets = thread_k_triplets[static_cast<std::size_t>(tid)];

            auto &local_m_triplets = thread_m_triplets[static_cast<std::size_t>(tid)];

            #pragma omp for schedule(static)
            
            for (std::size_t e = 0; e < region.elem_ids.size(); ++e)
            {
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
                const ElementMatrix *em_ptr = &uniform_em;

                if (!fem.uniform_element_matrices)
                {
                    local_em = element->create_local_matrix(fem, Eigen::VectorXd::Zero(element->dof_count()), true, true);
                                                     
                    em_ptr = &local_em;
                }

                const ElementMatrix &em = *em_ptr;

                const auto &dofs = fem.ElementDofs.at( static_cast<std::size_t>(elem_id));

                for (int i = 0; i < em.ndof; ++i)
                {
                    const int row = static_cast<int>( dofs[static_cast<std::size_t>(i)]);

                    for (int j = 0; j < em.ndof; ++j)
                    {
                        const int col = static_cast<int>(dofs[static_cast<std::size_t>(j)]);

                        local_k_triplets.emplace_back(row, col, (fem.uniform_element_matrices ? Eeff : 1.0) * em.ke(i, j));

                        local_m_triplets.emplace_back(row,col, (fem.uniform_element_matrices ? rhoeff : 1.0) * em.me(i, j));
                    }
                }
            }
        }
    }

    /*
     * Tính tổng số triplet thực tế rồi ghép các vector của thread.
     * Phần này chạy tuần tự nhưng chỉ là sao chép/move dữ liệu liên tục.
     */
    std::size_t total_k_triplets = 0;
    std::size_t total_m_triplets = 0;

    for (int tid = 0; tid < num_threads; ++tid)
    {
        total_k_triplets += thread_k_triplets[static_cast<std::size_t>(tid)].size();

        total_m_triplets += thread_m_triplets[static_cast<std::size_t>(tid)].size();
    }

    std::vector<MyAlgebra::Triplet> k_triplets;
    std::vector<MyAlgebra::Triplet> m_triplets;

    k_triplets.reserve(total_k_triplets);
    m_triplets.reserve(total_m_triplets);

    for (int tid = 0; tid < num_threads; ++tid)
    {
        auto &local_k = thread_k_triplets[static_cast<std::size_t>(tid)];

        auto &local_m = thread_m_triplets[static_cast<std::size_t>(tid)];

        k_triplets.insert(
            k_triplets.end(),
            std::make_move_iterator(local_k.begin()),
            std::make_move_iterator(local_k.end()));

        m_triplets.insert(
            m_triplets.end(),
            std::make_move_iterator(local_m.begin()),
            std::make_move_iterator(local_m.end()));
    }

    K.resize(fem.ndof, fem.ndof);
    M.resize(fem.ndof, fem.ndof);

    /*
     * Eigen tự cộng các triplet có cùng (row, col).
     * Hai phần tử cùng đóng góp vào một global entry là hợp lệ.
     */
    K.setFromTriplets(
        k_triplets.begin(),
        k_triplets.end());

    M.setFromTriplets(
        m_triplets.begin(),
        m_triplets.end());

    K.makeCompressed();
    M.makeCompressed();
}
