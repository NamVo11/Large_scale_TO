#include "Analysis/Analysis.hpp"
#include "MatlabLikeData.hpp"
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

    using Sparse = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;

    std::pair<double, double> interp_region(const Region &region, double x)
    {
        const double Eeff = region.interp.elastic_interp
                                ? region.interp.elastic_interp(region.mat_paras.E_inf, x)
                                : RAMP(region.mat_paras.E_inf, x, region.interp.p_E_elastic);
        const double rhoeff = region.interp.mass_density_interp
                                  ? region.interp.mass_density_interp(region.mat_paras.rho, x)
                                  : RAMP(region.mat_paras.rho, x, region.interp.p_rho);
        return {Eeff, rhoeff};
    }

    int omp_threads_from_env()
    {
        const char *value = std::getenv("OMP_NUM_THREADS");
        if (!value || !*value) return 1;

        char *end = nullptr;
        const long n = std::strtol(value, &end, 10);
        if (end == value || n < 1) return 1;
        return static_cast<int>(n);
    }

} // namespace


// Serial reference version kept for checking the parallel assembly.
/*
void EigenDynamicLinearElastic::assemble_K_M(const FEM &fem,
                                             const Eigen::MatrixXd &y_elem,
                                             Eigen::SparseMatrix<double> &K,
                                             Eigen::SparseMatrix<double> &M) const
{
    const Analysis &analysis = *this;
    std::vector<Triplet> k_triplets;
    std::vector<Triplet> m_triplets;
    const int ndofe_max = fem.dim == 2 ? 8 : 24;
    const std::size_t reserve_nnz = static_cast<std::size_t>(fem.NElem) *
                                    static_cast<std::size_t>(ndofe_max) *
                                    static_cast<std::size_t>(ndofe_max);
    k_triplets.reserve(reserve_nnz);
    m_triplets.reserve(reserve_nnz);

    for (const Region &region : analysis.regions)
    {
        const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices)
        {
            if (region.elem_ids.empty())
                throw std::runtime_error("eigen assemble_K_M: region has no elements");
            uniform_em = fem.compute_element_matrices(region.elem_ids.front(), region_nu, fem.Thickness);
        }
        for (int elem_id : region.elem_ids)
        {
            const double x = y_elem(elem_id, 0);
            const auto [Eeff, rhoeff] = interp_region(region, x);
            ElementMatrix local_em;
            const ElementMatrix *em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices)
            {
                local_em = fem.compute_element_matrices(elem_id, region_nu, fem.Thickness);
                em_ptr = &local_em;
            }
            const ElementMatrix &em = *em_ptr;
            const auto &rows = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            for (int i = 0; i < em.ndof; ++i)
            {
                const int row = static_cast<int>(rows[static_cast<std::size_t>(i)]);
                for (int j = 0; j < em.ndof; ++j)
                {
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
*/

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
    std::vector<std::vector<Triplet>> thread_k_triplets(
        static_cast<std::size_t>(num_threads));

    std::vector<std::vector<Triplet>> thread_m_triplets(
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

    for (const Region &region : analysis.regions)
    {
        const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();

        ElementMatrix uniform_em;

        if (fem.uniform_element_matrices)
        {
            if (region.elem_ids.empty())
            {
                throw std::runtime_error( "eigen assemble_K_M: region has no elements");
            }

            uniform_em = fem.compute_element_matrices(region.elem_ids.front(), region_nu, fem.Thickness);
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
                const auto [Eeff, rhoeff] = interp_region(region, x);

                ElementMatrix local_em;
                const ElementMatrix *em_ptr = &uniform_em;

                if (!fem.uniform_element_matrices)
                {
                    local_em = fem.compute_element_matrices(elem_id, region_nu, fem.Thickness);
                                                     
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

                        local_k_triplets.emplace_back(row, col, Eeff * em.ke(i, j));

                        local_m_triplets.emplace_back(row,col, rhoeff * em.me(i, j));
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

    std::vector<Triplet> k_triplets;
    std::vector<Triplet> m_triplets;

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

namespace
{

    void set_neum(const Analysis &analysis, Eigen::VectorXd &f, int istep)
    {
        f.setZero();
        for (const auto &bc : analysis.Neum)
        {
            const double value = bc.fun_value ? bc.fun_value(istep) : bc_value(bc, analysis, istep);
            for (int dof : bc.dof_ids)
                f(dof) += value;
        }
    }

    void set_dirichlet_rhs(const Analysis &analysis, Eigen::VectorXd &rhs, int istep)
    {
        for (const auto &bc : analysis.Dirich)
        {
            const double value = bc.fun_value ? bc.fun_value(istep) : bc_value(bc, analysis, istep);
            for (int dof : bc.dof_ids)
                rhs(dof) = value;
        }
    }

    ROMContext make_context(const Analysis &analysis,
                            int Iter,
                            int solve_step,
                            int time_step,
                            const std::string &system_type,
                            bool is_end)
    {
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

    void clear_stored_eigen(Analysis& analysis)
    {
        if (analysis.ROM) analysis.ROM->set_factorized_solver(analysis.results.K_inf_eigen, nullptr);
        analysis.results.K_inf_eigen.resize(0, 0);
        analysis.results.M_eigen.resize(0, 0);
        analysis.results.K_eff_eigen.resize(0, 0);
        analysis.results.factorized_K_initial_eigen.reset();
        analysis.results.factorized_K_eff_eigen.reset();
    }

} // namespace

void EigenDynamicLinearElastic::solver(const FEM &fem, const Eigen::MatrixXd &y_elem, int Iter)
{
    Analysis &analysis = *this;
    (void)Iter;
    if (!analysis.ROM)
        throw std::runtime_error("EigenDynamicLinearElastic::solver requires analysis.ROM initialized");
    clear_stored_eigen(analysis);

    const int ndof = fem.ndof;
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    MemoryDebug::print("forward_begin");
    // Memory-saving flow: K_inf_eigen is assembled as raw K first. After
    // K_eff_eigen is formed, K_inf_eigen is converted in-place to K_initial with SPC.
    Sparse& K = analysis.results.K_inf_eigen;
    Sparse& M = analysis.results.M_eigen;
    assemble_K_M(fem, y_elem, K, M);
    MemoryDebug::print("forward_after_assemble_K_M");
    analysis.results.assembly_time = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "  assembly[s]: eigen K/M = " << analysis.results.assembly_time << "\n";

    const std::vector<int> fixed = analysis.SPC();

    const double beta = 0.25;
    const double gamma = 0.5;
    const double a0c = 1.0 / (beta * analysis.dt * analysis.dt);
    analysis.results.K_eff_eigen = K + a0c * M;
    Sparse& K_initial = analysis.results.K_inf_eigen;
    Sparse& K_eff = analysis.results.K_eff_eigen;
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
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(ndof);
    MemoryDebug::print("forward_after_vec_create");

    std::shared_ptr<ROMBase> &rom = analysis.ROM;

    set_neum(analysis, f_ext, 0);
    rhs = f_ext;
    set_dirichlet_rhs(analysis, rhs, 0);
    const SolveStats initial = rom->main(
        K_initial, rhs, u, make_context(analysis, Iter, 1, 0, "initial_equilibrium", analysis.Nstep == 0));

    if (analysis.save_factorized_K_initial) {
        analysis.results.factorized_K_initial_eigen = rom->get_factorized_solver(K_initial);
    } else {
        rom->set_factorized_solver(K_initial, nullptr);
        analysis.results.factorized_K_initial_eigen.reset();
    }

    MemoryDebug::print("forward_after_initial_solve_store_factor");
    std::cout << "step   0: solve_res = " << initial.residual_norm << "\n";

    const int nstore = analysis.Nstep + 1;
    analysis.results.U.resize(ndof, nstore);
    analysis.results.V.resize(0, 0); // save memory by not storing velocity history
    analysis.results.A.resize(ndof, nstore);
    analysis.results.F_ext.resize(ndof, nstore);
    MemoryDebug::print("forward_after_history_alloc");
    analysis.results.ksp_iterations.clear();
    analysis.results.ksp_residuals.clear();
    analysis.results.SPCdofs = fixed;
    analysis.results.free = analysis.free_dofs(ndof);
    analysis.results.U.col(0) = u;
    // Velocity history is not used by the current objective/sensitivity flow.
    // Keep results.V empty in this memory-saving version.
    // analysis.results.V.col(0) = v;
    analysis.results.A.col(0) = acc;
    analysis.results.F_ext.col(0) = f_ext;

    t0 = Clock::now();
    for (int istep = 1; istep <= analysis.Nstep; ++istep)
    {
        u_pred = u + analysis.dt * v + analysis.dt * analysis.dt * (0.5 - beta) * acc;
        v_pred = v + analysis.dt * (1.0 - gamma) * acc;

        set_neum(analysis, f_ext, istep);
        rhs.noalias() = f_ext + a0c * (M * u_pred);
        set_dirichlet_rhs(analysis, rhs, istep);

        const SolveStats step = rom->main(
            K_eff, rhs, u, make_context(analysis, Iter, istep + 1, istep, "newmark_effective", istep == analysis.Nstep));

        if (!analysis.results.factorized_K_eff_eigen)
            analysis.results.factorized_K_eff_eigen = rom->get_factorized_solver(K_eff);
        if (istep == 1 || istep == analysis.Nstep) {
            MemoryDebug::print("forward_after_step_" + std::to_string(istep));
        }
        analysis.results.ksp_iterations.push_back(step.ksp_iterations);
        analysis.results.ksp_residuals.push_back(step.residual_norm);

        acc = a0c * (u - u_pred);
        v = v_pred + gamma * analysis.dt * acc;

        std::cout << "step " << istep << ": solve_res = "
                  << step.residual_norm << ", ksp_it = " << step.ksp_iterations
                  << ", ksp_res = " << step.residual_norm << "\n";

        analysis.results.U.col(istep) = u;
        // Velocity history is intentionally not stored in this memory-saving flow.
        // analysis.results.V.col(istep) = v;
        analysis.results.A.col(istep) = acc;
        analysis.results.F_ext.col(istep) = f_ext;
    }
    analysis.results.solve_time = std::chrono::duration<double>(Clock::now() - t0).count();
    MemoryDebug::print("forward_after_store_matrices");
    MemoryDebug::print("forward_after_destroy_temp");
}
