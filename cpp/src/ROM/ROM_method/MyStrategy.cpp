#include "ROM/ROM_method/MyStrategy.hpp"

#include "ROM/Reduced_Basis_Agorithm/UpdateBase.hpp"
#include "ROM/Utils/rom_petsc_utils.hpp"

#include <chrono>
#include <stdexcept>

namespace stage2 {

MyStrategy::MyStrategy(std::vector<std::shared_ptr<UpdateBase>> basis_list_in,
                       double residual_tol_in,
                       int n_p_in,
                       std::shared_ptr<ROMBase> fom_solver_in,
                       std::shared_ptr<ErrorEstimatorBase> error_estimator_in,
                       bool online_in)
    : ROMBase(std::move(basis_list_in)),
      residual_tol(residual_tol_in),
      n_p(n_p_in),
      fom_solver(std::move(fom_solver_in)),
      error_estimator(std::move(error_estimator_in)),
      online(online_in) {
    name = "MyStrategy";
}

void MyStrategy::set_factorization(Mat factor) {
    if (!fom_solver) throw std::runtime_error("MyStrategy needs fom_solver");
    fom_solver->set_factorization(factor);
}

void MyStrategy::clear_reduced_cache() {
    A_r_cache.clear();
    A_r_lhs_cache_ = nullptr;
    A_r_rank_cache_ = 0;
}

bool MyStrategy::solve_rom(Mat lhs, Vec rhs, Vec u_rom, const std::vector<Vec>& V, SolveStats& stats) {
    const int r = static_cast<int>(V.size());
    if (r <= 0) return false;
    if (fom_flag || A_r_lhs_cache_ != lhs || A_r_rank_cache_ != r) {
        A_r_cache.assign(static_cast<std::size_t>(r * r), 0.0);
        Vec work = nullptr;
        VecDuplicate(rhs, &work);
        for (int j = 0; j < r; ++j) {
            MatMult(lhs, V[static_cast<std::size_t>(j)], work);
            for (int i = 0; i < r; ++i) {
                PetscScalar dot = 0.0;
                VecDot(V[static_cast<std::size_t>(i)], work, &dot);
                A_r_cache[static_cast<std::size_t>(i * r + j)] = PetscRealPart(dot);
            }
        }
        VecDestroy(&work);
        A_r_lhs_cache_ = lhs;
        A_r_rank_cache_ = r;
    }

    std::vector<double> b_r(static_cast<std::size_t>(r), 0.0);
    for (int i = 0; i < r; ++i) {
        PetscScalar dot = 0.0;
        VecDot(V[static_cast<std::size_t>(i)], rhs, &dot);
        b_r[static_cast<std::size_t>(i)] = PetscRealPart(dot);
    }

    std::vector<double> coeff;
    if (!solve_dense_system(A_r_cache, std::move(b_r), coeff)) return false;

    VecSet(u_rom, 0.0);
    for (int i = 0; i < r; ++i) {
        VecAXPY(u_rom, coeff[static_cast<std::size_t>(i)], V[static_cast<std::size_t>(i)]);
    }
    stats.ksp_iterations = 0;
    stats.residual_norm = compute_relative_residual(lhs, rhs, u_rom);
    return true;
}

SolveStats MyStrategy::main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) {
    if (!fom_solver || !error_estimator) throw std::runtime_error("MyStrategy is not fully initialized");
    if (basis_list.size() < 2) throw std::runtime_error("MyStrategy expects basis_list{1} and basis_list{2}");

    ROMInfo info;
    info.name = name;
    const auto& V = basis_list[1]->getBasis(context);
    info.rank = static_cast<int>(V.size());

    bool use_fom = !use_mor || V.empty();
    Vec u_rom = nullptr;
    SolveStats stats;

    if (!use_fom) {
        VecDuplicate(rhs, &u_rom);
        SolveStats rom_stats;
        if (!solve_rom(lhs, rhs, u_rom, V, rom_stats)) {
            use_fom = true;
            fom_flag = 1;
        } else {
            const ErrorAssessment assessment = error_estimator->assess(lhs, rhs, u_rom, context);
            info.rank = static_cast<int>(V.size());
            info.rank_true = static_cast<int>(V.size());
            info.error_indicator = assessment.error_indicator;
            info.residual = assessment.residual;
            info.residual_norm = assessment.residual;
            info.threshold = assessment.threshold;
            info.residual_tol_eff = assessment.residual_tol_eff;
            info.accepted = assessment.accepted;
            use_fom = !assessment.accepted;
            stats = rom_stats;
        }
    }

    if (use_fom) {
        fom_flag = 1;
        stats = fom_solver->FOM_main(lhs, rhs, u, context);
        info.fom_used = true;
        info.ksp_iterations = stats.ksp_iterations;
        if (u_rom) {
            info.error = compute_relative_solution_error(u, u_rom);
            info.basis_error = compute_basis_error(V, u);
        }
        if (online) {
            basis_list[1]->update(u, context);
        }
    } else {
        VecCopy(u_rom, u);
        info.rom_used = true;
        info.used_rom = true;
        info.accepted = true;
        fom_flag = context.time_step == 0 ? 1 : 0;
    }

    basis_list[0]->addSnapshot(u, context);

    if (context.is_end_of_analysis) {
        if (basis_list[0]->max_rank != 0) {
            basis_list[0]->update(u, context);
            basis_list[1]->setBasis(basis_list[0]->V, basis_list[0]->s);
        } else {
            basis_list[1]->reset();
        }
        basis_list[1]->clearSnapshots();
        basis_list[0]->reset();
        fom_flag = 1;
        clear_reduced_cache();
    }

    if (u_rom) VecDestroy(&u_rom);
    store_info(info, context);
    return stats;
}

} // namespace stage2
