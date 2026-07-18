#include "ROM/ROM_method/FOM_iterative.hpp"

#include <iostream>

namespace stage2 {
namespace {

constexpr PetscReal kZeroRhsTol = 1e-14;

} // namespace

FOM_iterative::FOM_iterative(std::string method_in) : ROMBase({}), method(std::move(method_in)) {
    name = "FOM_iterative";
    use_mor = false;
}

FOM_iterative::~FOM_iterative() {
    if (ksp_) KSPDestroy(&ksp_);
}

void FOM_iterative::set_factorization(Mat factor_in) {
    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    if (ksp_) {
        KSPDestroy(&ksp_);
        ksp_ = nullptr;
    }
    factor = factor_in;
    KSPCreate(PETSC_COMM_WORLD, &ksp_);
    KSPSetOperators(ksp_, factor, factor);
    KSPSetType(ksp_, KSPCG);
    KSPSetTolerances(ksp_, pcg_tol, PETSC_DEFAULT, PETSC_DEFAULT, pcg_maxit);
    PC pc = nullptr;
    KSPGetPC(ksp_, &pc);
    PCSetType(pc, PCGAMG);
    KSPSetFromOptions(ksp_);
    KSPSetUp(ksp_);
    PetscTime(&t1);
    last_factorization_time_ = static_cast<double>(t1 - t0);
}

KSP FOM_iterative::get_factorized_solver(Mat lhs) const {
    if (lhs == factor && ksp_) return ksp_;
    return nullptr;
}

void FOM_iterative::set_factorized_solver(Mat lhs, KSP ksp) {
    if (ksp) PetscObjectReference((PetscObject)ksp);
    if (ksp_) KSPDestroy(&ksp_);
    ksp_ = ksp;
    factor = lhs;
}

SolveStats FOM_iterative::FOM_main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) {
    SolveStats stats;
    PetscReal rhs_norm = 0.0;
    VecNorm(rhs, NORM_2, &rhs_norm);
    if (rhs_norm < kZeroRhsTol) {
        VecSet(u, 0.0);
        stats.residual_norm = static_cast<double>(rhs_norm);
        return stats;
    }

    double factorization_time = 0.0;
    if (!ksp_ || lhs != factor) {
        set_factorization(lhs);
        factorization_time = last_factorization_time_;
        PetscMPIInt rank = 0;
        MPI_Comm_rank(PetscObjectComm((PetscObject)lhs), &rank);
        if (rank == 0) {
            std::cout << "  factorization[s]: " << context.role
                      << " " << context.system_type << " = "
                      << factorization_time << "\n";
        }
    }
    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    KSPSolve(ksp_, rhs, u);
    PetscTime(&t1);
    PetscInt its = 0;
    PetscReal rnorm = 0.0;
    KSPGetIterationNumber(ksp_, &its);
    KSPGetResidualNorm(ksp_, &rnorm);
    stats.ksp_iterations = static_cast<int>(its);
    stats.residual_norm = static_cast<double>(rnorm);
    stats.factorization_time = factorization_time;
    stats.solve_time = static_cast<double>(t1 - t0);
    return stats;
}

SolveStats FOM_iterative::main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) {
    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    const SolveStats stats = FOM_main(lhs, rhs, u, context);
    PetscTime(&t1);
    ROMInfo info;
    info.name = name;
    info.fom_used = true;
    info.accepted = true;
    info.ksp_iterations = stats.ksp_iterations;
    info.residual_norm = stats.residual_norm;
    info.timing_fom_factorization = stats.factorization_time;
    info.timing_fom_solve = stats.solve_time;
    info.timing_total = static_cast<double>(t1 - t0);
    store_info(info, context);
    return stats;
}

} // namespace stage2
