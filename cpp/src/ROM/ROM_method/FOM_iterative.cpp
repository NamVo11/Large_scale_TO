#include "ROM/ROM_method/FOM_iterative.hpp"

namespace stage2 {

FOM_iterative::FOM_iterative(std::string method_in) : ROMBase({}), method(std::move(method_in)) {
    name = "FOM_iterative";
    use_mor = false;
}

FOM_iterative::~FOM_iterative() {
    if (ksp_) KSPDestroy(&ksp_);
}

void FOM_iterative::set_factorization(Mat factor_in) {
    factor = factor_in;
    if (!ksp_) KSPCreate(PETSC_COMM_WORLD, &ksp_);
    KSPSetOperators(ksp_, factor, factor);
    KSPSetType(ksp_, KSPCG);
    KSPSetTolerances(ksp_, pcg_tol, PETSC_DEFAULT, PETSC_DEFAULT, pcg_maxit);
    PC pc = nullptr;
    KSPGetPC(ksp_, &pc);
    PCSetType(pc, PCGAMG);
    KSPSetFromOptions(ksp_);
    KSPSetUp(ksp_);
}

SolveStats FOM_iterative::FOM_main(Mat lhs, Vec rhs, Vec u, const ROMContext&) {
    if (!ksp_ || lhs != factor) set_factorization(lhs);
    KSPSolve(ksp_, rhs, u);
    SolveStats stats;
    PetscInt its = 0;
    PetscReal rnorm = 0.0;
    KSPGetIterationNumber(ksp_, &its);
    KSPGetResidualNorm(ksp_, &rnorm);
    stats.ksp_iterations = static_cast<int>(its);
    stats.residual_norm = static_cast<double>(rnorm);
    return stats;
}

SolveStats FOM_iterative::main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) {
    const SolveStats stats = FOM_main(lhs, rhs, u, context);
    ROMInfo info;
    info.name = name;
    info.fom_used = true;
    info.accepted = true;
    info.ksp_iterations = stats.ksp_iterations;
    info.residual_norm = stats.residual_norm;
    store_info(info, context);
    return stats;
}

} // namespace stage2

