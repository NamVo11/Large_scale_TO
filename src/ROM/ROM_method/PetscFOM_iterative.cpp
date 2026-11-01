#include "ROM/ROM_method/FOM.hpp"

#include "helper/PetscEigenBridge.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
constexpr PetscReal kZeroRhsTol = 1e-14;
} // namespace

PetscFOM_iterative::PetscFOM_iterative(std::string method_in)
    : ROMBase({}), method(std::move(method_in)) {
    name = "PetscFOM_iterative";
    use_mor = false;
}

PetscFOM_iterative::~PetscFOM_iterative() {
    if (ksp_) KSPDestroy(&ksp_);
}

void PetscFOM_iterative::set_petsc_system(Mat lhs, Vec rhs, Vec u) {
    current_lhs_ = lhs;
    current_rhs_ = rhs;
    current_u_ = u;
}

void PetscFOM_iterative::set_factorization(Mat factor_in) {
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

KSP PetscFOM_iterative::get_petsc_factorized_solver(Mat lhs) const {
    if (lhs == factor && ksp_) return ksp_;
    return nullptr;
}

void PetscFOM_iterative::set_petsc_factorized_solver(Mat lhs, KSP ksp) {
    if (ksp) PetscObjectReference((PetscObject)ksp);
    if (ksp_) KSPDestroy(&ksp_);
    ksp_ = ksp;
    factor = lhs;
}

SolveStats PetscFOM_iterative::FOM_main(const Eigen::SparseMatrix<double>&,
                                        const Eigen::VectorXd&,
                                        Eigen::VectorXd& u) {
    // const ROMContext& context = ctx;
    if (!current_lhs_ || !current_rhs_ || !current_u_) {
        throw std::runtime_error("PetscFOM_iterative needs set_petsc_system before FOM_main");
    }

    SolveStats stats;
    PetscReal rhs_norm = 0.0;
    VecNorm(current_rhs_, NORM_2, &rhs_norm);
    if (rhs_norm < kZeroRhsTol) {
        VecSet(current_u_, 0.0);
        u.setZero();
        stats.residual_norm = static_cast<double>(rhs_norm);
        return stats;
    }

    double factorization_time = 0.0;
    if (!ksp_ || current_lhs_ != factor) {
        set_factorization(current_lhs_);
        factorization_time = last_factorization_time_;
        // PetscMPIInt rank = 0;
        // MPI_Comm_rank(PetscObjectComm((PetscObject)current_lhs_), &rank);
        // if (rank == 0) {
            // std::cout << "  factorization[s]: " << context.role
            //           << " " << context.system_type << " = "
            //           << factorization_time << "\n";
        // }
    }

    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    KSPSolve(ksp_, current_rhs_, current_u_);
    PetscTime(&t1);

    PetscInt its = 0;
    PetscReal rnorm = 0.0;
    KSPGetIterationNumber(ksp_, &its);
    KSPGetResidualNorm(ksp_, &rnorm);
    stats.ksp_iterations = static_cast<int>(its);
    stats.residual_norm = static_cast<double>(rnorm);
    stats.factorization_time = factorization_time;
    stats.solve_time = static_cast<double>(t1 - t0);
    u = petsc_vec_to_eigen(current_u_);
    return stats;
}

SolveStats PetscFOM_iterative::main(const Eigen::SparseMatrix<double>& lhs,
                                    const Eigen::VectorXd& rhs,
                                    Eigen::VectorXd& u) {
    const ROMContext& context = ctx;
    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    const SolveStats stats = FOM_main(lhs, rhs, u);
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
