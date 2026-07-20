#include "ROM/ROM_method/FOM_direct.hpp"

#include <petscsys.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#if defined(__has_include)
#if __has_include(<mkl_service.h>)
#include <mkl_service.h>
#define STAGE2_HAS_MKL_SERVICE 1
#endif
#endif

namespace {

constexpr PetscReal kZeroRhsTol = 1e-14;

int env_int_or_default(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    return std::max(1, std::atoi(value));
}

int factor_threads_for_mat(Mat A) {
    PetscMPIInt size = 1;
    MPI_Comm_size(PetscObjectComm((PetscObject)A), &size);
    if (size != 1) return 1;
    return env_int_or_default("FACTOR_NUM_THREADS", env_int_or_default("MKL_NUM_THREADS", 1));
}

int runtime_mkl_threads_for_mat(Mat A) {
    PetscMPIInt size = 1;
    MPI_Comm_size(PetscObjectComm((PetscObject)A), &size);
    if (size != 1) return 1;
    return env_int_or_default("MKL_NUM_THREADS", 1);
}

class MKLThreadGuard {
public:
    MKLThreadGuard(int factor_threads, int runtime_threads)
        : runtime_threads_(std::max(1, runtime_threads)) {
#if defined(STAGE2_HAS_MKL_SERVICE)
        mkl_set_num_threads_local(std::max(1, factor_threads));
#else
        (void)factor_threads;
#endif
    }

    ~MKLThreadGuard() {
#if defined(STAGE2_HAS_MKL_SERVICE)
        mkl_set_num_threads_local(runtime_threads_);
#endif
    }

private:
    int runtime_threads_ = 1;
};

} // namespace

FOM_direct::FOM_direct() : ROMBase({}) {
    name = "FOM_direct";
    use_mor = false;
}

FOM_direct::~FOM_direct() {
    if (ksp_) KSPDestroy(&ksp_);
}

void FOM_direct::set_factorization(Mat factor_in) {
    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    const int requested_factor_threads = factor_threads_for_mat(factor_in);
    const int requested_runtime_threads = runtime_mkl_threads_for_mat(factor_in);
    MKLThreadGuard mkl_threads(factor_threads_for_mat(factor_in),
                               runtime_mkl_threads_for_mat(factor_in));
    // TEMP THREAD DEBUG: comment out this block after comparing master/worker.
    {
        PetscMPIInt rank = 0;
        MPI_Comm_rank(PetscObjectComm((PetscObject)factor_in), &rank);
        if (rank == 0) {
#if defined(STAGE2_HAS_MKL_SERVICE)
            std::cout << "  thread_debug factorization: requested_factor_threads="
                      << requested_factor_threads
                      << " requested_runtime_threads=" << requested_runtime_threads
                      << " mkl_get_max_threads=" << mkl_get_max_threads()
                      << "\n";
#else
            std::cout << "  thread_debug factorization: MKL service header not available"
                      << " requested_factor_threads=" << requested_factor_threads
                      << " requested_runtime_threads=" << requested_runtime_threads
                      << "\n";
#endif
        }
    }
    if (ksp_) {
        KSPDestroy(&ksp_);
        ksp_ = nullptr;
    }
    factor = factor_in;
    KSPCreate(PETSC_COMM_WORLD, &ksp_);
    KSPSetOperators(ksp_, factor, factor);
    KSPSetType(ksp_, KSPPREONLY);
    PC pc = nullptr;
    KSPGetPC(ksp_, &pc);
    PCSetType(pc, PCLU);
    KSPSetFromOptions(ksp_);
    KSPSetUp(ksp_);
    PetscTime(&t1);
    last_factorization_time_ = static_cast<double>(t1 - t0);
}

KSP FOM_direct::get_factorized_solver(Mat lhs) const {
    if (lhs == factor && ksp_) return ksp_;
    return nullptr;
}

void FOM_direct::set_factorized_solver(Mat lhs, KSP ksp) {
    if (ksp) PetscObjectReference((PetscObject)ksp);
    if (ksp_) KSPDestroy(&ksp_);
    ksp_ = ksp;
    factor = lhs;
}

SolveStats FOM_direct::FOM_main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) {
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
    PetscInt its = 0;
    PetscReal rnorm = 0.0;
    KSPGetIterationNumber(ksp_, &its);
    KSPGetResidualNorm(ksp_, &rnorm);
    stats.ksp_iterations = static_cast<int>(its);
    stats.residual_norm = static_cast<double>(rnorm);
    stats.factorization_time = factorization_time;
    PetscTime(&t1);
    stats.solve_time = static_cast<double>(t1 - t0);
    return stats;
}

SolveStats FOM_direct::main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) {
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

