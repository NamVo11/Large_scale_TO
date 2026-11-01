#include "ROM/ROM_method/FOM.hpp"

#include "helper/MemoryDebug.hpp"
#include "helper/PetscEigenBridge.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

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

PetscFOM_direct::PetscFOM_direct() : ROMBase({}) {
    name = "PetscFOM_direct";
    use_mor = false;
}

PetscFOM_direct::~PetscFOM_direct() {
    if (ksp_) KSPDestroy(&ksp_);
}

void PetscFOM_direct::set_petsc_system(Mat lhs, Vec rhs, Vec u) {
    current_lhs_ = lhs;
    current_rhs_ = rhs;
    current_u_ = u;
}

void PetscFOM_direct::set_factorization(Mat factor_in) {
    MemoryDebug::print("fom_direct_before_set_factorization");
    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    const int requested_factor_threads = factor_threads_for_mat(factor_in);
    const int requested_runtime_threads = runtime_mkl_threads_for_mat(factor_in);
    MKLThreadGuard mkl_threads(requested_factor_threads, requested_runtime_threads);
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PetscObjectComm((PetscObject)factor_in), &rank);
    // if (rank == 0) {
// #if defined(STAGE2_HAS_MKL_SERVICE)
//         std::cout << "  thread_debug factorization: requested_factor_threads="
//                   << requested_factor_threads
//                   << " requested_runtime_threads=" << requested_runtime_threads
//                   << " mkl_get_max_threads=" << mkl_get_max_threads()
//                   << "\n";
// #else
//         std::cout << "  thread_debug factorization: MKL service header not available"
//                   << " requested_factor_threads=" << requested_factor_threads
//                   << " requested_runtime_threads=" << requested_runtime_threads
//                   << "\n";
// #endif
    // }
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
#if defined(PETSC_HAVE_MKL_PARDISO)
    Mat F = nullptr;
    PCFactorGetMatrix(pc, &F);
    if (F) {
        MatMkl_PardisoSetCntl(F, 8, 0);
        if (rank == 0) std::cout << "  petsc_pardiso_debug: iparm(8) refinement_steps = 0\n";
    }
#endif
    MemoryDebug::print("fom_direct_after_KSPSetUp");
    PetscTime(&t1);
    last_factorization_time_ = static_cast<double>(t1 - t0);
}

KSP PetscFOM_direct::get_petsc_factorized_solver(Mat lhs) const {
    if (lhs == factor && ksp_) return ksp_;
    return nullptr;
}

void PetscFOM_direct::set_petsc_factorized_solver(Mat lhs, KSP ksp) {
    if (ksp) PetscObjectReference((PetscObject)ksp);
    if (ksp_) KSPDestroy(&ksp_);
    ksp_ = ksp;
    factor = lhs;
}

SolveStats PetscFOM_direct::FOM_main(const Eigen::SparseMatrix<double>&,
                                     const Eigen::VectorXd&,
                                     Eigen::VectorXd& u) {
    // const ROMContext& context = ctx;
    if (!current_lhs_ || !current_rhs_ || !current_u_) {
        throw std::runtime_error("PetscFOM_direct needs set_petsc_system before FOM_main");
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
    MemoryDebug::print("fom_direct_before_KSPSolve");
    KSPSolve(ksp_, current_rhs_, current_u_);
    MemoryDebug::print("fom_direct_after_KSPSolve");
    PetscInt its = 0;
    PetscReal rnorm = 0.0;
    KSPGetIterationNumber(ksp_, &its);
    KSPGetResidualNorm(ksp_, &rnorm);
    PetscTime(&t1);

    stats.ksp_iterations = static_cast<int>(its);
    stats.residual_norm = static_cast<double>(rnorm);
    stats.factorization_time = factorization_time;
    stats.solve_time = static_cast<double>(t1 - t0);
    u = petsc_vec_to_eigen(current_u_);
    return stats;
}

SolveStats PetscFOM_direct::main(const Eigen::SparseMatrix<double>& lhs,
                                 const Eigen::VectorXd& rhs,
                                 Eigen::VectorXd& u) {
    const ROMContext& context = ctx;
    const double t0 = PetscLogDouble{};
    PetscLogDouble t_start = 0.0;
    PetscLogDouble t_end = 0.0;
    (void)t0;
    PetscTime(&t_start);
    const SolveStats stats = FOM_main(lhs, rhs, u);
    PetscTime(&t_end);
    ROMInfo info;
    info.name = name;
    info.fom_used = true;
    info.accepted = true;
    info.ksp_iterations = stats.ksp_iterations;
    info.residual_norm = stats.residual_norm;
    info.timing_fom_factorization = stats.factorization_time;
    info.timing_fom_solve = stats.solve_time;
    info.timing_total = static_cast<double>(t_end - t_start);
    store_info(info, context);
    return stats;
}
