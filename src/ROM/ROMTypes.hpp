#pragma once

#ifdef EIGEN_USE_LAPACKE
#include <mkl_spblas.h>
using MklSparseMatrixHandle = sparse_matrix_t;
#else
using MklSparseMatrixHandle = void*;
#endif

#include <string>

struct ROMInfo {
    std::string name;
    bool used_rom = false;
    bool rom_used = false;
    bool fom_used = false;
    bool accepted = false;
    int ksp_iterations = 0;
    double residual_norm = 0.0;
    int rank = 0;
    int rank_true = 0;
    double threshold = 0.0;
    double residual = 0.0;
    double residual_tol_eff = 0.0;
    double error_indicator = 0.0;
    double error = 0.0;
    double basis_error = 0.0;

    double timing_total = 0.0;
    double timing_petsc_to_eigen_vec = 0.0;
    double timing_petsc_to_eigen_mat = 0.0;
    double timing_projection = 0.0;
    double timing_reduced_rhs = 0.0;
    double timing_reduced_solve = 0.0;
    double timing_reconstruct = 0.0;
    double timing_residual = 0.0;
    double timing_fom_fallback = 0.0;
    double timing_fom_factorization = 0.0;
    double timing_fom_solve = 0.0;
    double timing_basis_update = 0.0;
    double timing_basis1_update = 0.0;
    double timing_online_basis_update = 0.0;
    double timing_end_basis_update = 0.0;
    double timing_basis_copy_reset = 0.0;
    double timing_snapshot = 0.0;
};

struct SolveStats {
    int ksp_iterations = 0;
    double residual_norm = 0.0;
    double factorization_time = 0.0;
    double solve_time = 0.0;
};
