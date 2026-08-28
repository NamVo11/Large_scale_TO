#include "ROM/ROM_method/ROM_Base.hpp"

#include "Analysis/Analysis.hpp"
#include "ROM/Reduced_Basis_Algorithm/UpdateBase.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef EIGEN_USE_LAPACKE
#include <mkl_cblas.h>
#include <mkl_spblas.h>
#endif

std::shared_ptr<FactorizedSolver> ROMBase::get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const {
    return fom_solver ? fom_solver->get_factorized_solver(lhs) : nullptr;
}

void ROMBase::set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                                    std::shared_ptr<FactorizedSolver> solver) {
    if (fom_solver) fom_solver->set_factorized_solver(lhs, std::move(solver));
}

KSP ROMBase::get_petsc_factorized_solver(Mat lhs) const {
    return fom_solver ? fom_solver->get_petsc_factorized_solver(lhs) : nullptr;
}

void ROMBase::set_petsc_factorized_solver(Mat lhs, KSP ksp) {
    if (fom_solver) fom_solver->set_petsc_factorized_solver(lhs, ksp);
}

void ROMBase::set_petsc_system(Mat lhs, Vec rhs, Vec u) {
    if (fom_solver) fom_solver->set_petsc_system(lhs, rhs, u);
}

const ROMBase::ROMContext& ROMBase::InitializeROMContext(const Analysis& analysis,
                                                int opt_iter,
                                                int solve_step,
                                                int time_step,
                                                double time,
                                                const std::string& role,
                                                const std::string& system_type,
                                                bool is_end_of_analysis) {
    ctx.analysis_id = analysis.id;
    ctx.opt_iter = opt_iter;
    ctx.solve_step = solve_step;
    ctx.time_step = time_step;
    ctx.time = time;
    ctx.role = role;
    ctx.system_type = system_type;
    ctx.is_end_of_analysis = is_end_of_analysis;
    return ctx;
}

void ROMBase::store_info(const ROMInfo& info, const ROMBase::ROMContext& context) {
    const std::size_t iter = static_cast<std::size_t>(std::max(0, context.opt_iter));
    const std::size_t step = static_cast<std::size_t>(std::max(0, context.time_step));
    if (ROM_info.size() <= iter) ROM_info.resize(iter + 1);
    if (ROM_info[iter].size() <= step) ROM_info[iter].resize(step + 1);
    ROM_info[iter][step] = info;
}

double ROMBase::now_time() {
    using Clock = std::chrono::steady_clock;
    static const auto t0 = Clock::now();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

void ROMBase::check_strategy_ready(const char* strategy_name) const {
    if (!fom_solver || !error_estimator) {
        throw std::runtime_error(std::string(strategy_name) + " is not fully initialized");
    }
    if (basis_list.size() < 2) {
        throw std::runtime_error(std::string(strategy_name) + " expects basis_list{1} and basis_list{2}");
    }
}

void ROMBase::clear_reduced_cache(Eigen::MatrixXd& AV_cache,
                                  const Eigen::SparseMatrix<double>*& lhs_cache,
                                  int& rank_cache) {
    A_r_cache.resize(0, 0);
    AV_cache.resize(0, 0);
    lhs_cache = nullptr;
    rank_cache = 0;
}

void ROMBase::rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                                    const Eigen::MatrixXd& V,
                                    Eigen::MatrixXd& AV_cache,
                                    const Eigen::SparseMatrix<double>*& lhs_cache,
                                    int& rank_cache,
                                    ROMInfo& info) {

    const double t_mkl0 = now_time();
    mkl_rebuild_reduced_cache(A, V, AV_cache);
    const double mkl_projection_time = now_time() - t_mkl0;

    // Compare Eigen and MKL projection times and results.
    // Keep this block disabled in production because it computes projection twice.
    /*
    const Eigen::MatrixXd AV_cache_mkl = AV_cache;
    const Eigen::MatrixXd A_r_cache_mkl = A_r_cache;
    const auto factorized_A_r_mkl = factorized_A_r;

    const double t_eigen0 = now_time();
    Eigen::MatrixXd AV_cache_eigen;
    eigen_rebuild_reduced_cache(A, V, AV_cache_eigen);
    Eigen::MatrixXd A_r_cache_eigen = A_r_cache;
    const double eigen_projection_time = now_time() - t_eigen0;
    
    const double AV_diff = (AV_cache_mkl - AV_cache_eigen).norm() / std::max(AV_cache_eigen.norm(), 1e-30);
    const double Ar_diff = (A_r_cache_mkl - A_r_cache_eigen).norm() / std::max(A_r_cache_eigen.norm(), 1e-30);
  
    std::cout << "  ROM projection compare[s]"
              << ": eigen=" << eigen_projection_time
              << " mkl=" << mkl_projection_time
              << " AV_rel_diff=" << AV_diff
              << " Ar_rel_diff=" << Ar_diff << "\n";

    AV_cache = AV_cache_mkl;
    A_r_cache = A_r_cache_mkl;
    factorized_A_r = factorized_A_r_mkl;
    */
    info.timing_projection += mkl_projection_time;
    lhs_cache = &A;
    rank_cache = static_cast<int>(V.cols());
    fom_flag = 0;
}

void ROMBase::eigen_rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                                          const Eigen::MatrixXd& V,
                                          Eigen::MatrixXd& AV_cache) {
    AV_cache.noalias() = A * V;
    A_r_cache.noalias() = V.transpose() * AV_cache;
    factorized_A_r.compute(A_r_cache);
}

void ROMBase::mkl_rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                                        const Eigen::MatrixXd& V,
                                        Eigen::MatrixXd& AV_cache) {
#ifdef EIGEN_USE_LAPACKE
    ensure_mkl_lhs_cache(A);

    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    AV_cache.resize(A.rows(), V.cols());
    for (Eigen::Index j = 0; j < V.cols(); ++j) {
        const sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                                                       1.0,
                                                       mkl_lhs_cache_,
                                                       descr,
                                                       V.col(j).data(),
                                                       0.0,
                                                       AV_cache.col(j).data());
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("ROMBase::mkl_rebuild_reduced_cache: mkl_sparse_d_mv failed");
        }
    }

    A_r_cache.resize(V.cols(), V.cols());
    cblas_dgemm(CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                static_cast<MKL_INT>(V.cols()),
                static_cast<MKL_INT>(V.cols()),
                static_cast<MKL_INT>(V.rows()),
                1.0,
                V.data(),
                static_cast<MKL_INT>(V.outerStride()),
                AV_cache.data(),
                static_cast<MKL_INT>(AV_cache.outerStride()),
                0.0,
                A_r_cache.data(),
                static_cast<MKL_INT>(A_r_cache.outerStride()));

    factorized_A_r.compute(A_r_cache);

#else
    eigen_rebuild_reduced_cache(A, V, AV_cache);
#endif
}

#ifdef EIGEN_USE_LAPACKE
void ROMBase::clear_mkl_lhs_cache() const {
    if (mkl_lhs_cache_) {
        mkl_sparse_destroy(mkl_lhs_cache_);
        mkl_lhs_cache_ = nullptr;
    }
    mkl_lhs_cache_key_ = nullptr;
    mkl_lhs_value_key_ = nullptr;
    mkl_lhs_rows_key_ = 0;
    mkl_lhs_cols_key_ = 0;
    mkl_lhs_nnz_key_ = 0;
    mkl_lhs_outer_.clear();
    mkl_lhs_inner_.clear();
}

void ROMBase::ensure_mkl_lhs_cache(const Eigen::SparseMatrix<double>& A) const {
    if (mkl_lhs_cache_key_ == &A &&
        mkl_lhs_value_key_ == A.valuePtr() &&
        mkl_lhs_rows_key_ == A.rows() &&
        mkl_lhs_cols_key_ == A.cols() &&
        mkl_lhs_nnz_key_ == A.nonZeros() &&
        mkl_lhs_cache_) return;
    if (!A.isCompressed()) {
        throw std::runtime_error("ROMBase::ensure_mkl_lhs_cache requires compressed Eigen sparse matrix");
    }

    clear_mkl_lhs_cache();

    const MKL_INT* outer_ptr = nullptr;
    const MKL_INT* inner_ptr = nullptr;
    using StorageIndex = Eigen::SparseMatrix<double>::StorageIndex;
    if constexpr (std::is_same_v<StorageIndex, MKL_INT>) {
        outer_ptr = reinterpret_cast<const MKL_INT*>(A.outerIndexPtr());
        inner_ptr = reinterpret_cast<const MKL_INT*>(A.innerIndexPtr());
    } else {
        mkl_lhs_outer_.resize(static_cast<std::size_t>(A.outerSize() + 1));
        for (Eigen::Index i = 0; i <= A.outerSize(); ++i) {
            mkl_lhs_outer_[static_cast<std::size_t>(i)] = static_cast<MKL_INT>(A.outerIndexPtr()[i]);
        }

        mkl_lhs_inner_.resize(static_cast<std::size_t>(A.nonZeros()));
        for (Eigen::Index i = 0; i < A.nonZeros(); ++i) {
            mkl_lhs_inner_[static_cast<std::size_t>(i)] = static_cast<MKL_INT>(A.innerIndexPtr()[i]);
        }
        outer_ptr = mkl_lhs_outer_.data();
        inner_ptr = mkl_lhs_inner_.data();
    }

    const sparse_status_t status = mkl_sparse_d_create_csc(&mkl_lhs_cache_,
                                                            SPARSE_INDEX_BASE_ZERO,
                                                            static_cast<MKL_INT>(A.rows()),
                                                            static_cast<MKL_INT>(A.cols()),
                                                            const_cast<MKL_INT*>(outer_ptr),
                                                            const_cast<MKL_INT*>(outer_ptr + 1),
                                                            const_cast<MKL_INT*>(inner_ptr),
                                                            const_cast<double*>(A.valuePtr()));
                                                        

    if (status != SPARSE_STATUS_SUCCESS) {
        clear_mkl_lhs_cache();
        throw std::runtime_error("ROMBase::ensure_mkl_lhs_cache: mkl_sparse_d_create_csc failed");
    }

    mkl_lhs_cache_key_ = &A;
    mkl_lhs_value_key_ = A.valuePtr();
    mkl_lhs_rows_key_ = A.rows();
    mkl_lhs_cols_key_ = A.cols();
    mkl_lhs_nnz_key_ = A.nonZeros();
}
#endif

void ROMBase::append_basis_debug(const char* phase,
                                 const ROMBase::ROMContext& context,
                                 int basis_id,
                                 const UpdateBase& basis) const {
    const char* dir = std::getenv("STAGE2_ROM_DEBUG_DIR");
    if (dir == nullptr || *dir == '\0') return;

    std::ostringstream path;
    path << dir << "/cpp_rom_basis_debug.csv";
    std::ifstream existing(path.str());
    const bool needs_header = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
    existing.close();

    std::ofstream out(path.str(), std::ios::app);
    if (!out) return;
    if (needs_header) out << "phase,role,iter,step,basis_id,rank,singular_values\n";

    out << phase << ',' << context.role << ',' << context.opt_iter << ','
        << context.time_step << ',' << basis_id << ',' << basis.V.cols() << ',';
    out << std::setprecision(17);
    for (Eigen::Index i = 0; i < basis.s.size(); ++i) {
        if (i > 0) out << ';';
        out << basis.s[i];
    }
    out << '\n';
}

void ROMBase::append_decision_debug(const ROMBase::ROMContext& context,
                                    int rank,
                                    double residual,
                                    double threshold,
                                    bool accepted) const {
    const char* dir = std::getenv("STAGE2_ROM_DECISION_DEBUG_DIR");
    if (dir == nullptr || *dir == '\0') return;

    std::ostringstream path;
    path << dir << "/cpp_rom_decision_debug.csv";
    std::ifstream existing(path.str());
    const bool needs_header = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
    existing.close();

    std::ofstream out(path.str(), std::ios::app);
    if (!out) return;
    if (needs_header) out << "role,iter,step,rank,residual,threshold,accepted\n";
    out << std::setprecision(17)
        << context.role << ',' << context.opt_iter << ',' << context.time_step << ','
        << rank << ',' << residual << ',' << threshold << ',' << (accepted ? 1 : 0) << '\n';
}
