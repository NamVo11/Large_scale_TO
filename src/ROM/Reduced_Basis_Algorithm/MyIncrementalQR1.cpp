#include "ROM/Reduced_Basis_Algorithm/MyIncrementalQR1.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef EIGEN_USE_LAPACKE
#include <mkl_cblas.h>
#include <mkl_spblas.h>
#endif

namespace {

double wall_time() {
    using Clock = std::chrono::steady_clock;
    static const auto t0 = Clock::now();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

} // namespace

MyIncrementalQR1::MyIncrementalQR1(double e_rb_in, int max_rank_in)
    : UpdateBase(max_rank_in), e_rb(e_rb_in)
{
    name = "MyIncrementalQR1";
}

MyIncrementalQR1::~MyIncrementalQR1()
{
#ifdef EIGEN_USE_LAPACKE
    clear_mkl_lhs_cache();
#endif
}

void MyIncrementalQR1::reset()
{
    UpdateBase::reset();
    base_rank = 0;
    last_iter = 0;
    has_last_iter = false;
}

void MyIncrementalQR1::update(const Eigen::VectorXd &solution,
                              const ROMBase::ROMContext &context)
{
    update_flag = false;
    const Eigen::MatrixXd &V_old = getBasis(context);
    const Eigen::Index rows = solution.size();
    const int rank_old = static_cast<int>(V_old.cols());

    const int iter = context.opt_iter;
    if (!has_last_iter)
    {
        last_iter = iter;
        has_last_iter = true;
    }
    else if (iter != last_iter)
    {
        base_rank = rank_old;
        last_iter = iter;
    }

    base_rank = std::clamp(base_rank, 0, rank_old);

    int r_max = max_rank - base_rank;
    if (r_max < 0)
        r_max = 0;

    const double c_norm = solution.norm();
    if (c_norm < std::numeric_limits<double>::epsilon())
    {
        return;
    }

    const Eigen::VectorXd q = solution / c_norm;

    const int recent_rank = rank_old - base_rank;
    const bool recent_full = r_max > 0 && recent_rank >= r_max;

    Eigen::VectorXd r = q;

    if (rank_old > 0)
    {
        const Eigen::VectorXd coeff = V_old.transpose() * q;

        r.noalias() -= V_old * coeff;

        if (recent_full)
        {
            // Hoàn lại đóng góp của recent mode cũ nhất.
            r.noalias() += V_old.col(base_rank) * coeff(base_rank);
        }
    }

    const double nr = r.norm();
    const bool changed_basis = nr > e_rb && r_max > 0;

    if (!changed_basis)
    {
        /*
         * Snapshot không mang hướng mới đủ lớn.
         *
         * Nếu recent basis đã đầy, mode cũ nhất chỉ mới được
         * "loại thử" trong phép tính residual; basis thực tế
         * vẫn được giữ nguyên.
         */
        return;
    }

    if (!recent_full)
    {
        /*
         * Recent basis chưa đầy:
         * thêm hướng mới vào cuối basis.
         */
        const int rank_new = rank_old + 1;

        Eigen::MatrixXd V_new(rows, rank_new);

        if (rank_old > 0)
        {
            V_new.leftCols(rank_old) = V_old;
        }

        V_new.col(rank_old) = r / nr;

        Eigen::VectorXd s_new = Eigen::VectorXd::Ones(rank_new);

        setBasis(std::move(V_new), std::move(s_new));
        update_flag = true;

        return;
    }

    /*
     * Recent basis đã đầy:
     *
     * V_old = [V_base | v_oldest | V_recent_keep]
     *
     * V_new = [V_base | V_recent_keep | v_new]
     *
     * Rank không đổi.
     */
    Eigen::MatrixXd V_new(rows, rank_old);

    int col = 0;

    if (base_rank > 0)
    {
        V_new.leftCols(base_rank) = V_old.leftCols(base_rank);

        col += base_rank;
    }

    const int recent_keep = recent_rank - 1;

    if (recent_keep > 0)
    {
        V_new.middleCols(col, recent_keep) = V_old.middleCols(base_rank + 1, recent_keep);

        col += recent_keep;
    }

    V_new.col(col) = r / nr;

    Eigen::VectorXd s_new = Eigen::VectorXd::Ones(rank_old);

    setBasis(std::move(V_new), std::move(s_new));
    update_flag = true;
}

bool MyIncrementalQR1::update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                                        const Eigen::MatrixXd& V_runtime,
                                                        Eigen::MatrixXd& A_r_cache,
                                                        Eigen::LLT<Eigen::MatrixXd>& factorized_A_r,
                                                        Eigen::MatrixXd& AV_cache,
                                                        int& A_r_rank_cache,
                                                        ROMInfo& info) const {
    const int rank = static_cast<int>(V_runtime.cols());
    const int old_rank = A_r_rank_cache;
    if (rank <= 0 || A_r_cache.rows() != old_rank || AV_cache.cols() != old_rank) return false;

    // Compare Eigen and MKL incremental projection times and results.
    // Keep this block disabled in production because it computes the update twice.
    /*
    const Eigen::MatrixXd A_r_old = A_r_cache;
    const Eigen::MatrixXd AV_old = AV_cache;

    const double t_mkl0 = wall_time();
    const bool mkl_ok = mkl_update_reduced_cache_incremental(A,
                                                             V_runtime,
                                                             A_r_cache,
                                                             factorized_A_r,
                                                             AV_cache,
                                                             A_r_rank_cache);
    const double mkl_time = wall_time() - t_mkl0;
    if (!mkl_ok) {
        A_r_cache = A_r_old;
        AV_cache = AV_old;
        factorized_A_r.compute(A_r_cache);
        return false;
    }

    const Eigen::MatrixXd A_r_mkl = A_r_cache;
    const Eigen::MatrixXd AV_mkl = AV_cache;

    A_r_cache = A_r_old;
    AV_cache = AV_old;
    factorized_A_r.compute(A_r_cache);
    A_r_rank_cache = old_rank;

    const double t_eigen0 = wall_time();
    const bool eigen_ok = eigen_update_reduced_cache_incremental(A,
                                                                 V_runtime,
                                                                 A_r_cache,
                                                                 factorized_A_r,
                                                                 AV_cache,
                                                                 A_r_rank_cache);
    const double eigen_time = wall_time() - t_eigen0;

    double AV_diff = 0.0;
    double Ar_diff = 0.0;
    if (eigen_ok) {
        AV_diff = (AV_mkl - AV_cache).norm() / std::max(AV_cache.norm(), 1e-30);
        Ar_diff = (A_r_mkl - A_r_cache).norm() / std::max(A_r_cache.norm(), 1e-30);
    }

    std::cout << "  ROM incremental projection compare[s]"
              << ": eigen=" << eigen_time
              << " mkl=" << mkl_time
              << " eigen_ok=" << (eigen_ok ? 1 : 0)
              << " AV_rel_diff=" << AV_diff
              << " Ar_rel_diff=" << Ar_diff << "\n";

    A_r_cache = A_r_mkl;
    AV_cache = AV_mkl;
    factorized_A_r.compute(A_r_cache);
    A_r_rank_cache = rank;

    info.timing_projection += mkl_time;
    return true;
    */

    const double t0 = wall_time();
    const bool ok = mkl_update_reduced_cache_incremental(A,
                                                         V_runtime,
                                                         A_r_cache,
                                                         factorized_A_r,
                                                         AV_cache,
                                                         A_r_rank_cache);
    if (ok) info.timing_projection += wall_time() - t0;
    return ok;
}

bool MyIncrementalQR1::eigen_update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                                              const Eigen::MatrixXd& V_runtime,
                                                              Eigen::MatrixXd& A_r_cache,
                                                              Eigen::LLT<Eigen::MatrixXd>& factorized_A_r,
                                                              Eigen::MatrixXd& AV_cache,
                                                              int& A_r_rank_cache) const {
    const int rank = static_cast<int>(V_runtime.cols());
    const int old_rank = A_r_rank_cache;
    if (rank <= 0 || A_r_cache.rows() != old_rank || AV_cache.cols() != old_rank) return false;

    const int drop_index = base_rank;

    auto append_column = [&](const Eigen::MatrixXd& A_r_base,
                             const Eigen::MatrixXd& AV_base,
                             const Eigen::VectorXd& v_new,
                             int new_rank) {
        const Eigen::VectorXd Av_new = A * v_new;
        Eigen::MatrixXd A_r_new(new_rank, new_rank);
        if (new_rank > 1) {
            A_r_new.topLeftCorner(new_rank - 1, new_rank - 1) = A_r_base;
            A_r_new.topRightCorner(new_rank - 1, 1) = V_runtime.leftCols(new_rank - 1).transpose() * Av_new;
            A_r_new.bottomLeftCorner(1, new_rank - 1) = v_new.transpose() * AV_base;
        }
        A_r_new(new_rank - 1, new_rank - 1) = v_new.dot(Av_new);

        Eigen::MatrixXd AV_new(AV_base.rows(), new_rank);
        if (new_rank > 1) AV_new.leftCols(new_rank - 1) = AV_base;
        AV_new.col(new_rank - 1) = Av_new;
        A_r_cache = std::move(A_r_new);
        factorized_A_r.compute(A_r_cache);
        AV_cache = std::move(AV_new);
    };

    if (rank == old_rank + 1) {
        append_column(A_r_cache, AV_cache, V_runtime.col(rank - 1), rank);
        A_r_rank_cache = rank;
        return true;
    }

    if (drop_index >= 0 && drop_index < old_rank &&
        (rank == old_rank || rank == old_rank - 1)) {
        const int kept_rank = old_rank - 1;
        Eigen::MatrixXd A_r_kept(kept_rank, kept_rank);
        Eigen::MatrixXd AV_kept(AV_cache.rows(), kept_rank);
        int ii = 0;
        for (int i = 0; i < old_rank; ++i) {
            if (i == drop_index) continue;
            AV_kept.col(ii) = AV_cache.col(i);
            int jj = 0;
            for (int j = 0; j < old_rank; ++j) {
                if (j == drop_index) continue;
                A_r_kept(ii, jj) = A_r_cache(i, j);
                ++jj;
            }
            ++ii;
        }

        if (rank == kept_rank) {
            A_r_cache = std::move(A_r_kept);
            factorized_A_r.compute(A_r_cache);
            AV_cache = std::move(AV_kept);
        } else {
            append_column(A_r_kept, AV_kept, V_runtime.col(rank - 1), rank);
        }
        A_r_rank_cache = rank;
        return true;
    }

    return false;
}

bool MyIncrementalQR1::mkl_update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                                            const Eigen::MatrixXd& V_runtime,
                                                            Eigen::MatrixXd& A_r_cache,
                                                            Eigen::LLT<Eigen::MatrixXd>& factorized_A_r,
                                                            Eigen::MatrixXd& AV_cache,
                                                            int& A_r_rank_cache) const {
#ifdef EIGEN_USE_LAPACKE
    ensure_mkl_lhs_cache(A);

    const int rank = static_cast<int>(V_runtime.cols());
    const int old_rank = A_r_rank_cache;
    if (rank <= 0 || A_r_cache.rows() != old_rank || AV_cache.cols() != old_rank) return false;

    const int drop_index = base_rank;

    auto append_column = [&](const Eigen::MatrixXd& A_r_base,
                             const Eigen::MatrixXd& AV_base,
                             const Eigen::VectorXd& v_new,
                             int new_rank) {
        Eigen::VectorXd Av_new(A.rows());
        matrix_descr descr{};
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        const sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                                                       1.0,
                                                       mkl_lhs_cache_,
                                                       descr,
                                                       v_new.data(),
                                                       0.0,
                                                       Av_new.data());
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("MyIncrementalQR1::mkl_update_reduced_cache_incremental: mkl_sparse_d_mv failed");
        }

        Eigen::MatrixXd A_r_new(new_rank, new_rank);
        if (new_rank > 1) {
            const int kept_rank = new_rank - 1;
            A_r_new.topLeftCorner(kept_rank, kept_rank) = A_r_base;

            cblas_dgemv(CblasColMajor,
                        CblasTrans,
                        static_cast<MKL_INT>(V_runtime.rows()),
                        static_cast<MKL_INT>(kept_rank),
                        1.0,
                        V_runtime.data(),
                        static_cast<MKL_INT>(V_runtime.outerStride()),
                        Av_new.data(),
                        1,
                        0.0,
                        A_r_new.col(new_rank - 1).data(),
                        1);

            cblas_dgemv(CblasColMajor,
                        CblasTrans,
                        static_cast<MKL_INT>(AV_base.rows()),
                        static_cast<MKL_INT>(kept_rank),
                        1.0,
                        AV_base.data(),
                        static_cast<MKL_INT>(AV_base.outerStride()),
                        v_new.data(),
                        1,
                        0.0,
                        A_r_new.row(new_rank - 1).data(),
                        static_cast<MKL_INT>(A_r_new.outerStride()));
        }
        A_r_new(new_rank - 1, new_rank - 1) =
            cblas_ddot(static_cast<MKL_INT>(v_new.size()), v_new.data(), 1, Av_new.data(), 1);

        Eigen::MatrixXd AV_new(AV_base.rows(), new_rank);
        if (new_rank > 1) AV_new.leftCols(new_rank - 1) = AV_base;
        AV_new.col(new_rank - 1) = Av_new;
        A_r_cache = std::move(A_r_new);
        factorized_A_r.compute(A_r_cache);
        AV_cache = std::move(AV_new);
    };

    if (rank == old_rank + 1) {
        append_column(A_r_cache, AV_cache, V_runtime.col(rank - 1), rank);
        A_r_rank_cache = rank;
        return true;
    }

    if (drop_index >= 0 && drop_index < old_rank &&
        (rank == old_rank || rank == old_rank - 1)) {
        const int kept_rank = old_rank - 1;
        Eigen::MatrixXd A_r_kept(kept_rank, kept_rank);
        Eigen::MatrixXd AV_kept(AV_cache.rows(), kept_rank);
        int ii = 0;
        for (int i = 0; i < old_rank; ++i) {
            if (i == drop_index) continue;
            AV_kept.col(ii) = AV_cache.col(i);
            int jj = 0;
            for (int j = 0; j < old_rank; ++j) {
                if (j == drop_index) continue;
                A_r_kept(ii, jj) = A_r_cache(i, j);
                ++jj;
            }
            ++ii;
        }

        if (rank == kept_rank) {
            A_r_cache = std::move(A_r_kept);
            factorized_A_r.compute(A_r_cache);
            AV_cache = std::move(AV_kept);
        } else {
            append_column(A_r_kept, AV_kept, V_runtime.col(rank - 1), rank);
        }
        A_r_rank_cache = rank;
        return true;
    }

    return false;
#else
    return eigen_update_reduced_cache_incremental(A,
                                                  V_runtime,
                                                  A_r_cache,
                                                  factorized_A_r,
                                                  AV_cache,
                                                  A_r_rank_cache);
#endif
}

#ifdef EIGEN_USE_LAPACKE
void MyIncrementalQR1::clear_mkl_lhs_cache() const {
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

void MyIncrementalQR1::ensure_mkl_lhs_cache(const Eigen::SparseMatrix<double>& A) const {
    if (mkl_lhs_cache_key_ == &A &&
        mkl_lhs_value_key_ == A.valuePtr() &&
        mkl_lhs_rows_key_ == A.rows() &&
        mkl_lhs_cols_key_ == A.cols() &&
        mkl_lhs_nnz_key_ == A.nonZeros() &&
        mkl_lhs_cache_) return;
    if (!A.isCompressed()) {
        throw std::runtime_error("MyIncrementalQR1::ensure_mkl_lhs_cache requires compressed Eigen sparse matrix");
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
        throw std::runtime_error("MyIncrementalQR1::ensure_mkl_lhs_cache: mkl_sparse_d_create_csc failed");
    }

    mkl_lhs_cache_key_ = &A;
    mkl_lhs_value_key_ = A.valuePtr();
    mkl_lhs_rows_key_ = A.rows();
    mkl_lhs_cols_key_ = A.cols();
    mkl_lhs_nnz_key_ = A.nonZeros();
}
#endif
