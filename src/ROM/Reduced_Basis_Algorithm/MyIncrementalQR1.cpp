#include "ROM/Reduced_Basis_Algorithm/MyIncrementalQR1.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <limits>

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

void MyIncrementalQR1::reset()
{
    UpdateBase::reset();
    base_rank = 0;
    last_iter = 0;
    has_last_iter = false;
}

void MyIncrementalQR1::update(const Eigen::VectorXd &solution,
                              const ROMContext &context)
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

    const int drop_index = base_rank;
    const double t0 = wall_time();

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
        info.timing_projection += wall_time() - t0;
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
        info.timing_projection += wall_time() - t0;
        A_r_rank_cache = rank;
        return true;
    }

    return false;
}
