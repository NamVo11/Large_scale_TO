#include "ROM/Common/Reduced_Basis_Agorithm/MyIncrementalQR1.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <limits>

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
