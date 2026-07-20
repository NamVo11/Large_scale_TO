#include "ROM/Reduced_Basis_Agorithm/MyIncrementalQR1.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <limits>

MyIncrementalQR1::MyIncrementalQR1(double e_rb_in, int max_rank_in)
    : UpdateBase(max_rank_in), e_rb(e_rb_in) {
    name = "MyIncrementalQR1";
}

void MyIncrementalQR1::reset() {
    UpdateBase::reset();
    base_rank = 0;
    last_iter = 0;
    has_last_iter = false;
}

void MyIncrementalQR1::update(const Eigen::VectorXd& solution, const ROMContext& context) {
    const Eigen::MatrixXd& V_old = getBasis(context);
    const Eigen::Index rows = solution.size();
    const int rank_old = static_cast<int>(V_old.cols());

    const int iter = context.opt_iter;
    if (!has_last_iter) {
        last_iter = iter;
        has_last_iter = true;
    } else if (iter != last_iter) {
        base_rank = rank_old;
        last_iter = iter;
    }

    base_rank = std::clamp(base_rank, 0, rank_old);
    int r_max = max_rank - base_rank;
    if (r_max < 0) r_max = 0;

    const double c_norm = solution.norm();
    if (c_norm < std::numeric_limits<double>::epsilon()) {
        setBasis(V_old, s);
        return;
    }

    const Eigen::VectorXd q = solution / c_norm;

    Eigen::VectorXd r;
    if (rank_old == 0) {
        r = q;
    } else {
        const Eigen::VectorXd coeff = V_old.transpose() * q;
        r = q - V_old * coeff;
    }

    const double nr = r.norm();
    const bool update_flag = nr > e_rb && r_max > 0;
    int recent_start = base_rank;
    int recent_rank_new = rank_old - base_rank + (update_flag ? 1 : 0);
    if (r_max > 0 && recent_rank_new > r_max) {
        recent_start += recent_rank_new - r_max;
        recent_rank_new = r_max;
    }
    const int recent_rank_ref = std::max(0, rank_old - recent_start);
    const int rank_new = base_rank + recent_rank_ref + (update_flag ? 1 : 0);

    Eigen::MatrixXd V_new(rows, rank_new);
    int col = 0;
    if (base_rank > 0) {
        V_new.leftCols(base_rank) = V_old.leftCols(base_rank);
        col += base_rank;
    }
    if (recent_rank_ref > 0) {
        V_new.middleCols(col, recent_rank_ref) = V_old.middleCols(recent_start, recent_rank_ref);
        col += recent_rank_ref;
    }
    if (update_flag) {
        V_new.col(col) = r / nr;
    }
    Eigen::VectorXd s_new = Eigen::VectorXd::Ones(V_new.cols());
    setBasis(std::move(V_new), std::move(s_new));
}

