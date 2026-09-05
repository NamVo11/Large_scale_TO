#include "ROM/Reduced_Basis_Algorithm/MyPOD_SVD1.hpp"

#include <algorithm>
#include <stdexcept>

namespace {

int select_rank(const Eigen::VectorXd& singular_values,
                double tol,
                double energy_tol,
                int max_rank,
                int max_cols) {
    if (singular_values.size() == 0 || singular_values.maxCoeff() <= 0.0) return 0;
    int r = 0;
    if (energy_tol > 0.0) {
        const double total = singular_values.array().square().sum();
        double cumulative = 0.0;
        for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
            cumulative += singular_values[i] * singular_values[i];
            ++r;
            if (total > 0.0 && cumulative / total >= energy_tol) break;
        }
    } else {
        const double cutoff = tol * singular_values.maxCoeff();
        for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
            if (singular_values[i] > cutoff) ++r;
        }
    }
    return std::min({max_rank, r, static_cast<int>(singular_values.size()), max_cols});
}

} // namespace

void MyPOD_SVD::update(const Eigen::VectorXd&, const ROMBase::ROMContext&) {
    update_flag = false;
    V.resize(0, 0);
    s.resize(0);
    W.resize(0, 0);
    if (max_rank <= 0 || snapshots.cols() == 0) return;
    if (!pod_solver) throw std::runtime_error("MyPOD_SVD: pod_solver is not initialized");

    MyAlgebra::PODResult pod = pod_solver->compute(snapshots, max_rank);
    if (pod.s_all.size() == 0 || pod.s_all.maxCoeff() <= 0.0) return;

    const int r_new = select_rank(pod.s_all, tol, energy_tol, max_rank, static_cast<int>(snapshots.cols()));
    if (r_new <= 0) return;

    V = pod.V.leftCols(r_new).eval();
    s = pod.s_all.head(r_new).eval();
    W = pod.W.leftCols(r_new).eval();
    update_flag = true;
}
