#include "ROM/Reduced_Basis_Algorithm/MyPOD_SVD1.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
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

struct RecoveredBasis {
    Eigen::MatrixXd V;
    Eigen::VectorXd s;
    Eigen::MatrixXd W;
};

RecoveredBasis recover_energy_after_normalization(const MyAlgebra::PODResult& pod,
                                                  const Eigen::VectorXd& snapshot_norm,
                                                  int r_new) {
    Eigen::MatrixXd V_new = pod.V.leftCols(r_new).eval();
    Eigen::VectorXd s_new = pod.s.head(r_new).eval();
    Eigen::MatrixXd W_new = pod.W.leftCols(r_new).eval();

    Eigen::MatrixXd coefficients_raw = s_new.asDiagonal() * W_new.transpose();
    for (Eigen::Index j = 0; j < coefficients_raw.cols(); ++j) {
        coefficients_raw.col(j) *= snapshot_norm[j];
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> raw_svd(coefficients_raw, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd s_energy = raw_svd.singularValues();
    if (s_energy.size() == 0 || s_energy.maxCoeff() <= 0.0) return {};

    const int r_keep = std::min({r_new,
                                 static_cast<int>(raw_svd.matrixU().cols()),
                                 static_cast<int>(s_energy.size())});
    if (r_keep <= 0) return {};

    RecoveredBasis out;
    out.V = V_new * raw_svd.matrixU().leftCols(r_keep);
    out.s = s_energy.head(r_keep);
    out.W = raw_svd.matrixV().leftCols(r_keep);
    return out;
}

} // namespace

void MyPOD_SVD1::update(const Eigen::VectorXd&, const ROMBase::ROMContext&) {
    update_flag = false;
    V.resize(0, 0);
    s.resize(0);
    W.resize(0, 0);
    if (max_rank <= 0 || snapshots.cols() == 0) return;
    if (!pod_solver) throw std::runtime_error("MyPOD_SVD1: pod_solver is not initialized");

    Eigen::VectorXd snapshot_norm(snapshots.cols());
    for (Eigen::Index j = 0; j < snapshots.cols(); ++j) snapshot_norm[j] = snapshots.col(j).norm();

    Eigen::MatrixXd X_new = snapshots;
    for (Eigen::Index j = 0; j < X_new.cols(); ++j) {
        if (snapshot_norm[j] > 1e-14) X_new.col(j) /= snapshot_norm[j];
        else X_new.col(j).setZero();
    }

    MyAlgebra::PODResult pod = pod_solver->compute(X_new, max_rank);
    if (pod.s_all.size() == 0 || pod.s_all.maxCoeff() <= 0.0) return;

    const int r_new = select_rank(pod.s_all, tol, energy_tol, max_rank, static_cast<int>(X_new.cols()));
    if (r_new <= 0) return;

    const RecoveredBasis recovered = recover_energy_after_normalization(pod, snapshot_norm, r_new);
    if (recovered.s.size() == 0) return;

    V = recovered.V;
    s = recovered.s;
    W = recovered.W;
    update_flag = true;
}
