#include "ROM/Reduced_Basis_Agorithm/MyPOD_SVD1.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#ifdef EIGEN_USE_LAPACKE
#include <mkl_cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <stdexcept>

namespace stage2 {
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

void compute_normalized_pod_bdcsvd(const Eigen::MatrixXd& X_new,
                                   int r_new,
                                   Eigen::MatrixXd& V_new,
                                   Eigen::VectorXd& s_new,
                                   Eigen::MatrixXd& W_new,
                                   Eigen::VectorXd& s_all) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(X_new.rows(), X_new.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.compute(X_new);
    s_all = svd.singularValues();
    V_new = svd.matrixU().leftCols(r_new);
    s_new = s_all.head(r_new);
    W_new = svd.matrixV().leftCols(r_new);
}

bool compute_normalized_pod_snapshot(const Eigen::MatrixXd& X_new,
                                     int r_new,
                                     Eigen::MatrixXd& V_new,
                                     Eigen::VectorXd& s_new,
                                     Eigen::MatrixXd& W_new,
                                     Eigen::VectorXd& s_all) {
    if (X_new.cols() == 0) return false;

    Eigen::MatrixXd C = X_new.transpose() * X_new;
    C = 0.5 * (C + C.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C);
    if (eig.info() != Eigen::Success) return false;

    const Eigen::VectorXd evals = eig.eigenvalues();
    const Eigen::MatrixXd evecs = eig.eigenvectors();
    const Eigen::Index n = evals.size();
    s_all.resize(n);
    W_new.resize(X_new.cols(), r_new);
    s_new.resize(r_new);

    for (Eigen::Index k = 0; k < n; ++k) {
        const Eigen::Index idx = n - 1 - k;
        s_all[k] = std::sqrt(std::max(0.0, evals[idx]));
        if (k < r_new) {
            W_new.col(k) = evecs.col(idx);
            s_new[k] = s_all[k];
        }
    }

    V_new = X_new * W_new;
    for (Eigen::Index j = 0; j < r_new; ++j) {
        if (s_new[j] > 1e-14) {
            V_new.col(j) /= s_new[j];
        } else {
            V_new.col(j).setZero();
        }
    }
    return true;
}

bool compute_normalized_pod_snapshot_mkl(const Eigen::MatrixXd& X_new,
                                         int r_new,
                                         Eigen::MatrixXd& V_new,
                                         Eigen::VectorXd& s_new,
                                         Eigen::MatrixXd& W_new,
                                         Eigen::VectorXd& s_all) {
#ifdef EIGEN_USE_LAPACKE
    if (X_new.cols() == 0) return false;

    const int m = static_cast<int>(X_new.rows());
    const int n = static_cast<int>(X_new.cols());
    Eigen::MatrixXd C(n, n);
    C.setZero();
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                n, m, 1.0, X_new.data(), m, 0.0, C.data(), n);
    for (int j = 0; j < n; ++j) {
        for (int i = j + 1; i < n; ++i) C(i, j) = C(j, i);
    }

    Eigen::VectorXd evals(n);
    const int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', n, C.data(), n, evals.data());
    if (info != 0) return false;

    s_all.resize(n);
    W_new.resize(n, r_new);
    s_new.resize(r_new);
    for (int k = 0; k < n; ++k) {
        const int idx = n - 1 - k;
        s_all[k] = std::sqrt(std::max(0.0, evals[idx]));
        if (k < r_new) {
            W_new.col(k) = C.col(idx);
            s_new[k] = s_all[k];
        }
    }

    V_new.resize(m, r_new);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, r_new, n, 1.0,
                X_new.data(), m,
                W_new.data(), n,
                0.0, V_new.data(), m);
    for (int j = 0; j < r_new; ++j) {
        if (s_new[j] > 1e-14) {
            V_new.col(j) /= s_new[j];
        } else {
            V_new.col(j).setZero();
        }
    }
    return true;
#else
    (void)X_new;
    (void)r_new;
    (void)V_new;
    (void)s_new;
    (void)W_new;
    (void)s_all;
    return false;
#endif
}

} // namespace

MyPOD_SVD1::MyPOD_SVD1(double snapshot_window_in,
                       double tol_in,
                       int max_rank_in,
                       double energy_tol_in)
    : UpdateBase(max_rank_in) {
    name = "MyPOD_SVD_recover_energy_after_normalization";
    snapshot_window = snapshot_window_in;
    tol = tol_in;
    energy_tol = energy_tol_in;
    backend = backendFromEnvironment();
}

MyPOD_SVD1::Backend MyPOD_SVD1::backendFromEnvironment() {
    const char* value = std::getenv("STAGE2_POD_BACKEND");
    if (!value) return Backend::SnapshotMKL;
    const std::string backend_name(value);
    if (backend_name == "bdcsvd" || backend_name == "EigenBDCSVD") return Backend::EigenBDCSVD;
    if (backend_name == "snapshot_eigen" || backend_name == "SnapshotEigen") return Backend::SnapshotEigen;
    if (backend_name == "snapshot_mkl" || backend_name == "SnapshotMKL") return Backend::SnapshotMKL;
    return Backend::SnapshotEigen;
}

void MyPOD_SVD1::update(const Eigen::VectorXd&, const ROMContext&) {
    V.resize(0, 0);
    s.resize(0);
    W.resize(0, 0);
    ++basis_version;
    if (max_rank <= 0 || snapshots.cols() == 0) return;

    Eigen::VectorXd snapshot_norm(snapshots.cols());
    for (Eigen::Index j = 0; j < snapshots.cols(); ++j) snapshot_norm[j] = snapshots.col(j).norm();

    Eigen::MatrixXd X_new = snapshots;
    for (Eigen::Index j = 0; j < X_new.cols(); ++j) {
        if (snapshot_norm[j] > 1e-14) X_new.col(j) /= snapshot_norm[j];
        else X_new.col(j).setZero();
    }

    Eigen::VectorXd s_all;
    Eigen::MatrixXd V_new;
    Eigen::VectorXd s_new;
    Eigen::MatrixXd W_new;
    bool basis_precomputed = false;

    if (backend == Backend::EigenBDCSVD) {
        Eigen::BDCSVD<Eigen::MatrixXd> svd(X_new.rows(), X_new.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        svd.compute(X_new);
        s_all = svd.singularValues();
    } else if (backend == Backend::SnapshotMKL) {
        if (!compute_normalized_pod_snapshot_mkl(X_new, std::min(max_rank, static_cast<int>(X_new.cols())),
                                                 V_new, s_new, W_new, s_all)) {
            backend = Backend::SnapshotEigen;
        } else {
            basis_precomputed = true;
        }
    }
    if (backend == Backend::SnapshotEigen) {
        Eigen::MatrixXd C = X_new.transpose() * X_new;
        C = 0.5 * (C + C.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C, Eigen::EigenvaluesOnly);
        if (eig.info() != Eigen::Success) return;
        const Eigen::VectorXd evals = eig.eigenvalues();
        s_all.resize(evals.size());
        for (Eigen::Index k = 0; k < evals.size(); ++k) {
            const Eigen::Index idx = evals.size() - 1 - k;
            s_all[k] = std::sqrt(std::max(0.0, evals[idx]));
        }
    }
    if (s_all.size() == 0 || s_all.maxCoeff() <= 0.0) return;

    const int r_new = select_rank(s_all, tol, energy_tol, max_rank, static_cast<int>(X_new.cols()));
    if (r_new <= 0) return;

    if (basis_precomputed) {
        V_new = V_new.leftCols(r_new).eval();
        s_new = s_new.head(r_new).eval();
        W_new = W_new.leftCols(r_new).eval();
    } else if (backend == Backend::EigenBDCSVD) {
        compute_normalized_pod_bdcsvd(X_new, r_new, V_new, s_new, W_new, s_all);
    } else if (backend == Backend::SnapshotMKL) {
        if (!compute_normalized_pod_snapshot_mkl(X_new, r_new, V_new, s_new, W_new, s_all)) {
            compute_normalized_pod_snapshot(X_new, r_new, V_new, s_new, W_new, s_all);
        }
    } else if (!compute_normalized_pod_snapshot(X_new, r_new, V_new, s_new, W_new, s_all)) {
        compute_normalized_pod_bdcsvd(X_new, r_new, V_new, s_new, W_new, s_all);
    }

    Eigen::MatrixXd coefficients_raw = s_new.asDiagonal() * W_new.transpose();
    for (Eigen::Index j = 0; j < coefficients_raw.cols(); ++j) {
        coefficients_raw.col(j) *= snapshot_norm[j];
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> raw_svd(coefficients_raw,
                                              Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd s_energy = raw_svd.singularValues();
    if (s_energy.size() == 0 || s_energy.maxCoeff() <= 0.0) return;

    const int r_keep = std::min({r_new,
                                 static_cast<int>(raw_svd.matrixU().cols()),
                                 static_cast<int>(s_energy.size())});
    if (r_keep <= 0) return;

    V = V_new * raw_svd.matrixU().leftCols(r_keep);
    s = s_energy.head(r_keep);
    W = raw_svd.matrixV().leftCols(r_keep);
    ++basis_version;
}

} // namespace stage2
