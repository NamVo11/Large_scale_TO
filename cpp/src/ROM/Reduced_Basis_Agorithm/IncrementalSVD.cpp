#include "ROM/Reduced_Basis_Agorithm/IncrementalSVD.hpp"

#include <Eigen/QR>
#include <Eigen/SVD>

#ifdef EIGEN_USE_LAPACKE
#include <mkl_cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>

namespace stage2 {

IncrementalSVD::IncrementalSVD(int r_reset_in, double eps_svd_in, double tol_in, int max_rank_in)
    : UpdateBase(max_rank_in) {
    name = "IncrementalSVD";
    r_reset = r_reset_in;
    eps_svd = eps_svd_in;
    tol = tol_in;
    backend = backendFromEnvironment();
}

IncrementalSVD::Backend IncrementalSVD::backendFromEnvironment() {
    const char* value = std::getenv("STAGE2_ONLINE_BACKEND");
    if (!value) return Backend::MKL;
    const std::string backend_name(value);
    if (backend_name == "incremental_eigen" || backend_name == "eigen") return Backend::Eigen;
    if (backend_name == "incremental_mkl" || backend_name == "mkl") return Backend::MKL;
    return Backend::MKL;
}

void IncrementalSVD::reset() {
    UpdateBase::reset();
    update_count = 0;
}

int IncrementalSVD::resolveResetRank() const {
    return r_reset > 0 ? r_reset : max_rank;
}

bool IncrementalSVD::initialize(const Eigen::VectorXd& c) {
    const double c_norm = c.norm();
    if (c_norm < 1e-14) {
        V.resize(0, 0);
        s.resize(0);
        ++basis_version;
        return false;
    }
    V.resize(c.size(), 1);
    V.col(0) = c / c_norm;
    s.resize(1);
    s[0] = c_norm;
    ++basis_version;
    return true;
}

void IncrementalSVD::truncateState(Eigen::MatrixXd& V_new, Eigen::VectorXd& s_new) {
    if (s_new.size() == 0 || s_new.maxCoeff() <= 0.0) {
        V_new.resize(0, 0);
        s_new.resize(0);
        return;
    }
    const double cutoff = tol * s_new.maxCoeff();
    int keep_count = 0;
    for (Eigen::Index i = 0; i < s_new.size(); ++i) {
        if (s_new[i] > cutoff) ++keep_count;
    }
    const int r = std::min({max_rank, keep_count, static_cast<int>(V_new.cols()), static_cast<int>(s_new.size())});
    if (r <= 0) {
        V_new.resize(0, 0);
        s_new.resize(0);
        return;
    }
    V_new = V_new.leftCols(r).eval();
    s_new = s_new.head(r).eval();
}

bool IncrementalSVD::rankOneUpdate(const Eigen::VectorXd& c) {
    if (backend == Backend::MKL) return rankOneUpdateMKL(c);
    return rankOneUpdateEigen(c);
}

bool IncrementalSVD::rankOneUpdateEigen(const Eigen::VectorXd& c) {
    const int r_old = static_cast<int>(V.cols());
    Eigen::VectorXd ell;
    ell.noalias() = V.transpose() * c;
    Eigen::VectorXd h = c;
    h.noalias() -= V * ell;
    const double p = h.norm();
    const bool update_flag = p > eps_svd;

    Eigen::MatrixXd Q;
    if (update_flag) {
        Q.setZero(r_old + 1, r_old + 1);
        Q.topLeftCorner(r_old, r_old) = s.asDiagonal();
        Q.topRightCorner(r_old, 1) = ell;
        Q(r_old, r_old) = p;
    } else {
        Q.setZero(r_old, r_old + 1);
        Q.leftCols(r_old) = s.asDiagonal();
        Q.rightCols(1) = ell;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q, Eigen::ComputeThinU);
    Eigen::VectorXd s_new = svd.singularValues();
    const Eigen::MatrixXd& U = svd.matrixU();
    Eigen::MatrixXd V_new(V.rows(), U.cols());
    V_new.noalias() = V * U.topRows(r_old);
    if (update_flag) {
        V_new.noalias() += (h / p) * U.row(r_old);
    }
    truncateState(V_new, s_new);

    V = std::move(V_new);
    s = std::move(s_new);
    ++basis_version;
    return update_flag;
}

bool IncrementalSVD::rankOneUpdateMKL(const Eigen::VectorXd& c) {
#ifdef EIGEN_USE_LAPACKE
    const int rows = static_cast<int>(V.rows());
    const int r_old = static_cast<int>(V.cols());
    Eigen::VectorXd ell(r_old);
    cblas_dgemv(CblasColMajor, CblasTrans,
                rows, r_old, 1.0,
                V.data(), rows,
                c.data(), 1,
                0.0, ell.data(), 1);

    Eigen::VectorXd h = c;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                rows, r_old, -1.0,
                V.data(), rows,
                ell.data(), 1,
                1.0, h.data(), 1);
    const double p = cblas_dnrm2(rows, h.data(), 1);
    const bool update_flag = p > eps_svd;

    Eigen::MatrixXd Q;
    if (update_flag) {
        Q.setZero(r_old + 1, r_old + 1);
        Q.topLeftCorner(r_old, r_old) = s.asDiagonal();
        Q.topRightCorner(r_old, 1) = ell;
        Q(r_old, r_old) = p;
    } else {
        Q.setZero(r_old, r_old + 1);
        Q.leftCols(r_old) = s.asDiagonal();
        Q.rightCols(1) = ell;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q, Eigen::ComputeThinU);
    Eigen::VectorXd s_new = svd.singularValues();
    const Eigen::MatrixXd& U = svd.matrixU();
    Eigen::MatrixXd V_new(rows, U.cols());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                rows, static_cast<int>(U.cols()), r_old,
                1.0,
                V.data(), rows,
                U.data(), static_cast<int>(U.rows()),
                0.0,
                V_new.data(), rows);
    if (update_flag) {
        for (int j = 0; j < static_cast<int>(U.cols()); ++j) {
            cblas_daxpy(rows,
                        U(r_old, j) / p,
                        h.data(), 1,
                        V_new.col(j).data(), 1);
        }
    }
    truncateState(V_new, s_new);

    V = std::move(V_new);
    s = std::move(s_new);
    ++basis_version;
    return update_flag;
#else
    return rankOneUpdateEigen(c);
#endif
}

double IncrementalSVD::orthError(const Eigen::MatrixXd& basis) const {
    if (basis.cols() == 0) return 0.0;
    return (basis.transpose() * basis - Eigen::MatrixXd::Identity(basis.cols(), basis.cols())).norm();
}

void IncrementalSVD::reorthogonalizeIfNeeded() {
    if (V.cols() == 0 || orthError(V) <= orth_tol) return;

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(V);
    const Eigen::Index rows = V.rows();
    const Eigen::Index cols = V.cols();
    Eigen::MatrixXd Vq = qr.householderQ() * Eigen::MatrixXd::Identity(rows, cols);
    Eigen::MatrixXd R = qr.matrixQR().topRows(cols).template triangularView<Eigen::Upper>();
    Eigen::MatrixXd M = R * s.asDiagonal();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU);
    Eigen::MatrixXd V_new;
    V_new.noalias() = Vq * svd.matrixU();
    Eigen::VectorXd s_new = svd.singularValues();
    truncateState(V_new, s_new);
    V = std::move(V_new);
    s = std::move(s_new);
    ++basis_version;
}

void IncrementalSVD::update(const Eigen::VectorXd& solution, const ROMContext& context) {
    if (store_snapshots) addSnapshot(solution, context);
    const int rank_old = static_cast<int>(V.cols());
    const int r_reset_eff = resolveResetRank();
    const bool reset_flag = V.cols() == 0 || s.size() == 0 || rank_old >= r_reset_eff;
    bool update_flag = false;
    if (reset_flag) {
        update_flag = initialize(solution);
        update_count = 0;
    } else {
        update_flag = rankOneUpdate(solution);
    }
    if (update_flag) ++update_count;
    reorthogonalizeIfNeeded();
}

} // namespace stage2
