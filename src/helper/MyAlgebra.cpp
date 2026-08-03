#include "helper/MyAlgebra.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#ifdef EIGEN_USE_LAPACKE
#include <mkl_cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace MyAlgebra {
namespace {

int bounded_rank(const Eigen::MatrixXd& X, int max_rank) {
    return std::min({max_rank, static_cast<int>(X.rows()), static_cast<int>(X.cols())});
}

void check_input(const Eigen::MatrixXd& X, int max_rank, const char* solver_name) {
    if (X.cols() == 0 || X.rows() == 0) throw std::runtime_error(std::string(solver_name) + ": empty input matrix");
    if (max_rank <= 0) throw std::runtime_error(std::string(solver_name) + ": max_rank must be positive");
}

PODResult compute_snapshot_from_eigenvectors(const Eigen::MatrixXd& X,
                                             const Eigen::MatrixXd& eigenvectors,
                                             const Eigen::VectorXd& eigenvalues,
                                             int max_rank) {
    const int n = static_cast<int>(eigenvalues.size());
    const int r = std::min(max_rank, n);

    PODResult result;
    result.s_all.resize(n);
    result.W.resize(X.cols(), r);
    result.s.resize(r);

    for (int k = 0; k < n; ++k) {
        const int idx = n - 1 - k;
        result.s_all[k] = std::sqrt(std::max(0.0, eigenvalues[idx]));
        if (k < r) {
            result.W.col(k) = eigenvectors.col(idx);
            result.s[k] = result.s_all[k];
        }
    }

    result.V = X * result.W;
    for (int j = 0; j < r; ++j) {
        if (result.s[j] > 1e-14) result.V.col(j) /= result.s[j];
        else result.V.col(j).setZero();
    }

    return result;
}

} // namespace

std::vector<int> to_ints(const std::vector<PetscInt>& in) {
    std::vector<int> out(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) out[i] = static_cast<int>(in[i]);
    return out;
}

namespace Petsc {

void apply_dirichlet_to_matrix(Mat A, const std::vector<int>& fixed, double diag) {
    if (!A || fixed.empty()) return;
    std::vector<PetscInt> ids;
    ids.reserve(fixed.size());
    for (int d : fixed) ids.push_back(static_cast<PetscInt>(d));
    IS is = nullptr;
    ISCreateGeneral(PetscObjectComm(reinterpret_cast<PetscObject>(A)),
                    static_cast<PetscInt>(ids.size()),
                    ids.data(),
                    PETSC_COPY_VALUES,
                    &is);
    MatZeroRowsColumnsIS(A, is, diag, nullptr, nullptr);
    ISDestroy(&is);
}

} // namespace Petsc

namespace EigenOps {

void apply_dirichlet_to_matrix(Eigen::SparseMatrix<double>& A, const std::vector<int>& fixed, double diag) {
    if (fixed.empty()) return;
    A.makeCompressed();
    for (int d : fixed) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, d); it; ++it) {
            if (it.row() != d) it.valueRef() = 0.0;
        }
    }
    for (int col = 0; col < A.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, col); it; ++it) {
            if (it.row() != col && std::binary_search(fixed.begin(), fixed.end(), static_cast<int>(it.row()))) {
                it.valueRef() = 0.0;
            }
        }
    }
    for (int d : fixed) A.coeffRef(d, d) = diag;
    A.prune(0.0);
    A.makeCompressed();
}

} // namespace EigenOps

PODResult MyEigenBDCSVD::compute(const Eigen::MatrixXd& X, int max_rank) const {
    check_input(X, max_rank, "MyEigenBDCSVD");
    const int r = bounded_rank(X, max_rank);

    using BDCSVDThin = Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV>;

    BDCSVDThin svd(X);

    PODResult result;
    result.s_all = svd.singularValues();
    result.V = svd.matrixU().leftCols(r);
    result.s = result.s_all.head(r);
    result.W = svd.matrixV().leftCols(r);
    return result;
}

PODResult MyEigenSnapshot::compute(const Eigen::MatrixXd& X, int max_rank) const {
    check_input(X, max_rank, "MyEigenSnapshot");
    Eigen::MatrixXd C = X.transpose() * X;
    C = 0.5 * (C + C.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C);
    if (eig.info() != Eigen::Success) throw std::runtime_error("MyEigenSnapshot: SelfAdjointEigenSolver failed");
    return compute_snapshot_from_eigenvectors(X, eig.eigenvectors(), eig.eigenvalues(), max_rank);
}

PODResult MyEigenJacobiSVD::compute(const Eigen::MatrixXd& X, int max_rank) const {
    check_input(X, max_rank, "MyEigenJacobiSVD");
    const int r = bounded_rank(X, max_rank);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    PODResult result;
    result.s_all = svd.singularValues();
    result.V = svd.matrixU().leftCols(r);
    result.s = result.s_all.head(r);
    result.W = svd.matrixV().leftCols(r);
    return result;
}

PODResult MyLapackSnapshot::compute(const Eigen::MatrixXd& X, int max_rank) const {
    check_input(X, max_rank, "MyLapackSnapshot");
#ifdef EIGEN_USE_LAPACKE
    const int m = static_cast<int>(X.rows());
    const int n = static_cast<int>(X.cols());
    Eigen::MatrixXd C(n, n);
    C.setZero();
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, n, m, 1.0, X.data(), m, 0.0, C.data(), n);
    for (int j = 0; j < n; ++j) {
        for (int i = j + 1; i < n; ++i) C(i, j) = C(j, i);
    }
    Eigen::VectorXd evals(n);
    const int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', n, C.data(), n, evals.data());
    if (info != 0) throw std::runtime_error("MyLapackSnapshot: LAPACKE_dsyevd failed");
    return compute_snapshot_from_eigenvectors(X, C, evals, max_rank);
#else
    (void)X;
    (void)max_rank;
    throw std::runtime_error("MyLapackSnapshot requires EIGEN_USE_LAPACKE");
#endif
}

PODResult MyLapackDGESDD::compute(const Eigen::MatrixXd& X, int max_rank) const {
    check_input(X, max_rank, "MyLapackDGESDD");
#ifdef EIGEN_USE_LAPACKE
    const int m = static_cast<int>(X.rows());
    const int n = static_cast<int>(X.cols());
    const int r = bounded_rank(X, max_rank);
    const int min_mn = std::min(m, n);
    Eigen::MatrixXd A = X;
    Eigen::VectorXd s_all(min_mn);
    Eigen::MatrixXd U(m, min_mn);
    Eigen::MatrixXd VT(min_mn, n);
    const int info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S', m, n, A.data(), m, s_all.data(), U.data(), m, VT.data(), min_mn);
    if (info != 0) throw std::runtime_error("MyLapackDGESDD: LAPACKE_dgesdd failed");
    PODResult result;
    result.s_all = s_all;
    result.V = U.leftCols(r);
    result.s = result.s_all.head(r);
    result.W = VT.topRows(r).transpose();
    return result;
#else
    (void)X;
    (void)max_rank;
    throw std::runtime_error("MyLapackDGESDD requires EIGEN_USE_LAPACKE");
#endif
}

std::unique_ptr<MyPODBase> create_pod_solver(const std::string& backend_name) {
    if (backend_name == "bdcsvd" || backend_name == "EigenBDCSVD" || backend_name == "eigen_bdcsvd" || backend_name == "MyEigenBDCSVD") return std::make_unique<MyEigenBDCSVD>();
    if (backend_name == "snapshot_eigen" || backend_name == "SnapshotEigen" || backend_name == "eigen_snapshot" || backend_name == "MyEigenSnapshot") return std::make_unique<MyEigenSnapshot>();
    if (backend_name == "jacobi" || backend_name == "jacobisvd" || backend_name == "eigen_jacobi" || backend_name == "MyEigenJacobiSVD") return std::make_unique<MyEigenJacobiSVD>();
    if (backend_name == "snapshot_mkl" || backend_name == "SnapshotMKL" || backend_name == "lapack_snapshot" || backend_name == "MyLapackSnapshot") return std::make_unique<MyLapackSnapshot>();
    if (backend_name == "lapack_dgesdd" || backend_name == "dgesdd" || backend_name == "lapack_bdcsvd" || backend_name == "MyLapackDGESDD") return std::make_unique<MyLapackDGESDD>();
    throw std::runtime_error("Unknown POD backend: " + backend_name);
}

std::unique_ptr<MyPODBase> create_pod_solver_from_environment() {
    const char* value = std::getenv("STAGE2_POD_BACKEND");
    return create_pod_solver(value ? std::string(value) : std::string("snapshot_mkl"));
}

// =========================================Incremental SVD==========================================

bool MyISVDBase::initialize(Eigen::MatrixXd& V, Eigen::VectorXd& s, const Eigen::VectorXd& c) const {
    const double c_norm = c.norm();
    if (c_norm < 1e-14) {
        V.resize(0, 0);
        s.resize(0);
        return false;
    }
    V.resize(c.size(), 1);
    V.col(0) = c / c_norm;
    s.resize(1);
    s[0] = c_norm;
    return true;
}

void MyISVDBase::truncateState(Eigen::MatrixXd& V_new, Eigen::VectorXd& s_new) const {
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

double MyISVDBase::orthError(const Eigen::MatrixXd& basis) const {
    if (basis.cols() == 0) return 0.0;
    return (basis.transpose() * basis - Eigen::MatrixXd::Identity(basis.cols(), basis.cols())).norm();
}

bool MyISVDBase::reorthogonalizeIfNeeded(Eigen::MatrixXd& V, Eigen::VectorXd& s) const {
    if (V.cols() == 0 || orthError(V) <= orth_tol) return false;

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
    return true;
}

bool MyEigenISVD::update(Eigen::MatrixXd& V, Eigen::VectorXd& s, const Eigen::VectorXd& c) const {
    const int r_old = static_cast<int>(V.cols());
    Eigen::VectorXd ell;
    ell.noalias() = V.transpose() * c;
    Eigen::VectorXd h = c;
    h.noalias() -= V * ell;
    const double p = h.norm();
    const bool changed_basis = p > eps_svd;

    Eigen::MatrixXd Q;
    if (changed_basis) {
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
    if (changed_basis) {
        V_new.noalias() += (h / p) * U.row(r_old);
    }
    truncateState(V_new, s_new);

    V = std::move(V_new);
    s = std::move(s_new);
    return true;
}

bool MyLapackISVD::update(Eigen::MatrixXd& V, Eigen::VectorXd& s, const Eigen::VectorXd& c) const {
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
    const bool changed_basis = p > eps_svd;

    Eigen::MatrixXd Q;
    if (changed_basis) {
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
    if (changed_basis) {
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
    return true;
#else
    MyEigenISVD fallback(max_rank, eps_svd, tol, orth_tol);
    return fallback.update(V, s, c);
#endif
}

std::unique_ptr<MyISVDBase> create_isvd_solver(const std::string& backend_name,
                                               int max_rank,
                                               double eps_svd,
                                               double tol,
                                               double orth_tol) {
    if (backend_name == "incremental_eigen" || backend_name == "eigen" || backend_name == "MyEigenISVD") {
        return std::make_unique<MyEigenISVD>(max_rank, eps_svd, tol, orth_tol);
    }
    if (backend_name == "incremental_mkl" || backend_name == "mkl" || backend_name == "lapack" || backend_name == "MyLapackISVD") {
        return std::make_unique<MyLapackISVD>(max_rank, eps_svd, tol, orth_tol);
    }
    throw std::runtime_error("Unknown ISVD backend: " + backend_name);
}

std::unique_ptr<MyISVDBase> create_isvd_solver_from_environment(int max_rank,
                                                                double eps_svd,
                                                                double tol,
                                                                double orth_tol) {
    const char* value = std::getenv("STAGE2_ONLINE_BACKEND");
    return create_isvd_solver(value ? std::string(value) : std::string("incremental_mkl"),
                              max_rank,
                              eps_svd,
                              tol,
                              orth_tol);
}

} // namespace MyAlgebra
