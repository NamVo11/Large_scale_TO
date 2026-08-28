#include "ROM/Reduced_Basis_Algorithm/IncrementalSVD.hpp"
#include "ROM/Reduced_Basis_Algorithm/MyPOD_SVD1.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int arg_int(int argc, char** argv, const std::string& key, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) return std::atoi(argv[i + 1]);
    }
    return fallback;
}

double subspace_error(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    if (A.cols() == 0 && B.cols() == 0) return 0.0;
    if (A.rows() != B.rows()) return std::numeric_limits<double>::infinity();
    const int r = std::min(static_cast<int>(A.cols()), static_cast<int>(B.cols()));
    if (r == 0) return 1.0;
    Eigen::MatrixXd C = A.leftCols(r).transpose() * B.leftCols(r);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C);
    const Eigen::VectorXd sigma = svd.singularValues().array().min(1.0).max(0.0);
    return std::sqrt(std::max(0.0, static_cast<double>(r) - sigma.array().square().sum()));
}

Eigen::VectorXd make_snapshot(int ndof, int step, int rank_seed) {
    Eigen::VectorXd x(ndof);
    const double a = 0.017 * static_cast<double>(step + 1);
    const double b = 0.031 * static_cast<double>(rank_seed + 1);
    for (int i = 0; i < ndof; ++i) {
        const double t = static_cast<double>((i % 1009) + 1);
        x[i] = std::sin(a * t) + 0.5 * std::cos(b * t) + 0.01 * std::sin(0.0001 * static_cast<double>(i * (step + 3)));
    }
    return x;
}

} // namespace

int main(int argc, char** argv) {
    const int ndof = arg_int(argc, argv, "--ndof", 200000);
    const int nsnapshot = arg_int(argc, argv, "--nsnapshot", 50);
    const int max_rank = arg_int(argc, argv, "--max_rank", 15);
    const int online_updates = arg_int(argc, argv, "--online_updates", 50);

    std::cout << "basis_update_benchmark:"
              << " ndof=" << ndof
              << " nsnapshot=" << nsnapshot
              << " max_rank=" << max_rank
              << " online_updates=" << online_updates << "\n";

    ROMBase::ROMContext context;
    context.opt_iter = 0;

    MyPOD_SVD1 pod_snapshot(0.0, 1e-10, max_rank);
    pod_snapshot.pod_solver = MyAlgebra::create_pod_solver("snapshot_eigen");
    for (int j = 0; j < nsnapshot; ++j) pod_snapshot.addSnapshot(make_snapshot(ndof, j, max_rank), context);
    auto t0 = Clock::now();
    pod_snapshot.update(Eigen::VectorXd(), context);
    const double snapshot_time = elapsed(t0);

    MyPOD_SVD1 pod_mkl(0.0, 1e-10, max_rank);
    pod_mkl.pod_solver = MyAlgebra::create_pod_solver("snapshot_mkl");
    pod_mkl.snapshots = pod_snapshot.snapshots;
    t0 = Clock::now();
    pod_mkl.update(Eigen::VectorXd(), context);
    const double snapshot_mkl_time = elapsed(t0);

    MyPOD_SVD1 pod_bdc(0.0, 1e-10, max_rank);
    pod_bdc.pod_solver = MyAlgebra::create_pod_solver("bdcsvd");
    pod_bdc.snapshots = pod_snapshot.snapshots;
    t0 = Clock::now();
    pod_bdc.update(Eigen::VectorXd(), context);
    const double bdc_time = elapsed(t0);

    std::cout << "offline_snapshot_eigen[s]=" << snapshot_time
              << " rank=" << pod_snapshot.V.cols() << "\n";
    std::cout << "offline_snapshot_mkl[s]=" << snapshot_mkl_time
              << " rank=" << pod_mkl.V.cols() << "\n";
    std::cout << "offline_bdcsvd[s]=" << bdc_time
              << " rank=" << pod_bdc.V.cols() << "\n";
    std::cout << "offline_subspace_error=" << subspace_error(pod_bdc.V, pod_snapshot.V) << "\n";
    std::cout << "offline_mkl_subspace_error=" << subspace_error(pod_snapshot.V, pod_mkl.V) << "\n";

    std::vector<Eigen::VectorXd> online_snapshots;
    online_snapshots.reserve(static_cast<std::size_t>(online_updates));
    for (int j = 0; j < online_updates; ++j) {
        online_snapshots.push_back(make_snapshot(ndof, j, max_rank));
    }

    IncrementalSVD online_fast(10000, 1e-10, 1e-12, max_rank);
    online_fast.ISVD_solver = MyAlgebra::create_isvd_solver("incremental_eigen", max_rank, 1e-10, 1e-12, online_fast.orth_tol);
    online_fast.store_snapshots = false;
    t0 = Clock::now();
    for (int j = 0; j < online_updates; ++j) online_fast.update(online_snapshots[static_cast<std::size_t>(j)], context);
    const double online_fast_time = elapsed(t0);

    IncrementalSVD online_mkl(10000, 1e-10, 1e-12, max_rank);
    online_mkl.ISVD_solver = MyAlgebra::create_isvd_solver("incremental_mkl", max_rank, 1e-10, 1e-12, online_mkl.orth_tol);
    online_mkl.store_snapshots = false;
    t0 = Clock::now();
    for (int j = 0; j < online_updates; ++j) online_mkl.update(online_snapshots[static_cast<std::size_t>(j)], context);
    const double online_mkl_time = elapsed(t0);

    IncrementalSVD online_store(10000, 1e-10, 1e-12, max_rank);
    online_store.store_snapshots = true;
    t0 = Clock::now();
    for (int j = 0; j < online_updates; ++j) online_store.update(online_snapshots[static_cast<std::size_t>(j)], context);
    const double online_store_time = elapsed(t0);

    std::cout << "online_eigen_no_snapshot_store[s]=" << online_fast_time
              << " rank=" << online_fast.V.cols() << "\n";
    std::cout << "online_mkl_no_snapshot_store[s]=" << online_mkl_time
              << " rank=" << online_mkl.V.cols() << "\n";
    std::cout << "online_with_snapshot_store[s]=" << online_store_time
              << " rank=" << online_store.V.cols() << "\n";
    std::cout << "online_subspace_error=" << subspace_error(online_store.V, online_fast.V) << "\n";
    std::cout << "online_mkl_subspace_error=" << subspace_error(online_fast.V, online_mkl.V) << "\n";
    return 0;
}
