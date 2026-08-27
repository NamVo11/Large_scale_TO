#include "ROM/Reduced_Basis_Algorithm/UpdateBase.hpp"

#include <stdexcept>
#include <utility>

const Eigen::MatrixXd& UpdateBase::getBasis(const ROMContext&) const {
    return V;
}

void UpdateBase::setBasis(const Eigen::MatrixXd& V_in, const Eigen::VectorXd& s_in) {
    V = V_in;
    s = s_in;
    update_flag = true;
}

void UpdateBase::setBasis(Eigen::MatrixXd&& V_in, Eigen::VectorXd&& s_in) {
    V = std::move(V_in);
    s = std::move(s_in);
    update_flag = true;
}

void UpdateBase::addSnapshot(const Eigen::VectorXd& solution, const ROMContext&) {
    if (solution.size() == 0) return;
    if (snapshots.size() == 0) {
        snapshots.resize(solution.size(), 1);
        snapshots.col(0) = solution;
        return;
    }
    if (snapshots.rows() != solution.size()) {
        throw std::runtime_error("UpdateBase::addSnapshot: snapshot size mismatch");
    }
    const Eigen::Index old_cols = snapshots.cols();
    snapshots.conservativeResize(Eigen::NoChange, old_cols + 1);
    snapshots.col(old_cols) = solution;
}

void UpdateBase::clearSnapshots() {
    snapshots.resize(0, 0);
}

bool UpdateBase::update_reduced_cache_incremental(const Eigen::SparseMatrix<double>&,
                                                  const Eigen::MatrixXd&,
                                                  Eigen::MatrixXd&,
                                                  Eigen::LLT<Eigen::MatrixXd>&,
                                                  Eigen::MatrixXd&,
                                                  int&,
                                                  ROMInfo&) const {
    return false;
}

void UpdateBase::reset() {
    V.resize(0, 0);
    snapshots.resize(0, 0);
    s.resize(0);
    W.resize(0, 0);
    update_flag = true;
}
