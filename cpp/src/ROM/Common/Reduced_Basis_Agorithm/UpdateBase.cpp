#include "ROM/Common/Reduced_Basis_Agorithm/UpdateBase.hpp"

#include <stdexcept>
#include <utility>

UpdateBase::UpdateBase(int max_rank_in) : max_rank(max_rank_in) {}

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

void UpdateBase::reset() {
    V.resize(0, 0);
    snapshots.resize(0, 0);
    s.resize(0);
    W.resize(0, 0);
    update_flag = true;
}
