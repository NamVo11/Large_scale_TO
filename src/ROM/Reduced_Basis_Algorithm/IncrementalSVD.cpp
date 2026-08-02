#include "ROM/Reduced_Basis_Algorithm/IncrementalSVD.hpp"

#include <stdexcept>

void IncrementalSVD::reset() {
    UpdateBase::reset();
}

int IncrementalSVD::resolveResetRank() const {
    return r_reset > 0 ? r_reset : max_rank;
}

void IncrementalSVD::update(const Eigen::VectorXd& solution, const ROMContext& context) {
    update_flag = false;
    if (store_snapshots) addSnapshot(solution, context);
    if (!ISVD_solver) throw std::runtime_error("IncrementalSVD: ISVD_solver is not initialized");

    const int rank_old = static_cast<int>(V.cols());
    const int r_reset_eff = resolveResetRank();
    const bool reset_flag = V.cols() == 0 || s.size() == 0 || rank_old >= r_reset_eff;

    if (reset_flag) {
        update_flag = ISVD_solver->initialize(V, s, solution);
    } else {
        update_flag = ISVD_solver->update(V, s, solution);
    }

    if (ISVD_solver->reorthogonalizeIfNeeded(V, s)) {
        update_flag = true;
    }
}
