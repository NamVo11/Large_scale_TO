#pragma once

#include "ROM/Common/Reduced_Basis_Agorithm/UpdateBase.hpp"

class MyPOD_SVD1 final : public UpdateBase {
public:
    enum class Backend {
        SnapshotEigen,
        SnapshotMKL,
        EigenBDCSVD
    };

    double tol = 1e-12;
    double energy_tol = 0.0;
    double snapshot_window = 0.0; // 0 means Inf, matching current MATLAB usage.
    Backend backend = Backend::SnapshotMKL;

    MyPOD_SVD1(double snapshot_window_in, double tol_in, int max_rank_in, double energy_tol_in = 0.0);
    void update(const Eigen::VectorXd& solution, const ROMContext& context) override;

private:
    static Backend backendFromEnvironment();
};

