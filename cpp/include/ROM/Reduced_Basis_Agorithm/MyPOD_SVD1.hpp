#pragma once

#include "ROM/Reduced_Basis_Agorithm/UpdateBase.hpp"

namespace stage2 {

class MyPOD_SVD1 final : public UpdateBase {
public:
    double tol = 1e-12;
    double energy_tol = 0.0;
    double snapshot_window = 0.0; // 0 means Inf, matching the current MATLAB usage.

    MyPOD_SVD1(double snapshot_window_in, double tol_in, int max_rank_in, double energy_tol_in = 0.0);
    void update(Vec solution, const ROMContext& context) override;
};

} // namespace stage2

