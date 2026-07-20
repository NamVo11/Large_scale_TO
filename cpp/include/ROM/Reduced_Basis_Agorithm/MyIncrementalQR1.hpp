#pragma once

#include "ROM/Reduced_Basis_Agorithm/UpdateBase.hpp"

class MyIncrementalQR1 final : public UpdateBase {
public:
    double e_rb = 1e-20;
    int base_rank = 0;
    int last_iter = 0;
    bool has_last_iter = false;

    MyIncrementalQR1(double e_rb_in = 1e-20, int max_rank_in = 50);

    void reset() override;
    void update(const Eigen::VectorXd& solution, const ROMContext& context) override;
};

