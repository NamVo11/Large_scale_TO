#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

namespace stage2 {

class FOM_direct final : public ROMBase {
public:
    Mat factor = nullptr;

    FOM_direct();
    ~FOM_direct() override;

    SolveStats main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    SolveStats FOM_main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    void set_factorization(Mat factor_in) override;

private:
    KSP ksp_ = nullptr;
    double last_factorization_time_ = 0.0;
};

} // namespace stage2
