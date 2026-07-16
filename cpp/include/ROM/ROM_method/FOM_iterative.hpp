#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

namespace stage2 {

class FOM_iterative final : public ROMBase {
public:
    std::string method = "pcg";
    double pcg_tol = 1e-8;
    int pcg_maxit = 1000;

    explicit FOM_iterative(std::string method_in = "pcg");
    ~FOM_iterative() override;

    SolveStats main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    SolveStats FOM_main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    void set_factorization(Mat factor_in) override;

private:
    KSP ksp_ = nullptr;
    Mat factor = nullptr;
    double last_factorization_time_ = 0.0;
};

} // namespace stage2
