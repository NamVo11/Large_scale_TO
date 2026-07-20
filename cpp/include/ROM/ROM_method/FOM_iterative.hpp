#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

class FOM_iterative final : public ROMBase {
public:
    std::string method = "pcg";
    double pcg_tol = 1e-8;
    int pcg_maxit = 1000;

    explicit FOM_iterative(std::string method_in = "pcg");
    ~FOM_iterative() override;

    SolveStats main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    SolveStats FOM_main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    KSP get_factorized_solver(Mat lhs) const override;
    void set_factorized_solver(Mat lhs, KSP ksp) override;

private:
    KSP ksp_ = nullptr;
    Mat factor = nullptr;
    double last_factorization_time_ = 0.0;
    void set_factorization(Mat factor_in);
};

