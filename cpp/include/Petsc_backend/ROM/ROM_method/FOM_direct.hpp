#pragma once

#include "Petsc_backend/ROM/ROM_method/ROM_Base.hpp"

namespace petsc_backend {

class FOM_direct final : public ROMBase {
public:
    Mat factor = nullptr;

    FOM_direct();
    ~FOM_direct() override;

    SolveStats main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    SolveStats FOM_main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    KSP get_factorized_solver(Mat lhs) const override;
    void set_factorized_solver(Mat lhs, KSP ksp) override;

private:
    KSP ksp_ = nullptr;
    double last_factorization_time_ = 0.0;
    void set_factorization(Mat factor_in);
};

} // namespace petsc_backend
