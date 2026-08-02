#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

class PetscFOM_iterative final : public ROMBase {
public:
    std::string method = "pcg";
    double pcg_tol = 1e-8;
    int pcg_maxit = 1000;

    explicit PetscFOM_iterative(std::string method_in = "pcg");
    ~PetscFOM_iterative() override;

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u,
                    const ROMContext& context) override;
    SolveStats FOM_main(const Eigen::SparseMatrix<double>& lhs,
                        const Eigen::VectorXd& rhs,
                        Eigen::VectorXd& u,
                        const ROMContext& context) override;
    KSP get_petsc_factorized_solver(Mat lhs) const override;
    void set_petsc_factorized_solver(Mat lhs, KSP ksp) override;
    void set_petsc_system(Mat lhs, Vec rhs, Vec u) override;

private:
    KSP ksp_ = nullptr;
    Mat factor = nullptr;
    Mat current_lhs_ = nullptr;
    Vec current_rhs_ = nullptr;
    Vec current_u_ = nullptr;
    double last_factorization_time_ = 0.0;

    void set_factorization(Mat factor_in);
};
