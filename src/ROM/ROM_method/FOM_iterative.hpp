#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"


class FOM_iterative final : public ROMBase {
public:
    std::string method = "pcg";
    double pcg_tol = 1e-8;
    int pcg_maxit = 1000;

    explicit FOM_iterative(std::string method_in = "pcg");

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u,
                    const ROMContext& context) override;
    SolveStats FOM_main(const Eigen::SparseMatrix<double>& lhs,
                        const Eigen::VectorXd& rhs,
                        Eigen::VectorXd& u,
                        const ROMContext& context) override;
};

