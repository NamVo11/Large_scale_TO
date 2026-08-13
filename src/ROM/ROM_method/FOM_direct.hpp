#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

#ifdef EIGEN_USE_LAPACKE
#include <Eigen/PardisoSupport>
#else
#include <Eigen/SparseCholesky>
#endif

class FactorizedSolver {
public:
#ifdef EIGEN_USE_LAPACKE
    Eigen::PardisoLLT<Eigen::SparseMatrix<double>> solver;
#else
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
#endif
};

class FOM_direct final : public ROMBase {
public:
    FOM_direct();

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u,
                    const ROMContext& context) override;
    SolveStats FOM_main(const Eigen::SparseMatrix<double>& lhs,
                        const Eigen::VectorXd& rhs,
                        Eigen::VectorXd& u,
                        const ROMContext& context) override;
    std::shared_ptr<FactorizedSolver> get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const override;
    void set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                               std::shared_ptr<FactorizedSolver> solver) override;

private:
    const Eigen::SparseMatrix<double>* factor_ = nullptr;
    std::shared_ptr<FactorizedSolver> solver_;
    double last_factorization_time_ = 0.0;
    void set_factorization(const Eigen::SparseMatrix<double>& lhs);
};
