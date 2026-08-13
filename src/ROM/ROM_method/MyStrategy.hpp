#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>
#include <utility>


class MyStrategy : public ROMBase {
public:
    double aaa = 0.0;
    double bbbb = 0.0;

    MyStrategy(std::vector<std::shared_ptr<UpdateBase>> basis_list_in,
               double residual_tol_in,
               int n_p_in,
               std::shared_ptr<ROMBase> fom_solver_in,
               std::shared_ptr<ErrorEstimatorBase> error_estimator_in,
               bool online_in)
        : ROMBase(std::move(basis_list_in)) {
        residual_tol = residual_tol_in;
        n_p = n_p_in;
        fom_solver = std::move(fom_solver_in);
        error_estimator = std::move(error_estimator_in);
        online = online_in;
        name = "MyStrategy";
    }

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u,
                    const ROMContext& context) override;

private:
    const Eigen::SparseMatrix<double>* A_r_lhs_cache_ = nullptr;
    int A_r_rank_cache_ = 0;
    Eigen::MatrixXd AV_cache_;
};
