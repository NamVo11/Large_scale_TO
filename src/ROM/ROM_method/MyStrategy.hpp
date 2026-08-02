#pragma once

#include "ROM/Error_Estimator/ErrorEstimatorBase.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>


class MyStrategy : public ROMBase {
public:
    double residual_tol = 1e-3;
    int n_p = 0;
    std::shared_ptr<ROMBase> fom_solver;
    std::shared_ptr<ErrorEstimatorBase> error_estimator;
    Eigen::MatrixXd A_r_cache;
    Eigen::LLT<Eigen::MatrixXd> factorized_A_r;
    int fom_flag = 1;
    double aaa = 0.0;
    double bbbb = 0.0;
    bool online = true;

    MyStrategy(std::vector<std::shared_ptr<UpdateBase>> basis_list_in,
               double residual_tol_in,
               int n_p_in,
               std::shared_ptr<ROMBase> fom_solver_in,
               std::shared_ptr<ErrorEstimatorBase> error_estimator_in,
               bool online_in);

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u,
                    const ROMContext& context) override;
    std::shared_ptr<FactorizedSolver> get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const override;
    void set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                               std::shared_ptr<FactorizedSolver> solver) override;
    KSP get_petsc_factorized_solver(Mat lhs) const override;
    void set_petsc_factorized_solver(Mat lhs, KSP ksp) override;
    void set_petsc_system(Mat lhs, Vec rhs, Vec u) override;

private:
    const Eigen::SparseMatrix<double>* A_r_lhs_cache_ = nullptr;
    int A_r_rank_cache_ = 0;
    Eigen::MatrixXd AV_cache_;

    void clear_reduced_cache();
    void rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                               const Eigen::MatrixXd& V,
                               ROMInfo& info);
};

class MyStrategy2 final : public MyStrategy {
public:
    MyStrategy2(std::vector<std::shared_ptr<UpdateBase>> basis_list_in,
                double residual_tol_in,
                int n_p_in,
                std::shared_ptr<ROMBase> fom_solver_in,
                std::shared_ptr<ErrorEstimatorBase> error_estimator_in,
                bool online_in);

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u,
                    const ROMContext& context) override;
    std::shared_ptr<FactorizedSolver> get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const override;
    void set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                               std::shared_ptr<FactorizedSolver> solver) override;
    KSP get_petsc_factorized_solver(Mat lhs) const override;
    void set_petsc_factorized_solver(Mat lhs, KSP ksp) override;
    void set_petsc_system(Mat lhs, Vec rhs, Vec u) override;

private:
    const Eigen::SparseMatrix<double>* A_r_lhs_cache2_ = nullptr;
    int A_r_rank_cache2_ = 0;
    Eigen::MatrixXd AV_cache2_;

    void clear_reduced_cache2();
    bool update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                          const Eigen::MatrixXd& V,
                                          const UpdateBase& basis_runtime,
                                          ROMInfo& info);
    void rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                               const Eigen::MatrixXd& V,
                               ROMInfo& info);
};
