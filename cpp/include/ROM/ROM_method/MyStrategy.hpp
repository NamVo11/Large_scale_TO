#pragma once

#include "ROM/Error_Estimator/ErrorEstimatorBase.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>

namespace stage2 {

class MyStrategy : public ROMBase {
public:
    double residual_tol = 1e-3;
    int n_p = 0;
    std::shared_ptr<ROMBase> fom_solver;
    std::shared_ptr<ErrorEstimatorBase> error_estimator;
    Eigen::MatrixXd A_r_cache;
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

    SolveStats main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    void set_factorization(Mat factor) override;

private:
    Mat A_r_lhs_cache_ = nullptr;
    int A_r_rank_cache_ = 0;
    int A_r_basis_version_cache_ = -1;
    Mat eigen_lhs_cache_ = nullptr;
    Eigen::SparseMatrix<double> eigen_lhs_cache_matrix_;

    void clear_reduced_cache();
    const Eigen::SparseMatrix<double>& eigen_lhs(Mat lhs, ROMInfo& info);
};

class MyStrategy2 final : public MyStrategy {
public:
    MyStrategy2(std::vector<std::shared_ptr<UpdateBase>> basis_list_in,
                double residual_tol_in,
                int n_p_in,
                std::shared_ptr<ROMBase> fom_solver_in,
                std::shared_ptr<ErrorEstimatorBase> error_estimator_in,
                bool online_in);

    SolveStats main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    void set_factorization(Mat factor) override;

private:
    Mat A_r_lhs_cache2_ = nullptr;
    int A_r_rank_cache2_ = 0;
    int A_r_basis_version_cache2_ = -1;
    Mat eigen_lhs_cache2_ = nullptr;
    Eigen::SparseMatrix<double> eigen_lhs_cache_matrix2_;
    Eigen::MatrixXd AV_cache2_;

    void clear_reduced_cache2();
    const Eigen::SparseMatrix<double>& eigen_lhs2(Mat lhs, ROMInfo& info);
    bool update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A_e,
                                          const Eigen::MatrixXd& V,
                                          const UpdateBase& basis_runtime,
                                          ROMInfo& info);
    void rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A_e,
                               const Eigen::MatrixXd& V,
                               Mat lhs,
                               const UpdateBase& basis_runtime,
                               ROMInfo& info);
};

} // namespace stage2

