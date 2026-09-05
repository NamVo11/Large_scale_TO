#pragma once

#include "ROM/Reduced_Basis_Algorithm/UpdateBase.hpp"

#ifdef EIGEN_USE_LAPACKE
#include <mkl_spblas.h>

#include <vector>
#endif

class MyIncrementalQR1 final : public UpdateBase {
public:
    double e_rb = 1e-20;
    int base_rank = 0;
    int last_iter = 0;
    bool has_last_iter = false;

    MyIncrementalQR1(double e_rb_in = 1e-20, int max_rank_in = 50);
    ~MyIncrementalQR1() override;

    void reset() override;
    void update(const Eigen::VectorXd& solution, const ROMBase::ROMContext& context) override;
    bool update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                          const Eigen::MatrixXd& V,
                                          Eigen::MatrixXd& A_r_cache,
                                          Eigen::LLT<Eigen::MatrixXd>& factorized_A_r,
                                          Eigen::MatrixXd& AV_cache,
                                          int& A_r_rank_cache,
                                          ROMInfo& info) const override;

private:
    bool eigen_update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                                const Eigen::MatrixXd& V,
                                                Eigen::MatrixXd& A_r_cache,
                                                Eigen::LLT<Eigen::MatrixXd>& factorized_A_r,
                                                Eigen::MatrixXd& AV_cache,
                                                int& A_r_rank_cache) const;
    bool mkl_update_reduced_cache_incremental(const Eigen::SparseMatrix<double>& A,
                                              const Eigen::MatrixXd& V,
                                              Eigen::MatrixXd& A_r_cache,
                                              Eigen::LLT<Eigen::MatrixXd>& factorized_A_r,
                                              Eigen::MatrixXd& AV_cache,
                                              int& A_r_rank_cache) const;

#ifdef EIGEN_USE_LAPACKE
    mutable sparse_matrix_t mkl_lhs_cache_ = nullptr;
    mutable const Eigen::SparseMatrix<double>* mkl_lhs_cache_key_ = nullptr;
    mutable const void* mkl_lhs_value_key_ = nullptr;
    mutable Eigen::Index mkl_lhs_rows_key_ = 0;
    mutable Eigen::Index mkl_lhs_cols_key_ = 0;
    mutable Eigen::Index mkl_lhs_nnz_key_ = 0;
    mutable std::vector<MKL_INT> mkl_lhs_outer_;
    mutable std::vector<MKL_INT> mkl_lhs_inner_;

    void clear_mkl_lhs_cache() const;
    void ensure_mkl_lhs_cache(const Eigen::SparseMatrix<double>& A) const;
#endif
};
