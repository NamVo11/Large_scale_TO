#pragma once

#include "ROM/Error_Estimator/ErrorEstimatorBase.hpp"
#include "ROM/ROMTypes.hpp"
#include "myjson.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <petscksp.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef EIGEN_USE_LAPACKE
#include <mkl_spblas.h>
#endif

class UpdateBase;
class FactorizedSolver;

class ROMBase {
public:
    bool use_mor = true;
    std::vector<std::shared_ptr<UpdateBase>> basis_list;
    std::string name = "ROM_Base";
    std::vector<std::vector<ROMInfo>> ROM_info;
    double residual_tol = 1e-3;
    int n_p = 0;
    std::shared_ptr<ROMBase> fom_solver;
    std::shared_ptr<ErrorEstimatorBase> error_estimator;
    Eigen::MatrixXd A_r_cache;
    Eigen::LLT<Eigen::MatrixXd> factorized_A_r;
    int fom_flag = 1;
    bool online = true;

    explicit ROMBase(std::vector<std::shared_ptr<UpdateBase>> basis_list_in = {});
    virtual ~ROMBase();

    virtual SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                            const Eigen::VectorXd& rhs,
                            Eigen::VectorXd& u,
                            const ROMContext& context) = 0;
    virtual SolveStats FOM_main(const Eigen::SparseMatrix<double>& lhs,
                                const Eigen::VectorXd& rhs,
                                Eigen::VectorXd& u,
                                const ROMContext& context);
    virtual std::shared_ptr<FactorizedSolver> get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const;
    virtual void set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                                       std::shared_ptr<FactorizedSolver> solver);
    virtual KSP get_petsc_factorized_solver(Mat lhs) const;
    virtual void set_petsc_factorized_solver(Mat lhs, KSP ksp);
    virtual void set_petsc_system(Mat lhs, Vec rhs, Vec u);

    static double now_time();
    void store_info(const ROMInfo& info, const ROMContext& context);

protected:
    void check_strategy_ready(const char* strategy_name) const;
    void clear_reduced_cache(Eigen::MatrixXd& AV_cache,
                             const Eigen::SparseMatrix<double>*& lhs_cache,
                             int& rank_cache);
    void rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                               const Eigen::MatrixXd& V,
                               Eigen::MatrixXd& AV_cache,
                               const Eigen::SparseMatrix<double>*& lhs_cache,
                               int& rank_cache,
                               ROMInfo& info);
    void eigen_rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                                     const Eigen::MatrixXd& V,
                                     Eigen::MatrixXd& AV_cache);
    void mkl_rebuild_reduced_cache(const Eigen::SparseMatrix<double>& A,
                                   const Eigen::MatrixXd& V,
                                   Eigen::MatrixXd& AV_cache);

    void append_basis_debug(const char* phase,
                            const ROMContext& context,
                            int basis_id,
                            const UpdateBase& basis) const;
    void append_decision_debug(const ROMContext& context,
                               int rank,
                               double residual,
                               double threshold,
                               bool accepted) const;

    mutable MklSparseMatrixHandle mkl_lhs_cache_ = nullptr;

#ifdef EIGEN_USE_LAPACKE
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

std::shared_ptr<ROMBase> InitializeFOMSolver(const std::string& fom_solver_type,
                                             const std::string& linear_backend = "eigen");

std::shared_ptr<ROMBase> InitializeDynamicROM(const std::string& method_name,
                                              const std::string& fom_solver_type,
                                              double residual_tol,
                                              int rank1,
                                              int max_rank,
                                              const std::string& offline_update,
                                              const std::string& online_update,
                                              const std::string& linear_backend = "eigen");

std::shared_ptr<ROMBase> InitializeROM(const myjson::json& rom_json,
                                       const myjson::json& run_json);

inline ROMBase::ROMBase(std::vector<std::shared_ptr<UpdateBase>> basis_list_in)
    : basis_list(std::move(basis_list_in)) {}

inline ROMBase::~ROMBase() {
#ifdef EIGEN_USE_LAPACKE
    clear_mkl_lhs_cache();
#endif
}

inline SolveStats ROMBase::FOM_main(const Eigen::SparseMatrix<double>&,
                                    const Eigen::VectorXd&,
                                    Eigen::VectorXd&,
                                    const ROMContext&) {
    throw std::runtime_error("FOM_main is not implemented for " + name);
}
