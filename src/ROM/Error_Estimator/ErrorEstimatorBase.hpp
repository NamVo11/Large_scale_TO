#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>

struct ErrorAssessment {
    std::string estimator_name;
    double error_indicator = 0.0;
    double residual = 0.0;
    double threshold = 0.0;
    double residual_tol_eff = 0.0;
    bool accepted = false;
};

class ErrorEstimatorBase {
public:
    std::string name = "ErrorEstimatorBase";
    virtual ~ErrorEstimatorBase() = default;

    virtual ErrorAssessment assess(const Eigen::SparseMatrix<double>& lhs,
                                   const Eigen::VectorXd& rhs,
                                   const Eigen::VectorXd& solution,
                                   const ROMBase::ROMContext& context,
                                   MklSparseMatrixHandle lhs_mkl = nullptr) = 0;

    virtual double relative_residual(const Eigen::SparseMatrix<double>& lhs,
                                     const Eigen::VectorXd& rhs,
                                     const Eigen::VectorXd& solution,
                                     MklSparseMatrixHandle lhs_mkl = nullptr) const = 0;
    virtual double eigen_relative_residual(const Eigen::SparseMatrix<double>& lhs,
                                           const Eigen::VectorXd& rhs,
                                           const Eigen::VectorXd& solution) const = 0;
    virtual double mkl_relative_residual(const Eigen::SparseMatrix<double>& lhs,
                                         const Eigen::VectorXd& rhs,
                                         const Eigen::VectorXd& solution,
                                         MklSparseMatrixHandle lhs_mkl) const = 0;
    virtual double relative_solution_error(const Eigen::VectorXd& fom_solution,
                                           const Eigen::VectorXd& rom_solution) const = 0;
    virtual double basis_error(const Eigen::MatrixXd& basis,
                               const Eigen::VectorXd& solution) const = 0;
};
