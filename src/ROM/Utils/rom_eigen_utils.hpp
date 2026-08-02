#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>


std::string lower_string(std::string s);

double eigen_relative_residual(const Eigen::SparseMatrix<double>& lhs,
                               const Eigen::VectorXd& rhs,
                               const Eigen::VectorXd& solution);

double eigen_relative_solution_error(const Eigen::VectorXd& fom_solution,
                                     const Eigen::VectorXd& rom_solution);

double eigen_basis_error(const Eigen::MatrixXd& basis,
                         const Eigen::VectorXd& solution);

