#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

void ensure_eigen_rom_single_rank();
Eigen::VectorXd petsc_vec_to_eigen(Vec v);
void eigen_to_petsc_vec(const Eigen::VectorXd& x, Vec v);
Eigen::SparseMatrix<double> petsc_mat_to_eigen(Mat A);

double eigen_relative_residual(const Eigen::SparseMatrix<double>& A,
                               const Eigen::VectorXd& rhs,
                               const Eigen::VectorXd& solution);
double eigen_relative_solution_error(const Eigen::VectorXd& u_fom,
                                     const Eigen::VectorXd& u_rom);
double eigen_basis_error(const Eigen::MatrixXd& V, const Eigen::VectorXd& solution);

