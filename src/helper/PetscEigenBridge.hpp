#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

void ensure_eigen_rom_single_rank();
Eigen::VectorXd petsc_vec_to_eigen(Vec v);
void eigen_to_petsc_vec(const Eigen::VectorXd& x, Vec v);
Eigen::SparseMatrix<double> petsc_mat_to_eigen(Mat A);
