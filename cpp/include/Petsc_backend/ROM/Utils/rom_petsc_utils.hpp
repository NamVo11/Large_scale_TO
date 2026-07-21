#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <string>
#include <vector>

namespace petsc_backend {

std::string lower_string(std::string s);
double compute_relative_residual(Mat lhs, Vec rhs, Vec solution);
double compute_relative_solution_error(Vec u_fom, Vec u_rom);
double compute_basis_error(const std::vector<Vec>& V, Vec solution);
bool solve_dense_system(std::vector<double> A, std::vector<double> b, std::vector<double>& x);
bool symmetric_eigen_jacobi(std::vector<double> A,
                            int n,
                            std::vector<double>& eigenvalues,
                            std::vector<double>& eigenvectors);

} // namespace petsc_backend
