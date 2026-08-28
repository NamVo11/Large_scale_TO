#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <petscksp.h>

#include <vector>

// Return the full PETSc vector on every rank. Use this only where the current
// algorithm genuinely needs full local history on all ranks.
Eigen::VectorXd gather_vec_to_all_local(Vec v);

// Store a PETSc vector as one column of a dense Eigen history matrix.
// For one rank this is a direct local copy; for MPI this gathers the full
// vector to all ranks to match the current stage flow.
void gather_vec_to_all_into_column(Vec v, Eigen::MatrixXd& history, int col);

// Set a PETSc vector from one dense Eigen matrix column.
void set_vec_from_eigen_column(Vec v, const Eigen::MatrixXd& values, int col, double scale = 1.0);

// Set only selected dofs from one dense Eigen matrix column.
void set_vec_fixed_from_eigen_column(Vec v,
                                     const std::vector<int>& fixed,
                                     const Eigen::MatrixXd& values,
                                     int col,
                                     double scale = 1.0);

SolveStats solve_petsc_with_rom(ROMBase& rom,
                                Mat lhs,
                                Vec rhs,
                                Vec solution,
                                Eigen::SparseMatrix<double>* lhs_e_cache = nullptr);
