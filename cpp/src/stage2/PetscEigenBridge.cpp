#include "stage2/PetscEigenBridge.hpp"

#include <mpi.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace stage2 {

void ensure_eigen_rom_single_rank() {
    PetscMPIInt size = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size != 1) {
        throw std::runtime_error("Eigen ROM backend currently supports single MPI rank only");
    }
}

Eigen::VectorXd petsc_vec_to_eigen(Vec v) {
    ensure_eigen_rom_single_rank();
    PetscInt n = 0;
    VecGetSize(v, &n);
    Eigen::VectorXd out(static_cast<Eigen::Index>(n));
    const PetscScalar* a = nullptr;
    VecGetArrayRead(v, &a);
    for (PetscInt i = 0; i < n; ++i) out[static_cast<Eigen::Index>(i)] = PetscRealPart(a[i]);
    VecRestoreArrayRead(v, &a);
    return out;
}

void eigen_to_petsc_vec(const Eigen::VectorXd& x, Vec v) {
    ensure_eigen_rom_single_rank();
    PetscInt n = 0;
    VecGetSize(v, &n);
    if (x.size() != static_cast<Eigen::Index>(n)) {
        throw std::runtime_error("eigen_to_petsc_vec: vector size mismatch");
    }
    std::vector<PetscInt> ids(static_cast<std::size_t>(n));
    std::vector<PetscScalar> vals(static_cast<std::size_t>(n));
    for (PetscInt i = 0; i < n; ++i) {
        ids[static_cast<std::size_t>(i)] = i;
        vals[static_cast<std::size_t>(i)] = x[static_cast<Eigen::Index>(i)];
    }
    VecSet(v, 0.0);
    VecSetValues(v, n, ids.data(), vals.data(), INSERT_VALUES);
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}

Eigen::SparseMatrix<double> petsc_mat_to_eigen(Mat A) {
    ensure_eigen_rom_single_rank();
    PetscInt m = 0;
    PetscInt n = 0;
    MatGetSize(A, &m, &n);
    std::vector<Eigen::Triplet<double>> triplets;
    for (PetscInt row = 0; row < m; ++row) {
        PetscInt ncols = 0;
        const PetscInt* cols = nullptr;
        const PetscScalar* vals = nullptr;
        MatGetRow(A, row, &ncols, &cols, &vals);
        for (PetscInt k = 0; k < ncols; ++k) {
            const double value = PetscRealPart(vals[k]);
            if (value != 0.0) {
                triplets.emplace_back(static_cast<Eigen::Index>(row),
                                      static_cast<Eigen::Index>(cols[k]),
                                      value);
            }
        }
        MatRestoreRow(A, row, &ncols, &cols, &vals);
    }
    Eigen::SparseMatrix<double> out(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));
    out.setFromTriplets(triplets.begin(), triplets.end());
    out.makeCompressed();
    return out;
}

double eigen_relative_residual(const Eigen::SparseMatrix<double>& A,
                               const Eigen::VectorXd& rhs,
                               const Eigen::VectorXd& solution) {
    const double denom = std::max(rhs.norm(), 1e-30);
    return (A * solution - rhs).norm() / denom;
}

double eigen_relative_solution_error(const Eigen::VectorXd& u_fom,
                                     const Eigen::VectorXd& u_rom) {
    const double denom = std::max(u_fom.norm(), 1e-30);
    return (u_fom - u_rom).norm() / denom;
}

double eigen_basis_error(const Eigen::MatrixXd& V, const Eigen::VectorXd& solution) {
    if (V.cols() == 0) return 1.0;
    const Eigen::VectorXd projection = V * (V.transpose() * solution);
    return eigen_relative_solution_error(solution, projection);
}

} // namespace stage2
