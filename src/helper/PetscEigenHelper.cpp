#include "helper/PetscEigenHelper.hpp"

#include <mpi.h>

#include <numeric>
#include <stdexcept>

Eigen::VectorXd gather_vec_to_all_local(Vec v) {
    PetscMPIInt size = 1;
    MPI_Comm_size(PetscObjectComm((PetscObject)v), &size);
    if (size == 1) {
        PetscInt n = 0;
        VecGetLocalSize(v, &n);
        Eigen::VectorXd out(static_cast<Eigen::Index>(n));
        const PetscScalar* a = nullptr;
        VecGetArrayRead(v, &a);
        for (PetscInt i = 0; i < n; ++i) out(static_cast<Eigen::Index>(i)) = PetscRealPart(a[i]);
        VecRestoreArrayRead(v, &a);
        return out;
    }

    Vec vall = nullptr;
    VecScatter scatter = nullptr;
    VecScatterCreateToAll(v, &scatter, &vall);
    VecScatterBegin(scatter, v, vall, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatter, v, vall, INSERT_VALUES, SCATTER_FORWARD);

    PetscInt n = 0;
    VecGetSize(vall, &n);
    Eigen::VectorXd out(static_cast<Eigen::Index>(n));
    const PetscScalar* a = nullptr;
    VecGetArrayRead(vall, &a);
    for (PetscInt i = 0; i < n; ++i) out(static_cast<Eigen::Index>(i)) = PetscRealPart(a[i]);
    VecRestoreArrayRead(vall, &a);

    VecScatterDestroy(&scatter);
    VecDestroy(&vall);
    return out;
}

void gather_vec_to_all_into_column(Vec v, Eigen::MatrixXd& history, int col) {
    const Eigen::VectorXd values = gather_vec_to_all_local(v);
    if (col < 0 || col >= history.cols() || history.rows() != values.size()) {
        throw std::runtime_error("gather_vec_to_all_into_column: history size mismatch");
    }
    history.col(col) = values;
}

void set_vec_from_eigen_column(Vec v, const Eigen::MatrixXd& values, int col, double scale) {
    if (values.size() == 0) {
        VecSet(v, 0.0);
        return;
    }
    if (col < 0 || col >= values.cols()) {
        throw std::runtime_error("set_vec_from_eigen_column: column out of range");
    }

    PetscMPIInt size = 1;
    MPI_Comm_size(PetscObjectComm((PetscObject)v), &size);
    if (size == 1) {
        PetscInt n = 0;
        VecGetLocalSize(v, &n);
        if (values.rows() != static_cast<Eigen::Index>(n)) {
            throw std::runtime_error("set_vec_from_eigen_column: vector size mismatch");
        }
        PetscScalar* a = nullptr;
        VecGetArray(v, &a);
        for (PetscInt i = 0; i < n; ++i) a[i] = scale * values(static_cast<Eigen::Index>(i), col);
        VecRestoreArray(v, &a);
        return;
    }

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PetscObjectComm((PetscObject)v), &rank);
    VecSet(v, 0.0);
    if (rank == 0) {
        std::vector<PetscInt> ids(static_cast<std::size_t>(values.rows()));
        std::vector<PetscScalar> vals(static_cast<std::size_t>(values.rows()));
        std::iota(ids.begin(), ids.end(), 0);
        for (Eigen::Index i = 0; i < values.rows(); ++i) {
            vals[static_cast<std::size_t>(i)] = scale * values(i, col);
        }
        VecSetValues(v, static_cast<PetscInt>(ids.size()), ids.data(), vals.data(), INSERT_VALUES);
    }
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}

void set_vec_fixed_from_eigen_column(Vec v,
                                     const std::vector<int>& fixed,
                                     const Eigen::MatrixXd& values,
                                     int col,
                                     double scale) {
    if (fixed.empty() || values.size() == 0) return;
    if (col < 0 || col >= values.cols()) {
        throw std::runtime_error("set_vec_fixed_from_eigen_column: column out of range");
    }

    PetscMPIInt size = 1;
    MPI_Comm_size(PetscObjectComm((PetscObject)v), &size);
    if (size == 1) {
        PetscInt n = 0;
        VecGetLocalSize(v, &n);
        PetscScalar* a = nullptr;
        VecGetArray(v, &a);
        for (int id : fixed) {
            if (id < 0 || id >= n) {
                VecRestoreArray(v, &a);
                throw std::runtime_error("set_vec_fixed_from_eigen_column: fixed dof out of range");
            }
            a[id] = scale * values(static_cast<Eigen::Index>(id), col);
        }
        VecRestoreArray(v, &a);
        return;
    }

    PetscMPIInt rank = 0;
    MPI_Comm_rank(PetscObjectComm((PetscObject)v), &rank);
    if (rank == 0) {
        std::vector<PetscInt> ids(fixed.begin(), fixed.end());
        std::vector<PetscScalar> vals(fixed.size());
        for (std::size_t i = 0; i < fixed.size(); ++i) {
            vals[i] = scale * values(static_cast<Eigen::Index>(fixed[i]), col);
        }
        VecSetValues(v, static_cast<PetscInt>(ids.size()), ids.data(), vals.data(), INSERT_VALUES);
    }
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}

