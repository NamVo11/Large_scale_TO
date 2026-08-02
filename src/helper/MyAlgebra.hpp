#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <petscmat.h>

#include <memory>
#include <string>
#include <vector>

namespace MyAlgebra {

std::vector<int> to_ints(const std::vector<PetscInt>& in);

namespace Petsc {
void apply_dirichlet_to_matrix(Mat A, const std::vector<int>& fixed, double diag = 1.0);
}

namespace EigenOps {
void apply_dirichlet_to_matrix(Eigen::SparseMatrix<double>& A,
                               const std::vector<int>& fixed,
                               double diag = 1.0);
}

struct PODResult {
    Eigen::MatrixXd V;
    Eigen::VectorXd s;
    Eigen::MatrixXd W;
    Eigen::VectorXd s_all;
};

class MyPODBase {
public:
    virtual ~MyPODBase() = default;
    virtual PODResult compute(const Eigen::MatrixXd& X, int max_rank) const = 0;
};

class MyEigenBDCSVD final : public MyPODBase {
public:
    PODResult compute(const Eigen::MatrixXd& X, int max_rank) const override;
};

class MyEigenSnapshot final : public MyPODBase {
public:
    PODResult compute(const Eigen::MatrixXd& X, int max_rank) const override;
};

class MyEigenJacobiSVD final : public MyPODBase {
public:
    PODResult compute(const Eigen::MatrixXd& X, int max_rank) const override;
};

class MyLapackSnapshot final : public MyPODBase {
public:
    PODResult compute(const Eigen::MatrixXd& X, int max_rank) const override;
};

class MyLapackDGESDD final : public MyPODBase {
public:
    PODResult compute(const Eigen::MatrixXd& X, int max_rank) const override;
};

std::unique_ptr<MyPODBase> create_pod_solver(const std::string& backend_name);
std::unique_ptr<MyPODBase> create_pod_solver_from_environment();

class MyISVDBase {
public:
    MyISVDBase(int max_rank_in, double eps_svd_in, double tol_in, double orth_tol_in)
        : max_rank(max_rank_in), eps_svd(eps_svd_in), tol(tol_in), orth_tol(orth_tol_in) {}
    virtual ~MyISVDBase() = default;

    bool initialize(Eigen::MatrixXd& V, Eigen::VectorXd& s, const Eigen::VectorXd& c) const;
    virtual bool update(Eigen::MatrixXd& V, Eigen::VectorXd& s, const Eigen::VectorXd& c) const = 0;
    bool reorthogonalizeIfNeeded(Eigen::MatrixXd& V, Eigen::VectorXd& s) const;

protected:
    int max_rank = 50;
    double eps_svd = 1e-9;
    double tol = 1e-12;
    double orth_tol = 1e-10;

    void truncateState(Eigen::MatrixXd& V_new, Eigen::VectorXd& s_new) const;
    double orthError(const Eigen::MatrixXd& basis) const;
};

class MyEigenISVD final : public MyISVDBase {
public:
    using MyISVDBase::MyISVDBase;
    bool update(Eigen::MatrixXd& V, Eigen::VectorXd& s, const Eigen::VectorXd& c) const override;
};

class MyLapackISVD final : public MyISVDBase {
public:
    using MyISVDBase::MyISVDBase;
    bool update(Eigen::MatrixXd& V, Eigen::VectorXd& s, const Eigen::VectorXd& c) const override;
};

std::unique_ptr<MyISVDBase> create_isvd_solver(const std::string& backend_name,
                                               int max_rank,
                                               double eps_svd,
                                               double tol,
                                               double orth_tol);
std::unique_ptr<MyISVDBase> create_isvd_solver_from_environment(int max_rank,
                                                                double eps_svd,
                                                                double tol,
                                                                double orth_tol);

} // namespace MyAlgebra
