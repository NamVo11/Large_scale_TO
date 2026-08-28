#pragma once

#include "ROM/ROM_method/ROM_Base.hpp"

#ifdef EIGEN_USE_LAPACKE
#include <Eigen/PardisoSupport>
#else
#include <Eigen/SparseCholesky>
#endif

#include <memory>
#include <string>

class FactorizedSolver {
public:
#ifdef EIGEN_USE_LAPACKE
    Eigen::PardisoLLT<Eigen::SparseMatrix<double>> solver;
#else
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
#endif
};

class EigenFOM_direct final : public ROMBase {
public:
    EigenFOM_direct();

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u) override;
    SolveStats FOM_main(const Eigen::SparseMatrix<double>& lhs,
                        const Eigen::VectorXd& rhs,
                        Eigen::VectorXd& u) override;
    std::shared_ptr<FactorizedSolver> get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const override;
    void set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                               std::shared_ptr<FactorizedSolver> solver) override;

private:
    const Eigen::SparseMatrix<double>* factor_ = nullptr;
    std::shared_ptr<FactorizedSolver> solver_;
    double last_factorization_time_ = 0.0;
    void set_factorization(const Eigen::SparseMatrix<double>& lhs);
};

class EigenFOM_iterative final : public ROMBase {
public:
    std::string method = "pcg";
    double pcg_tol = 1e-8;
    int pcg_maxit = 1000;

    explicit EigenFOM_iterative(std::string method_in = "pcg");

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u) override;
    SolveStats FOM_main(const Eigen::SparseMatrix<double>& lhs,
                        const Eigen::VectorXd& rhs,
                        Eigen::VectorXd& u) override;
};

class PetscFOM_direct final : public ROMBase {
public:
    Mat factor = nullptr;

    PetscFOM_direct();
    ~PetscFOM_direct() override;

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u) override;
    SolveStats FOM_main(const Eigen::SparseMatrix<double>& lhs,
                        const Eigen::VectorXd& rhs,
                        Eigen::VectorXd& u) override;
    KSP get_petsc_factorized_solver(Mat lhs) const override;
    void set_petsc_factorized_solver(Mat lhs, KSP ksp) override;
    void set_petsc_system(Mat lhs, Vec rhs, Vec u) override;

private:
    KSP ksp_ = nullptr;
    Mat current_lhs_ = nullptr;
    Vec current_rhs_ = nullptr;
    Vec current_u_ = nullptr;
    double last_factorization_time_ = 0.0;

    void set_factorization(Mat factor_in);
};

class PetscFOM_iterative final : public ROMBase {
public:
    std::string method = "pcg";
    double pcg_tol = 1e-8;
    int pcg_maxit = 1000;

    explicit PetscFOM_iterative(std::string method_in = "pcg");
    ~PetscFOM_iterative() override;

    SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                    const Eigen::VectorXd& rhs,
                    Eigen::VectorXd& u) override;
    SolveStats FOM_main(const Eigen::SparseMatrix<double>& lhs,
                        const Eigen::VectorXd& rhs,
                        Eigen::VectorXd& u) override;
    KSP get_petsc_factorized_solver(Mat lhs) const override;
    void set_petsc_factorized_solver(Mat lhs, KSP ksp) override;
    void set_petsc_system(Mat lhs, Vec rhs, Vec u) override;

private:
    KSP ksp_ = nullptr;
    Mat factor = nullptr;
    Mat current_lhs_ = nullptr;
    Vec current_rhs_ = nullptr;
    Vec current_u_ = nullptr;
    double last_factorization_time_ = 0.0;

    void set_factorization(Mat factor_in);
};
