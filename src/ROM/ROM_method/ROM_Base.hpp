#pragma once

#include "ROM/ROMTypes.hpp"

#include <Eigen/Sparse>
#include <petscksp.h>

#include <memory>
#include <string>
#include <vector>

class UpdateBase;
class FactorizedSolver;

class ROMBase {
public:
    bool use_mor = true;
    std::vector<std::shared_ptr<UpdateBase>> basis_list;
    std::string name = "ROM_Base";
    std::vector<std::vector<ROMInfo>> ROM_info;

    explicit ROMBase(std::vector<std::shared_ptr<UpdateBase>> basis_list_in = {});
    virtual ~ROMBase() = default;

    virtual SolveStats main(const Eigen::SparseMatrix<double>& lhs,
                            const Eigen::VectorXd& rhs,
                            Eigen::VectorXd& u,
                            const ROMContext& context) = 0;
    virtual SolveStats FOM_main(const Eigen::SparseMatrix<double>& lhs,
                                const Eigen::VectorXd& rhs,
                                Eigen::VectorXd& u,
                                const ROMContext& context);
    virtual std::shared_ptr<FactorizedSolver> get_factorized_solver(const Eigen::SparseMatrix<double>& lhs) const;
    virtual void set_factorized_solver(const Eigen::SparseMatrix<double>& lhs,
                                       std::shared_ptr<FactorizedSolver> solver);
    virtual KSP get_petsc_factorized_solver(Mat lhs) const;
    virtual void set_petsc_factorized_solver(Mat lhs, KSP ksp);
    virtual void set_petsc_system(Mat lhs, Vec rhs, Vec u);

    void store_info(const ROMInfo& info, const ROMContext& context);
};

std::shared_ptr<ROMBase> make_fom_solver(const std::string& fom_solver_type,
                                         const std::string& linear_backend = "eigen");


std::shared_ptr<ROMBase> make_dynamic_ROM(const std::string& method_name,
                                          const std::string& fom_solver_type,
                                          double residual_tol,
                                          int rank1,
                                          int max_rank,
                                          bool online,
                                          const std::string& linear_backend = "eigen");
