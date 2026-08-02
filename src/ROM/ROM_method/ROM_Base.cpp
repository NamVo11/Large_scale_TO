#include "ROM/ROM_method/ROM_Base.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>


ROMBase::ROMBase(std::vector<std::shared_ptr<UpdateBase>> basis_list_in)
    : basis_list(std::move(basis_list_in)) {}

SolveStats ROMBase::FOM_main(const Eigen::SparseMatrix<double>&,
                             const Eigen::VectorXd&,
                             Eigen::VectorXd&,
                             const ROMContext&) {
    throw std::runtime_error("FOM_main is not implemented for " + name);
}

std::shared_ptr<FactorizedSolver> ROMBase::get_factorized_solver(const Eigen::SparseMatrix<double>&) const {
    return nullptr;
}

void ROMBase::set_factorized_solver(const Eigen::SparseMatrix<double>&,
                                    std::shared_ptr<FactorizedSolver>) {
}

KSP ROMBase::get_petsc_factorized_solver(Mat) const {
    return nullptr;
}

void ROMBase::set_petsc_factorized_solver(Mat, KSP) {
}

void ROMBase::set_petsc_system(Mat, Vec, Vec) {
}

void ROMBase::store_info(const ROMInfo& info, const ROMContext& context) {
    const std::size_t iter = static_cast<std::size_t>(std::max(0, context.opt_iter));
    if (ROM_info.size() <= iter) ROM_info.resize(iter + 1);
    ROM_info[iter].push_back(info);
}
