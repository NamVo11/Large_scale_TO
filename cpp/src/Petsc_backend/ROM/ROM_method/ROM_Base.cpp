#include "Petsc_backend/ROM/ROM_method/ROM_Base.hpp"

#include <algorithm>
#include <stdexcept>

namespace petsc_backend {

ROMBase::ROMBase(std::vector<std::shared_ptr<UpdateBase>> basis_list_in)
    : basis_list(std::move(basis_list_in)) {}

SolveStats ROMBase::main_test(Mat lhs, Vec rhs, Vec u, const ROMContext& context) {
    return main(lhs, rhs, u, context);
}

SolveStats ROMBase::FOM_main(Mat, Vec, Vec, const ROMContext&) {
    throw std::runtime_error(name + " does not implement FOM_main");
}

KSP ROMBase::get_factorized_solver(Mat) const {
    return nullptr;
}

void ROMBase::set_factorized_solver(Mat, KSP) {
}

void ROMBase::store_info(const ROMInfo& info, const ROMContext& context) {
    const int row = std::max(0, context.opt_iter);
    const int col = std::max(0, context.time_step);
    if (static_cast<int>(ROM_info.size()) <= row) ROM_info.resize(static_cast<std::size_t>(row + 1));
    if (static_cast<int>(ROM_info[static_cast<std::size_t>(row)].size()) <= col) {
        ROM_info[static_cast<std::size_t>(row)].resize(static_cast<std::size_t>(col + 1));
    }
    ROM_info[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = info;
}

} // namespace petsc_backend
