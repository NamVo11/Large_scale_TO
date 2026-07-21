#pragma once

#include "ROM/Common/ROMTypes.hpp"

#include <petscksp.h>

#include <memory>
#include <string>
#include <vector>

class UpdateBase;

namespace petsc_backend {

class ROMBase {
public:
    bool use_mor = true;
    std::vector<std::shared_ptr<UpdateBase>> basis_list;
    std::string name = "ROM_Base";
    std::vector<std::vector<ROMInfo>> ROM_info;

    explicit ROMBase(std::vector<std::shared_ptr<UpdateBase>> basis_list_in = {});
    virtual ~ROMBase() = default;

    virtual SolveStats main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) = 0;
    virtual SolveStats main_test(Mat lhs, Vec rhs, Vec u, const ROMContext& context);
    virtual SolveStats FOM_main(Mat lhs, Vec rhs, Vec u, const ROMContext& context);
    virtual KSP get_factorized_solver(Mat lhs) const;
    virtual void set_factorized_solver(Mat lhs, KSP ksp);

    void store_info(const ROMInfo& info, const ROMContext& context);
};

std::shared_ptr<ROMBase> make_fom_solver(const std::string& fom_solver_type);
std::shared_ptr<ROMBase> make_dynamic_ROM(const std::string& method_name,
                                          const std::string& fom_solver_type,
                                          double residual_tol,
                                          int rank1,
                                          int max_rank,
                                          bool online);

} // namespace petsc_backend
