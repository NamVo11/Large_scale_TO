#pragma once

#include <petscksp.h>

#include <memory>
#include <string>
#include <vector>

namespace stage2 {

class UpdateBase;

struct ROMContext {
    int analysis_id = 0;
    int opt_iter = 0;
    int solve_step = 0;
    int time_step = 0;
    double time = 0.0;
    std::string role = "primal";
    std::string system_type;
    bool is_end_of_analysis = false;
};

struct ROMInfo {
    std::string name;
    bool used_rom = false;
    bool rom_used = false;
    bool fom_used = false;
    bool accepted = false;
    int ksp_iterations = 0;
    double residual_norm = 0.0;
    int rank = 0;
    int rank_true = 0;
    double threshold = 0.0;
    double residual = 0.0;
    double residual_tol_eff = 0.0;
    double error_indicator = 0.0;
    double error = 0.0;
    double basis_error = 0.0;

    double timing_total = 0.0;
    double timing_petsc_to_eigen_vec = 0.0;
    double timing_petsc_to_eigen_mat = 0.0;
    double timing_projection = 0.0;
    double timing_reduced_rhs = 0.0;
    double timing_reduced_solve = 0.0;
    double timing_reconstruct = 0.0;
    double timing_residual = 0.0;
    double timing_fom_fallback = 0.0;
    double timing_fom_factorization = 0.0;
    double timing_fom_solve = 0.0;
    double timing_basis_update = 0.0;
    double timing_basis1_update = 0.0;
    double timing_online_basis_update = 0.0;
    double timing_end_basis_update = 0.0;
    double timing_basis_copy_reset = 0.0;
    double timing_snapshot = 0.0;
};

struct SolveStats {
    int ksp_iterations = 0;
    double residual_norm = 0.0;
    double factorization_time = 0.0;
    double solve_time = 0.0;
};

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
    virtual void set_factorization(Mat factor) = 0;

    void store_info(const ROMInfo& info, const ROMContext& context);
};

std::shared_ptr<ROMBase> make_fom_solver(const std::string& fom_solver_type);
std::shared_ptr<ROMBase> make_dynamic_ROM(const std::string& method_name,
                                          const std::string& fom_solver_type,
                                          double residual_tol,
                                          int rank1,
                                          int max_rank,
                                          bool online);

} // namespace stage2
