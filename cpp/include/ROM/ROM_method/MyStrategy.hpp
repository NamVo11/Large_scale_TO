#pragma once

#include "ROM/Error_Estimator/ErrorEstimatorBase.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"

#include <memory>

namespace stage2 {

class MyStrategy final : public ROMBase {
public:
    double residual_tol = 1e-3;
    int n_p = 0;
    std::shared_ptr<ROMBase> fom_solver;
    std::shared_ptr<ErrorEstimatorBase> error_estimator;
    std::vector<double> A_r_cache;
    int fom_flag = 1;
    double aaa = 0.0;
    double bbbb = 0.0;
    bool online = true;

    MyStrategy(std::vector<std::shared_ptr<UpdateBase>> basis_list_in,
               double residual_tol_in,
               int n_p_in,
               std::shared_ptr<ROMBase> fom_solver_in,
               std::shared_ptr<ErrorEstimatorBase> error_estimator_in,
               bool online_in);

    SolveStats main(Mat lhs, Vec rhs, Vec u, const ROMContext& context) override;
    void set_factorization(Mat factor) override;

private:
    Mat A_r_lhs_cache_ = nullptr;
    int A_r_rank_cache_ = 0;

    bool solve_rom(Mat lhs, Vec rhs, Vec u_rom, const std::vector<Vec>& V, SolveStats& stats);
    void clear_reduced_cache();
};

} // namespace stage2

