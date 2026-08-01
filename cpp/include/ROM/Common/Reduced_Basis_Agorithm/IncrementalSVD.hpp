#pragma once

#include "ROM/Common/Reduced_Basis_Agorithm/UpdateBase.hpp"
#include "helper/MyAlgebra.hpp"

#include <memory>

class IncrementalSVD final : public UpdateBase {
public:
    int r_reset = 0;
    double eps_svd = 1e-9;
    double tol = 1e-12;
    double orth_tol = 1e-10;
    double energy_tol = 0.99999;
    bool store_snapshots = false;
    std::unique_ptr<MyAlgebra::MyISVDBase> ISVD_solver;

    IncrementalSVD(int r_reset_in, double eps_svd_in, double tol_in, int max_rank_in)
        : UpdateBase(max_rank_in),
          r_reset(r_reset_in),
          eps_svd(eps_svd_in),
          tol(tol_in),
          ISVD_solver(MyAlgebra::create_isvd_solver_from_environment(max_rank_in,
                                                                     eps_svd_in,
                                                                     tol_in,
                                                                     orth_tol)) {
        name = "IncrementalSVD";
    }

    void update(const Eigen::VectorXd& solution, const ROMContext& context) override;
    void reset() override;

private:
    int resolveResetRank() const;
};
