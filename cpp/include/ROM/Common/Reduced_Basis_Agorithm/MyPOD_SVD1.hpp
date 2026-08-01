#pragma once

#include "ROM/Common/Reduced_Basis_Agorithm/UpdateBase.hpp"
#include "helper/MyAlgebra.hpp"

#include <memory>

class MyPOD_SVD1 final : public UpdateBase {
public:
    double tol = 1e-12;
    double energy_tol = 0.0;
    double snapshot_window = 0.0; // 0 means Inf, matching current MATLAB usage.
    std::unique_ptr<MyAlgebra::MyPODBase> pod_solver;

    MyPOD_SVD1(double snapshot_window_in,
               double tol_in,
               int max_rank_in,
               double energy_tol_in = 0.0)
        : UpdateBase(max_rank_in),
          tol(tol_in),
          energy_tol(energy_tol_in),
          snapshot_window(snapshot_window_in),
          pod_solver(MyAlgebra::create_pod_solver_from_environment()) {
        name = "MyPOD_SVD_recover_energy_after_normalization";
    }

    void update(const Eigen::VectorXd& solution, const ROMContext& context) override;
};
