#pragma once

#include "ROM/Reduced_Basis_Agorithm/UpdateBase.hpp"

namespace stage2 {

class IncrementalSVD final : public UpdateBase {
public:
    enum class Backend {
        Eigen,
        MKL
    };

    int r_reset = 0;
    double eps_svd = 1e-9;
    double tol = 1e-12;
    double orth_tol = 1e-10;
    double energy_tol = 0.99999;
    int update_count = 0;
    bool store_snapshots = false;
    Backend backend = Backend::MKL;

    IncrementalSVD(int r_reset_in, double eps_svd_in, double tol_in, int max_rank_in);
    void update(const Eigen::VectorXd& solution, const ROMContext& context) override;
    void reset() override;

private:
    int resolveResetRank() const;
    bool initialize(const Eigen::VectorXd& c);
    bool rankOneUpdate(const Eigen::VectorXd& c);
    bool rankOneUpdateEigen(const Eigen::VectorXd& c);
    bool rankOneUpdateMKL(const Eigen::VectorXd& c);
    void truncateState(Eigen::MatrixXd& V_new, Eigen::VectorXd& s_new);
    void reorthogonalizeIfNeeded();
    double orthError(const Eigen::MatrixXd& basis) const;
    static Backend backendFromEnvironment();
};

} // namespace stage2
