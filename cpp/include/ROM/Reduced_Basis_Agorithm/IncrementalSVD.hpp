#pragma once

#include "ROM/Reduced_Basis_Agorithm/UpdateBase.hpp"

namespace stage2 {

class IncrementalSVD final : public UpdateBase {
public:
    int r_reset = 0;
    double eps_svd = 1e-9;
    double tol = 1e-12;
    double orth_tol = 1e-10;
    double energy_tol = 0.99999;
    int update_count = 0;

    IncrementalSVD(int r_reset_in, double eps_svd_in, double tol_in, int max_rank_in);
    void update(Vec solution, const ROMContext& context) override;
    void reset() override;

private:
    int resolveResetRank() const;
    bool initialize(Vec c);
    bool rankOneUpdate(Vec c);
    void truncateState(std::vector<Vec>& V_new, std::vector<double>& s_new);
    void reorthogonalizeIfNeeded();
    double orthError(const std::vector<Vec>& basis) const;
};

} // namespace stage2
