#include "ROM/Reduced_Basis_Agorithm/MyPOD_SVD1.hpp"

#include "ROM/Utils/rom_petsc_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace stage2 {

MyPOD_SVD1::MyPOD_SVD1(double snapshot_window_in,
                       double tol_in,
                       int max_rank_in,
                       double energy_tol_in)
    : UpdateBase(max_rank_in) {
    name = "MyPOD_SVD_recover_energy_after_normalization";
    snapshot_window = snapshot_window_in;
    tol = tol_in;
    energy_tol = energy_tol_in;
}

void MyPOD_SVD1::update(Vec, const ROMContext&) {
    clear_vectors(V);
    s.clear();
    W.clear();
    if (max_rank <= 0 || snapshots.empty()) return;

    std::vector<int> valid;
    std::vector<double> snapshot_norm;
    valid.reserve(snapshots.size());
    snapshot_norm.reserve(snapshots.size());
    for (std::size_t i = 0; i < snapshots.size(); ++i) {
        PetscReal nrm = 0.0;
        VecNorm(snapshots[i], NORM_2, &nrm);
        if (nrm > 1e-14) {
            valid.push_back(static_cast<int>(i));
            snapshot_norm.push_back(static_cast<double>(nrm));
        }
    }
    const int m = static_cast<int>(valid.size());
    if (m == 0) return;

    std::vector<double> gram(static_cast<std::size_t>(m * m), 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            PetscScalar dot = 0.0;
            VecDot(snapshots[static_cast<std::size_t>(valid[static_cast<std::size_t>(i)])],
                   snapshots[static_cast<std::size_t>(valid[static_cast<std::size_t>(j)])],
                   &dot);
            const double value = PetscRealPart(dot) /
                                 (snapshot_norm[static_cast<std::size_t>(i)] *
                                  snapshot_norm[static_cast<std::size_t>(j)]);
            gram[static_cast<std::size_t>(i * m + j)] = value;
            gram[static_cast<std::size_t>(j * m + i)] = value;
        }
    }

    std::vector<double> lambda;
    std::vector<double> W_svd;
    if (!symmetric_eigen_jacobi(std::move(gram), m, lambda, W_svd)) return;
    const double lambda_max = std::max(0.0, lambda.front());
    if (lambda_max <= 0.0) return;

    int r_new = 0;
    if (energy_tol > 0.0) {
        const double total = std::accumulate(lambda.begin(), lambda.end(), 0.0);
        double cumulative = 0.0;
        for (double value : lambda) {
            cumulative += std::max(0.0, value);
            ++r_new;
            if (total > 0.0 && cumulative / total >= energy_tol) break;
        }
    } else {
        for (double value : lambda) {
            if (value > tol * lambda_max) ++r_new;
        }
    }
    r_new = std::min({max_rank, r_new, m});
    if (r_new <= 0) return;

    std::vector<Vec> V_new(static_cast<std::size_t>(r_new), nullptr);
    for (int k = 0; k < r_new; ++k) {
        VecDuplicate(snapshots[static_cast<std::size_t>(valid.front())], &V_new[static_cast<std::size_t>(k)]);
        VecSet(V_new[static_cast<std::size_t>(k)], 0.0);
        const double sigma = std::sqrt(std::max(lambda[static_cast<std::size_t>(k)], 0.0));
        if (sigma <= 1e-14) continue;
        for (int j = 0; j < m; ++j) {
            const double coeff = W_svd[static_cast<std::size_t>(j * m + k)] /
                                 (sigma * snapshot_norm[static_cast<std::size_t>(j)]);
            VecAXPY(V_new[static_cast<std::size_t>(k)], coeff,
                    snapshots[static_cast<std::size_t>(valid[static_cast<std::size_t>(j)])]);
        }
    }

    std::vector<double> cct(static_cast<std::size_t>(r_new * r_new), 0.0);
    for (int a = 0; a < r_new; ++a) {
        for (int b = a; b < r_new; ++b) {
            double value = 0.0;
            for (int j = 0; j < m; ++j) {
                const double ca = std::sqrt(std::max(lambda[static_cast<std::size_t>(a)], 0.0)) *
                                  W_svd[static_cast<std::size_t>(j * m + a)] *
                                  snapshot_norm[static_cast<std::size_t>(j)];
                const double cb = std::sqrt(std::max(lambda[static_cast<std::size_t>(b)], 0.0)) *
                                  W_svd[static_cast<std::size_t>(j * m + b)] *
                                  snapshot_norm[static_cast<std::size_t>(j)];
                value += ca * cb;
            }
            cct[static_cast<std::size_t>(a * r_new + b)] = value;
            cct[static_cast<std::size_t>(b * r_new + a)] = value;
        }
    }

    std::vector<double> s_energy;
    std::vector<double> P_raw;
    if (!symmetric_eigen_jacobi(std::move(cct), r_new, s_energy, P_raw)) {
        clear_vectors(V_new);
        return;
    }
    for (double& value : s_energy) value = std::sqrt(std::max(value, 0.0));

    const int r_keep = std::min({r_new, static_cast<int>(s_energy.size())});
    for (int k = 0; k < r_keep; ++k) {
        Vec v = nullptr;
        VecDuplicate(snapshots[static_cast<std::size_t>(valid.front())], &v);
        VecSet(v, 0.0);
        for (int a = 0; a < r_new; ++a) {
            VecAXPY(v, P_raw[static_cast<std::size_t>(a * r_new + k)],
                    V_new[static_cast<std::size_t>(a)]);
        }
        PetscReal nrm = 0.0;
        VecNorm(v, NORM_2, &nrm);
        if (nrm > 1e-10) {
            VecScale(v, 1.0 / nrm);
            V.push_back(v);
            s.push_back(s_energy[static_cast<std::size_t>(k)]);
        } else {
            VecDestroy(&v);
        }
    }
    clear_vectors(V_new);
}

} // namespace stage2

