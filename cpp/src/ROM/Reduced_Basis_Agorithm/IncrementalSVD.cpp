#include "ROM/Reduced_Basis_Agorithm/IncrementalSVD.hpp"

#include "ROM/Utils/rom_petsc_utils.hpp"

#include <algorithm>
#include <cmath>

namespace stage2 {

IncrementalSVD::IncrementalSVD(int r_reset_in, double eps_svd_in, double tol_in, int max_rank_in)
    : UpdateBase(max_rank_in) {
    name = "IncrementalSVD";
    r_reset = r_reset_in;
    eps_svd = eps_svd_in;
    tol = tol_in;
}

void IncrementalSVD::reset() {
    UpdateBase::reset();
    update_count = 0;
}

int IncrementalSVD::resolveResetRank() const {
    return r_reset > 0 ? r_reset : max_rank;
}

bool IncrementalSVD::initialize(Vec c) {
    clear_vectors(V);
    s.clear();
    PetscReal c_norm = 0.0;
    VecNorm(c, NORM_2, &c_norm);
    if (c_norm < PETSC_MACHINE_EPSILON) return false;

    Vec v = duplicate_vec(c);
    VecScale(v, 1.0 / c_norm);
    V.push_back(v);
    s.push_back(static_cast<double>(c_norm));
    return true;
}

void IncrementalSVD::truncateState(std::vector<Vec>& V_new, std::vector<double>& s_new) {
    if (s_new.empty()) {
        clear_vectors(V_new);
        return;
    }
    const double s_max = *std::max_element(s_new.begin(), s_new.end());
    if (s_max <= 0.0) {
        clear_vectors(V_new);
        s_new.clear();
        return;
    }

    int keep_count = 0;
    for (double value : s_new) {
        if (value > tol * s_max) ++keep_count;
    }
    const int r = std::min({max_rank, keep_count, static_cast<int>(V_new.size()), static_cast<int>(s_new.size())});
    for (int i = r; i < static_cast<int>(V_new.size()); ++i) {
        if (V_new[static_cast<std::size_t>(i)]) VecDestroy(&V_new[static_cast<std::size_t>(i)]);
    }
    V_new.resize(static_cast<std::size_t>(std::max(r, 0)));
    s_new.resize(static_cast<std::size_t>(std::max(r, 0)));
}

bool IncrementalSVD::rankOneUpdate(Vec c) {
    const int r_old = static_cast<int>(V.size());
    std::vector<double> ell(static_cast<std::size_t>(r_old), 0.0);
    Vec h = duplicate_vec(c);
    for (int i = 0; i < r_old; ++i) {
        PetscScalar dot = 0.0;
        VecDot(V[static_cast<std::size_t>(i)], c, &dot);
        ell[static_cast<std::size_t>(i)] = PetscRealPart(dot);
        VecAXPY(h, -ell[static_cast<std::size_t>(i)], V[static_cast<std::size_t>(i)]);
    }

    PetscReal p_raw = 0.0;
    VecNorm(h, NORM_2, &p_raw);
    const double p = static_cast<double>(p_raw);
    const bool update_flag = p > eps_svd;
    const int q_rows = update_flag ? r_old + 1 : r_old;
    const int q_cols = r_old + 1;
    if (q_rows <= 0) {
        VecDestroy(&h);
        return false;
    }

    std::vector<Vec> V_aug;
    V_aug.reserve(static_cast<std::size_t>(q_rows));
    for (Vec v : V) V_aug.push_back(duplicate_vec(v));
    if (update_flag) {
        VecScale(h, 1.0 / p);
        V_aug.push_back(h);
        h = nullptr;
    }
    if (h) VecDestroy(&h);

    std::vector<double> q(static_cast<std::size_t>(q_rows * q_cols), 0.0);
    for (int i = 0; i < r_old; ++i) {
        q[static_cast<std::size_t>(i * q_cols + i)] = s[static_cast<std::size_t>(i)];
        q[static_cast<std::size_t>(i * q_cols + r_old)] = ell[static_cast<std::size_t>(i)];
    }
    if (update_flag) {
        q[static_cast<std::size_t>(r_old * q_cols + r_old)] = p;
    }

    std::vector<double> qqt(static_cast<std::size_t>(q_rows * q_rows), 0.0);
    for (int i = 0; i < q_rows; ++i) {
        for (int j = i; j < q_rows; ++j) {
            double value = 0.0;
            for (int k = 0; k < q_cols; ++k) {
                value += q[static_cast<std::size_t>(i * q_cols + k)] *
                         q[static_cast<std::size_t>(j * q_cols + k)];
            }
            qqt[static_cast<std::size_t>(i * q_rows + j)] = value;
            qqt[static_cast<std::size_t>(j * q_rows + i)] = value;
        }
    }

    std::vector<double> lambda;
    std::vector<double> Uq;
    if (!symmetric_eigen_jacobi(std::move(qqt), q_rows, lambda, Uq)) {
        clear_vectors(V_aug);
        return false;
    }

    std::vector<double> s_new(static_cast<std::size_t>(q_rows), 0.0);
    for (int i = 0; i < q_rows; ++i) {
        s_new[static_cast<std::size_t>(i)] = std::sqrt(std::max(0.0, lambda[static_cast<std::size_t>(i)]));
    }

    std::vector<Vec> V_new;
    V_new.reserve(static_cast<std::size_t>(q_rows));
    for (int k = 0; k < q_rows; ++k) {
        Vec v = nullptr;
        VecDuplicate(V_aug.front(), &v);
        VecSet(v, 0.0);
        for (int a = 0; a < q_rows; ++a) {
            VecAXPY(v, Uq[static_cast<std::size_t>(a * q_rows + k)], V_aug[static_cast<std::size_t>(a)]);
        }
        V_new.push_back(v);
    }

    clear_vectors(V_aug);
    truncateState(V_new, s_new);
    clear_vectors(V);
    V.swap(V_new);
    s.swap(s_new);
    clear_vectors(V_new);
    return update_flag;
}

double IncrementalSVD::orthError(const std::vector<Vec>& basis) const {
    const int r = static_cast<int>(basis.size());
    if (r == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < r; ++j) {
            PetscScalar dot = 0.0;
            VecDot(basis[static_cast<std::size_t>(i)], basis[static_cast<std::size_t>(j)], &dot);
            const double target = i == j ? 1.0 : 0.0;
            const double diff = PetscRealPart(dot) - target;
            sum += diff * diff;
        }
    }
    return std::sqrt(sum);
}

void IncrementalSVD::reorthogonalizeIfNeeded() {
    if (V.empty() || orthError(V) <= orth_tol) return;

    std::vector<Vec> q_basis;
    std::vector<double> rmat(static_cast<std::size_t>(V.size() * V.size()), 0.0);
    q_basis.reserve(V.size());
    for (int j = 0; j < static_cast<int>(V.size()); ++j) {
        Vec q = duplicate_vec(V[static_cast<std::size_t>(j)]);
        for (int i = 0; i < static_cast<int>(q_basis.size()); ++i) {
            PetscScalar dot = 0.0;
            VecDot(q_basis[static_cast<std::size_t>(i)], q, &dot);
            rmat[static_cast<std::size_t>(i * V.size() + j)] = PetscRealPart(dot);
            VecAXPY(q, -PetscRealPart(dot), q_basis[static_cast<std::size_t>(i)]);
        }
        PetscReal nrm = 0.0;
        VecNorm(q, NORM_2, &nrm);
        rmat[static_cast<std::size_t>(q_basis.size() * V.size() + j)] = static_cast<double>(nrm);
        if (nrm > 1e-14) {
            VecScale(q, 1.0 / nrm);
            q_basis.push_back(q);
        } else {
            VecDestroy(&q);
        }
    }

    const int q_rows = static_cast<int>(q_basis.size());
    const int q_cols = static_cast<int>(V.size());
    if (q_rows == 0) {
        clear_vectors(q_basis);
        clear_vectors(V);
        s.clear();
        return;
    }

    std::vector<double> m(static_cast<std::size_t>(q_rows * q_cols), 0.0);
    for (int i = 0; i < q_rows; ++i) {
        for (int j = 0; j < q_cols; ++j) {
            m[static_cast<std::size_t>(i * q_cols + j)] =
                rmat[static_cast<std::size_t>(i * V.size() + j)] * s[static_cast<std::size_t>(j)];
        }
    }

    std::vector<double> mmt(static_cast<std::size_t>(q_rows * q_rows), 0.0);
    for (int i = 0; i < q_rows; ++i) {
        for (int j = i; j < q_rows; ++j) {
            double value = 0.0;
            for (int k = 0; k < q_cols; ++k) {
                value += m[static_cast<std::size_t>(i * q_cols + k)] *
                         m[static_cast<std::size_t>(j * q_cols + k)];
            }
            mmt[static_cast<std::size_t>(i * q_rows + j)] = value;
            mmt[static_cast<std::size_t>(j * q_rows + i)] = value;
        }
    }

    std::vector<double> lambda;
    std::vector<double> Um;
    if (!symmetric_eigen_jacobi(std::move(mmt), q_rows, lambda, Um)) {
        clear_vectors(q_basis);
        return;
    }

    std::vector<Vec> V_new;
    std::vector<double> s_new(static_cast<std::size_t>(q_rows), 0.0);
    for (int k = 0; k < q_rows; ++k) {
        Vec v = nullptr;
        VecDuplicate(q_basis.front(), &v);
        VecSet(v, 0.0);
        for (int a = 0; a < q_rows; ++a) {
            VecAXPY(v, Um[static_cast<std::size_t>(a * q_rows + k)], q_basis[static_cast<std::size_t>(a)]);
        }
        V_new.push_back(v);
        s_new[static_cast<std::size_t>(k)] = std::sqrt(std::max(0.0, lambda[static_cast<std::size_t>(k)]));
    }

    truncateState(V_new, s_new);
    clear_vectors(V);
    V.swap(V_new);
    s.swap(s_new);
    clear_vectors(V_new);
    clear_vectors(q_basis);
}

void IncrementalSVD::update(Vec solution, const ROMContext& context) {
    addSnapshot(solution, context);
    const int rank_old = static_cast<int>(V.size());
    const int r_reset_eff = resolveResetRank();
    const bool reset_flag = V.empty() || s.empty() || rank_old >= r_reset_eff;
    bool update_flag = false;
    if (reset_flag) {
        update_flag = initialize(solution);
        update_count = 0;
    } else {
        update_flag = rankOneUpdate(solution);
    }
    if (update_flag) ++update_count;
    reorthogonalizeIfNeeded();
}

} // namespace stage2
