#include "ROM/Utils/rom_petsc_utils.hpp"

#include <algorithm>
#include <cmath>

namespace stage2 {

std::string lower_string(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

double compute_relative_residual(Mat lhs, Vec rhs, Vec solution) {
    Vec residual = nullptr;
    VecDuplicate(rhs, &residual);
    MatMult(lhs, solution, residual);
    VecAXPY(residual, -1.0, rhs);
    PetscReal residual_norm = 0.0;
    PetscReal rhs_norm = 0.0;
    VecNorm(residual, NORM_2, &residual_norm);
    VecNorm(rhs, NORM_2, &rhs_norm);
    VecDestroy(&residual);
    const double denom = std::max(static_cast<double>(rhs_norm), 1e-30);
    return static_cast<double>(residual_norm) / denom;
}

double compute_relative_solution_error(Vec u_fom, Vec u_rom) {
    Vec diff = nullptr;
    VecDuplicate(u_fom, &diff);
    VecCopy(u_fom, diff);
    VecAXPY(diff, -1.0, u_rom);
    PetscReal diff_norm = 0.0;
    PetscReal fom_norm = 0.0;
    VecNorm(diff, NORM_2, &diff_norm);
    VecNorm(u_fom, NORM_2, &fom_norm);
    VecDestroy(&diff);
    return static_cast<double>(diff_norm) / std::max(static_cast<double>(fom_norm), 1e-30);
}

double compute_basis_error(const std::vector<Vec>& V, Vec solution) {
    if (V.empty()) return 1.0;
    Vec projection = nullptr;
    VecDuplicate(solution, &projection);
    VecSet(projection, 0.0);
    for (Vec v : V) {
        PetscScalar coeff = 0.0;
        VecDot(v, solution, &coeff);
        VecAXPY(projection, coeff, v);
    }
    const double err = compute_relative_solution_error(solution, projection);
    VecDestroy(&projection);
    return err;
}

bool solve_dense_system(std::vector<double> A, std::vector<double> b, std::vector<double>& x) {
    const int n = static_cast<int>(b.size());
    x.assign(static_cast<std::size_t>(n), 0.0);
    if (n == 0) return false;
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double pivot_abs = std::abs(A[static_cast<std::size_t>(k * n + k)]);
        for (int i = k + 1; i < n; ++i) {
            const double v = std::abs(A[static_cast<std::size_t>(i * n + k)]);
            if (v > pivot_abs) {
                pivot_abs = v;
                pivot = i;
            }
        }
        if (pivot_abs <= 1e-14) return false;
        if (pivot != k) {
            for (int j = k; j < n; ++j) {
                std::swap(A[static_cast<std::size_t>(k * n + j)],
                          A[static_cast<std::size_t>(pivot * n + j)]);
            }
            std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot)]);
        }
        for (int i = k + 1; i < n; ++i) {
            const double factor = A[static_cast<std::size_t>(i * n + k)] /
                                  A[static_cast<std::size_t>(k * n + k)];
            A[static_cast<std::size_t>(i * n + k)] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                A[static_cast<std::size_t>(i * n + j)] -=
                    factor * A[static_cast<std::size_t>(k * n + j)];
            }
            b[static_cast<std::size_t>(i)] -= factor * b[static_cast<std::size_t>(k)];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[static_cast<std::size_t>(i * n + j)] * x[static_cast<std::size_t>(j)];
        }
        x[static_cast<std::size_t>(i)] = sum / A[static_cast<std::size_t>(i * n + i)];
    }
    return true;
}

bool symmetric_eigen_jacobi(std::vector<double> A,
                            int n,
                            std::vector<double>& eigenvalues,
                            std::vector<double>& eigenvectors) {
    eigenvalues.assign(static_cast<std::size_t>(n), 0.0);
    eigenvectors.assign(static_cast<std::size_t>(n * n), 0.0);
    if (n <= 0) return false;
    for (int i = 0; i < n; ++i) eigenvectors[static_cast<std::size_t>(i * n + i)] = 1.0;
    const int max_iter = std::max(50, 20 * n * n);
    for (int iter = 0; iter < max_iter; ++iter) {
        int p = 0;
        int q = n > 1 ? 1 : 0;
        double max_offdiag = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double v = std::abs(A[static_cast<std::size_t>(i * n + j)]);
                if (v > max_offdiag) {
                    max_offdiag = v;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_offdiag <= 1e-12) break;
        const double app = A[static_cast<std::size_t>(p * n + p)];
        const double aqq = A[static_cast<std::size_t>(q * n + q)];
        const double apq = A[static_cast<std::size_t>(p * n + q)];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) continue;
            const double akp = A[static_cast<std::size_t>(k * n + p)];
            const double akq = A[static_cast<std::size_t>(k * n + q)];
            const double new_kp = c * akp - s * akq;
            const double new_kq = s * akp + c * akq;
            A[static_cast<std::size_t>(k * n + p)] = new_kp;
            A[static_cast<std::size_t>(p * n + k)] = new_kp;
            A[static_cast<std::size_t>(k * n + q)] = new_kq;
            A[static_cast<std::size_t>(q * n + k)] = new_kq;
        }
        A[static_cast<std::size_t>(p * n + p)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[static_cast<std::size_t>(q * n + q)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[static_cast<std::size_t>(p * n + q)] = 0.0;
        A[static_cast<std::size_t>(q * n + p)] = 0.0;
        for (int k = 0; k < n; ++k) {
            const double vkp = eigenvectors[static_cast<std::size_t>(k * n + p)];
            const double vkq = eigenvectors[static_cast<std::size_t>(k * n + q)];
            eigenvectors[static_cast<std::size_t>(k * n + p)] = c * vkp - s * vkq;
            eigenvectors[static_cast<std::size_t>(k * n + q)] = s * vkp + c * vkq;
        }
    }
    for (int i = 0; i < n; ++i) eigenvalues[static_cast<std::size_t>(i)] = A[static_cast<std::size_t>(i * n + i)];
    std::vector<int> order(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return eigenvalues[static_cast<std::size_t>(a)] > eigenvalues[static_cast<std::size_t>(b)];
    });
    std::vector<double> sorted_values(static_cast<std::size_t>(n));
    std::vector<double> sorted_vectors(static_cast<std::size_t>(n * n));
    for (int col = 0; col < n; ++col) {
        const int src = order[static_cast<std::size_t>(col)];
        sorted_values[static_cast<std::size_t>(col)] = eigenvalues[static_cast<std::size_t>(src)];
        for (int row = 0; row < n; ++row) {
            sorted_vectors[static_cast<std::size_t>(row * n + col)] =
                eigenvectors[static_cast<std::size_t>(row * n + src)];
        }
    }
    eigenvalues.swap(sorted_values);
    eigenvectors.swap(sorted_vectors);
    return true;
}

} // namespace stage2

