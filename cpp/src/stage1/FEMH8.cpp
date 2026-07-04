#include "stage1/FEMH8.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace {

using Mat3 = std::array<double, 9>;

double det3(const Mat3& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7])
         - a[1] * (a[3] * a[8] - a[5] * a[6])
         + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

Mat3 inv3(const Mat3& a) {
    double d = det3(a);
    if (std::abs(d) < 1e-30) throw std::runtime_error("Singular H8 Jacobian");
    Mat3 b{};
    b[0] =  (a[4] * a[8] - a[5] * a[7]) / d;
    b[1] = -(a[1] * a[8] - a[2] * a[7]) / d;
    b[2] =  (a[1] * a[5] - a[2] * a[4]) / d;
    b[3] = -(a[3] * a[8] - a[5] * a[6]) / d;
    b[4] =  (a[0] * a[8] - a[2] * a[6]) / d;
    b[5] = -(a[0] * a[5] - a[2] * a[3]) / d;
    b[6] =  (a[3] * a[7] - a[4] * a[6]) / d;
    b[7] = -(a[0] * a[7] - a[1] * a[6]) / d;
    b[8] =  (a[0] * a[4] - a[1] * a[3]) / d;
    return b;
}

void h8Shape(double xi, double eta, double zeta,
             std::array<double, 8>& N,
             std::array<double, 8 * 3>& dNdxi) {
    constexpr int s[8][3] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}
    };

    for (int i = 0; i < 8; ++i) {
        const double sx = static_cast<double>(s[i][0]);
        const double sy = static_cast<double>(s[i][1]);
        const double sz = static_cast<double>(s[i][2]);
        N[i] = 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta);
        dNdxi[i * 3 + 0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);
        dNdxi[i * 3 + 1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta);
        dNdxi[i * 3 + 2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta);
    }
}

void addBtdb(std::array<double, 24 * 24>& ke,
             const std::array<double, 6 * 24>& B,
             const std::array<double, 6 * 6>& D,
             double weight) {
    for (int i = 0; i < 24; ++i) {
        for (int j = 0; j < 24; ++j) {
            double v = 0.0;
            for (int a = 0; a < 6; ++a) {
                double db = 0.0;
                for (int b = 0; b < 6; ++b) db += D[a * 6 + b] * B[b * 24 + j];
                v += B[a * 24 + i] * db;
            }
            ke[i * 24 + j] += v * weight;
        }
    }
}

} // namespace

double effectiveE(const MaterialData& mat, double x) {
    return mat.Emin + std::pow(x, mat.penalE) * (mat.E0 - mat.Emin);
}

double effectiveRho(const MaterialData& mat, double x) {
    return mat.rhomin + std::pow(x, mat.penalRho) * (mat.rho0 - mat.rhomin);
}

ElementMatrices computeH8ElementMatrices(const CaseData& data, int elemId) {
    ElementMatrices out;
    const auto& elem = data.elements.at(static_cast<std::size_t>(elemId));
    const double nu = data.material.nu;
    const double c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu));
    std::array<double, 6 * 6> D{};
    D[0 * 6 + 0] = c * (1.0 - nu);
    D[0 * 6 + 1] = c * nu;
    D[0 * 6 + 2] = c * nu;
    D[1 * 6 + 0] = c * nu;
    D[1 * 6 + 1] = c * (1.0 - nu);
    D[1 * 6 + 2] = c * nu;
    D[2 * 6 + 0] = c * nu;
    D[2 * 6 + 1] = c * nu;
    D[2 * 6 + 2] = c * (1.0 - nu);
    D[3 * 6 + 3] = c * (1.0 - 2.0 * nu) / 2.0;
    D[4 * 6 + 4] = c * (1.0 - 2.0 * nu) / 2.0;
    D[5 * 6 + 5] = c * (1.0 - 2.0 * nu) / 2.0;

    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};
    for (double xi : gp) {
        for (double eta : gp) {
            for (double zeta : gp) {
                std::array<double, 8> N{};
                std::array<double, 8 * 3> dNdxi{};
                h8Shape(xi, eta, zeta, N, dNdxi);

                Mat3 J{};
                for (int a = 0; a < 8; ++a) {
                    const auto& p = data.nodes.at(static_cast<std::size_t>(elem[a]));
                    for (int alpha = 0; alpha < 3; ++alpha) {
                        J[0 * 3 + alpha] += p[0] * dNdxi[a * 3 + alpha];
                        J[1 * 3 + alpha] += p[1] * dNdxi[a * 3 + alpha];
                        J[2 * 3 + alpha] += p[2] * dNdxi[a * 3 + alpha];
                    }
                }
                const double detJ = std::abs(det3(J));
                const Mat3 invJ = inv3(J);

                std::array<double, 8 * 3> dNdX{};
                for (int a = 0; a < 8; ++a) {
                    for (int beta = 0; beta < 3; ++beta) {
                        double v = 0.0;
                        for (int alpha = 0; alpha < 3; ++alpha) {
                            v += dNdxi[a * 3 + alpha] * invJ[alpha * 3 + beta];
                        }
                        dNdX[a * 3 + beta] = v;
                    }
                }

                std::array<double, 6 * 24> B{};
                for (int a = 0; a < 8; ++a) {
                    const int col = 3 * a;
                    const double dx = dNdX[a * 3 + 0];
                    const double dy = dNdX[a * 3 + 1];
                    const double dz = dNdX[a * 3 + 2];
                    B[0 * 24 + col + 0] = dx;
                    B[1 * 24 + col + 1] = dy;
                    B[2 * 24 + col + 2] = dz;
                    B[3 * 24 + col + 0] = dy;
                    B[3 * 24 + col + 1] = dx;
                    B[4 * 24 + col + 1] = dz;
                    B[4 * 24 + col + 2] = dy;
                    B[5 * 24 + col + 0] = dz;
                    B[5 * 24 + col + 2] = dx;
                }
                addBtdb(out.ke, B, D, detJ);

                for (int a = 0; a < 8; ++a) {
                    for (int b = 0; b < 8; ++b) {
                        const double v = N[a] * N[b] * detJ;
                        for (int d = 0; d < 3; ++d) {
                            out.me[(3 * a + d) * 24 + (3 * b + d)] += v;
                        }
                    }
                }
            }
        }
    }
    return out;
}
