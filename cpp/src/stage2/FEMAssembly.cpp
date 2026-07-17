#include "stage2/FEMAssembly.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace stage2 {
namespace {

using Mat3 = std::array<double, 9>;

double det3(const Mat3& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7])
         - a[1] * (a[3] * a[8] - a[5] * a[6])
         + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

Mat3 inv3(const Mat3& a) {
    const double d = det3(a);
    if (std::abs(d) < 1e-30) throw std::runtime_error("Singular H8 Jacobian");
    return {
         (a[4] * a[8] - a[5] * a[7]) / d,
        -(a[1] * a[8] - a[2] * a[7]) / d,
         (a[1] * a[5] - a[2] * a[4]) / d,
        -(a[3] * a[8] - a[5] * a[6]) / d,
         (a[0] * a[8] - a[2] * a[6]) / d,
        -(a[0] * a[5] - a[2] * a[3]) / d,
         (a[3] * a[7] - a[4] * a[6]) / d,
        -(a[0] * a[7] - a[1] * a[6]) / d,
         (a[0] * a[4] - a[1] * a[3]) / d};
}

void add_btdb(std::vector<double>& ke,
              const std::vector<double>& B,
              const std::vector<double>& D,
              int strain_dim,
              int ndof,
              double weight) {
    for (int i = 0; i < ndof; ++i) {
        for (int j = 0; j < ndof; ++j) {
            double v = 0.0;
            for (int a = 0; a < strain_dim; ++a) {
                double db = 0.0;
                for (int b = 0; b < strain_dim; ++b) db += D[a * strain_dim + b] * B[b * ndof + j];
                v += B[a * ndof + i] * db;
            }
            ke[static_cast<std::size_t>(i * ndof + j)] += v * weight;
        }
    }
}

ElementMatrix compute_q4(const FEM& fem, int elem_id, double nu, double thickness) {
    ElementMatrix out;
    out.ndof = 8;
    out.ke.assign(64, 0.0);
    out.me.assign(64, 0.0);

    const auto& elem = fem.Element.at(static_cast<std::size_t>(elem_id));
    const double c = 1.0 / (1.0 - nu * nu);
    const std::vector<double> D = {
        c, c * nu, 0.0,
        c * nu, c, 0.0,
        0.0, 0.0, c * (1.0 - nu) / 2.0};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};

    for (double xi : gp) {
        for (double eta : gp) {
            const double N[4] = {
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta)};
            const double dNdxi[8] = {
                -0.25 * (1.0 - eta), -0.25 * (1.0 - xi),
                 0.25 * (1.0 - eta), -0.25 * (1.0 + xi),
                 0.25 * (1.0 + eta),  0.25 * (1.0 + xi),
                -0.25 * (1.0 + eta),  0.25 * (1.0 - xi)};

            double J[4] = {0.0, 0.0, 0.0, 0.0};
            for (int a = 0; a < 4; ++a) {
                const auto& p = fem.Node.at(static_cast<std::size_t>(elem[a]));
                J[0] += p[0] * dNdxi[2 * a + 0];
                J[1] += p[0] * dNdxi[2 * a + 1];
                J[2] += p[1] * dNdxi[2 * a + 0];
                J[3] += p[1] * dNdxi[2 * a + 1];
            }
            const double detJ = J[0] * J[3] - J[1] * J[2];
            if (std::abs(detJ) < 1e-30) throw std::runtime_error("Singular Q4 Jacobian");
            const double invJ[4] = {J[3] / detJ, -J[1] / detJ, -J[2] / detJ, J[0] / detJ};

            std::vector<double> B(3 * 8, 0.0);
            for (int a = 0; a < 4; ++a) {
                const double dx = invJ[0] * dNdxi[2 * a + 0] + invJ[1] * dNdxi[2 * a + 1];
                const double dy = invJ[2] * dNdxi[2 * a + 0] + invJ[3] * dNdxi[2 * a + 1];
                B[0 * 8 + 2 * a + 0] = dx;
                B[1 * 8 + 2 * a + 1] = dy;
                B[2 * 8 + 2 * a + 0] = dy;
                B[2 * 8 + 2 * a + 1] = dx;
            }
            add_btdb(out.ke, B, D, 3, 8, std::abs(detJ) * thickness);

            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    const double v = N[a] * N[b] * std::abs(detJ) * thickness;
                    out.me[static_cast<std::size_t>((2 * a + 0) * 8 + (2 * b + 0))] += v;
                    out.me[static_cast<std::size_t>((2 * a + 1) * 8 + (2 * b + 1))] += v;
                }
            }
        }
    }
    return out;
}

ElementMatrix compute_h8(const FEM& fem, int elem_id, double nu) {
    ElementMatrix out;
    out.ndof = 24;
    out.ke.assign(24 * 24, 0.0);
    out.me.assign(24 * 24, 0.0);

    const auto& elem = fem.Element.at(static_cast<std::size_t>(elem_id));
    const double c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu));
    std::vector<double> D(6 * 6, 0.0);
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

    constexpr int s[8][3] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}};
    constexpr double gp[2] = {-0.57735026918962576451, 0.57735026918962576451};

    for (double xi : gp) {
        for (double eta : gp) {
            for (double zeta : gp) {
                std::array<double, 8> N{};
                std::array<double, 24> dNdxi{};
                for (int i = 0; i < 8; ++i) {
                    const double sx = s[i][0];
                    const double sy = s[i][1];
                    const double sz = s[i][2];
                    N[i] = 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta);
                    dNdxi[i * 3 + 0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);
                    dNdxi[i * 3 + 1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta);
                    dNdxi[i * 3 + 2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta);
                }
                Mat3 J{};
                for (int a = 0; a < 8; ++a) {
                    const auto& p = fem.Node.at(static_cast<std::size_t>(elem[a]));
                    for (int alpha = 0; alpha < 3; ++alpha) {
                        J[0 * 3 + alpha] += p[0] * dNdxi[a * 3 + alpha];
                        J[1 * 3 + alpha] += p[1] * dNdxi[a * 3 + alpha];
                        J[2 * 3 + alpha] += p[2] * dNdxi[a * 3 + alpha];
                    }
                }
                const double detJ = std::abs(det3(J));
                const Mat3 iJ = inv3(J);
                std::vector<double> B(6 * 24, 0.0);
                for (int a = 0; a < 8; ++a) {
                    double d[3]{};
                    for (int beta = 0; beta < 3; ++beta) {
                        for (int alpha = 0; alpha < 3; ++alpha) d[beta] += dNdxi[a * 3 + alpha] * iJ[alpha * 3 + beta];
                    }
                    const int col = 3 * a;
                    B[0 * 24 + col + 0] = d[0];
                    B[1 * 24 + col + 1] = d[1];
                    B[2 * 24 + col + 2] = d[2];
                    B[3 * 24 + col + 0] = d[1];
                    B[3 * 24 + col + 1] = d[0];
                    B[4 * 24 + col + 1] = d[2];
                    B[4 * 24 + col + 2] = d[1];
                    B[5 * 24 + col + 0] = d[2];
                    B[5 * 24 + col + 2] = d[0];
                }
                add_btdb(out.ke, B, D, 6, 24, detJ);
                for (int a = 0; a < 8; ++a) {
                    for (int b = 0; b < 8; ++b) {
                        const double v = N[a] * N[b] * detJ;
                        for (int d = 0; d < 3; ++d) out.me[static_cast<std::size_t>((3 * a + d) * 24 + (3 * b + d))] += v;
                    }
                }
            }
        }
    }
    return out;
}

std::pair<double, double> interp_region(const Region& region, double x) {
    const double E = region.interp.elastic_interp(region.mat_paras.E_inf, x);
    const double rho = region.interp.mass_density_interp(region.mat_paras.rho, x);
    return {E, rho};
}

} // namespace

ElementMatrix compute_element_matrices(const FEM& fem, int elem_id, double nu, double thickness) {
    if (fem.mesh_type == "2D_Q4") return compute_q4(fem, elem_id, nu, thickness);
    if (fem.mesh_type == "3D_H8") return compute_h8(fem, elem_id, nu);
    throw std::runtime_error("Unsupported element type");
}

void assemble_K_M(const FEM& fem,
                  const Analysis& analysis,
                  const std::vector<std::vector<double>>& y_elem,
                  Mat K,
                  Mat M) {
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    const int elem_start = (fem.NElem * rank) / size;
    const int elem_end = (fem.NElem * (rank + 1)) / size;

    for (const Region& region : analysis.regions) {
        const double region_nu = region.mat_paras.nu.empty() ? fem.nu : region.mat_paras.nu.back();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            if (region.elem_ids.empty()) throw std::runtime_error("assemble_K_M: region has no elements");
            uniform_em = compute_element_matrices(fem, region.elem_ids.front(), region_nu, fem.Thickness);
        }

        for (int elem_id : region.elem_ids) {
            if (elem_id < elem_start || elem_id >= elem_end) continue;
            const double x = y_elem.at(static_cast<std::size_t>(elem_id)).front();
            const auto [Eeff, rhoeff] = interp_region(region, x);
            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                local_em = compute_element_matrices(fem, elem_id, region_nu, fem.Thickness);
                em_ptr = &local_em;
            }
            const ElementMatrix& em = *em_ptr;
            const auto& rows = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));
            std::vector<PetscScalar> ke(em.ke.size());
            std::vector<PetscScalar> me(em.me.size());
            for (std::size_t i = 0; i < em.ke.size(); ++i) {
                ke[i] = Eeff * em.ke[i];
                me[i] = rhoeff * em.me[i];
            }
            MatSetValues(K, em.ndof, rows.data(), em.ndof, rows.data(), ke.data(), ADD_VALUES);
            MatSetValues(M, em.ndof, rows.data(), em.ndof, rows.data(), me.data(), ADD_VALUES);
        }
    }
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

} // namespace stage2
