#include "MatlabLikeData.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool in_range(double x, const std::array<double, 2>& r) {
    return x >= r[0] && x <= r[1];
}

std::vector<int> ids_by_id(const std::vector<BoundaryCondition>& all, const std::vector<int>& ids) {
    std::vector<int> out;
    for (int id : ids) {
        auto it = std::find_if(all.begin(), all.end(), [id](const BoundaryCondition& b) {
            return b.id == id;
        });
        if (it == all.end()) throw std::runtime_error("Boundary condition id not found");
        out.push_back(static_cast<int>(std::distance(all.begin(), it)));
    }
    return out;
}

std::vector<int> active_components(const BoundaryCondition& bc, int dim) {
    std::vector<int> comps;
    for (int d = 0; d < dim; ++d) {
        if (bc.flag[static_cast<std::size_t>(d)] != 0) comps.push_back(d);
    }
    if (comps.empty()) throw std::runtime_error("BC flag has no active component");
    return comps;
}

void ensure_bc_ranges(const BoundaryCondition& bc, int dim) {
    if (bc.x_range.empty() || bc.y_range.empty()) throw std::runtime_error("BC x/y ranges are required");
    if (bc.x_range.size() != bc.y_range.size()) throw std::runtime_error("BC x/y range size mismatch");
    if (dim == 3 && bc.z_range.size() != bc.x_range.size()) throw std::runtime_error("BC z range size mismatch");
}

} // namespace

double RAMP(const std::vector<double>& para, double x, double p) {
    if (para.size() < 2) throw std::runtime_error("RAMP needs at least two parameters");
    const double denom = 1.0 + p * (1.0 - x);
    const double g = x / denom;
    return para[0] + (para[1] - para[0]) * g;
}

FEM Mesh(const Geometry& geom, FEM fem) {
    const std::string type = lower(fem.type);
    fem.Node.clear();
    fem.Element.clear();
    fem.ElementDofs.clear();
    fem.elem_center.clear();
    fem.nelx = geom.nelx;
    fem.nely = geom.nely;
    fem.nelz = geom.nelz;

    if (type == "plane_stress" || type == "plane_strain") {
        fem.dim = 2;
        fem.mesh_type = "2D_Q4";
        const double ex = geom.Lx / static_cast<double>(geom.nelx);
        const double ey = geom.Ly / static_cast<double>(geom.nely);
        const int nx = geom.nelx + 1;
        const int ny = geom.nely + 1;
        auto nid = [ny](int i, int j) { return i * ny + j; };

        fem.Node.reserve(static_cast<std::size_t>(nx * ny));
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                fem.Node.push_back({i * ex, j * ey, 0.0});
            }
        }

        fem.Element.reserve(static_cast<std::size_t>(geom.nelx * geom.nely));
        fem.elem_center.reserve(static_cast<std::size_t>(geom.nelx * geom.nely));
        for (int i = 0; i < geom.nelx; ++i) {
            for (int j = 0; j < geom.nely; ++j) {
                std::vector<int> elem{nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)};
                fem.Element.push_back(elem);
                fem.elem_center.push_back({(i + 0.5) * ex, (j + 0.5) * ey, 0.0});
            }
        }
    } else if (type == "3d_brick") {
        fem.dim = 3;
        fem.mesh_type = "3D_H8";
        const double ex = geom.Lx / static_cast<double>(geom.nelx);
        const double ey = geom.Ly / static_cast<double>(geom.nely);
        const double ez = geom.Lz / static_cast<double>(geom.nelz);
        const int nx = geom.nelx + 1;
        const int ny = geom.nely + 1;
        const int nz = geom.nelz + 1;
        auto nid = [nx, ny](int i, int j, int k) { return k * nx * ny + i * ny + j; };

        fem.Node.reserve(static_cast<std::size_t>(nx * ny * nz));
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    fem.Node.push_back({i * ex, j * ey, k * ez});
                }
            }
        }

        fem.Element.reserve(static_cast<std::size_t>(geom.nelx * geom.nely * geom.nelz));
        fem.elem_center.reserve(static_cast<std::size_t>(geom.nelx * geom.nely * geom.nelz));
        for (int k = 0; k < geom.nelz; ++k) {
            for (int i = 0; i < geom.nelx; ++i) {
                for (int j = 0; j < geom.nely; ++j) {
                    fem.Element.push_back({
                        nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                        nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)});
                    fem.elem_center.push_back({(i + 0.5) * ex, (j + 0.5) * ey, (k + 0.5) * ez});
                }
            }
        }
    } else {
        throw std::runtime_error("Mesh: unsupported fem.type");
    }

    fem.NNode = static_cast<int>(fem.Node.size());
    fem.NElem = static_cast<int>(fem.Element.size());
    fem.ndof = fem.NNode * fem.dim;
    fem.AllDofs.resize(static_cast<std::size_t>(fem.ndof));
    std::iota(fem.AllDofs.begin(), fem.AllDofs.end(), 0);
    return fem;
}

void PrecomputeFEM(FEM& fem) {
    if (fem.mesh_type != "2D_Q4" && fem.mesh_type != "3D_H8") {
        throw std::runtime_error("PrecomputeFEM: unsupported mesh type");
    }

    fem.ElementDofs.resize(static_cast<std::size_t>(fem.NElem));
    for (int e = 0; e < fem.NElem; ++e) {
        const auto& elem = fem.Element.at(static_cast<std::size_t>(e));
        auto& dofs = fem.ElementDofs[static_cast<std::size_t>(e)];
        dofs.resize(elem.size() * static_cast<std::size_t>(fem.dim));
        for (std::size_t a = 0; a < elem.size(); ++a) {
            for (int d = 0; d < fem.dim; ++d) {
                dofs[a * static_cast<std::size_t>(fem.dim) + static_cast<std::size_t>(d)] =
                    static_cast<PetscInt>(fem.dim * elem[a] + d);
            }
        }
    }
}

void assign_elements_to_regions(std::vector<Region>& regions, const FEM& fem) {
    std::vector<int> assigned(static_cast<std::size_t>(fem.NElem), 0);
    for (auto& region : regions) {
        region.elem_ids.clear();
        const std::string rt = lower(region.range_type);
        if (rt == "all") {
            region.elem_ids.resize(static_cast<std::size_t>(fem.NElem));
            std::iota(region.elem_ids.begin(), region.elem_ids.end(), 0);
        } else if (rt == "box") {
            if (region.x_range.empty() || region.y_range.empty()) throw std::runtime_error("Region box needs x/y ranges");
            for (int e = 0; e < fem.NElem; ++e) {
                const auto c = fem.elem_center[static_cast<std::size_t>(e)];
                bool inside = false;
                for (std::size_t r = 0; r < region.x_range.size(); ++r) {
                    bool ok = in_range(c[0], region.x_range[r]) && in_range(c[1], region.y_range[r]);
                    if (fem.dim == 3) ok = ok && in_range(c[2], region.z_range[r]);
                    inside = inside || ok;
                }
                if (inside) region.elem_ids.push_back(e);
            }
        } else if (rt == "other") {
            for (int e = 0; e < fem.NElem; ++e) {
                if (assigned[static_cast<std::size_t>(e)] == 0) region.elem_ids.push_back(e);
            }
        } else {
            throw std::runtime_error("Unknown region range_type");
        }

        if (region.elem_ids.empty()) throw std::runtime_error("Region selects no elements");
        for (int e : region.elem_ids) {
            if (assigned[static_cast<std::size_t>(e)] != 0) throw std::runtime_error("Overlapping regions");
            assigned[static_cast<std::size_t>(e)] = 1;
        }
        region.nElem = static_cast<int>(region.elem_ids.size());
    }

    if (std::any_of(assigned.begin(), assigned.end(), [](int v) { return v == 0; })) {
        throw std::runtime_error("Some elements are not assigned to any region");
    }
}

void assign_nodes_and_dofs_to_bcs(std::vector<BoundaryCondition>& BCs, const FEM& fem) {
    for (auto& bc : BCs) {
        ensure_bc_ranges(bc, fem.dim);
        const auto comps = active_components(bc, fem.dim);
        bc.node_ids.clear();
        bc.dof_ids.clear();
        for (int n = 0; n < fem.NNode; ++n) {
            const auto p = fem.Node[static_cast<std::size_t>(n)];
            bool selected = false;
            for (std::size_t r = 0; r < bc.x_range.size(); ++r) {
                bool ok = in_range(p[0], bc.x_range[r]) && in_range(p[1], bc.y_range[r]);
                if (fem.dim == 3) ok = ok && in_range(p[2], bc.z_range[r]);
                selected = selected || ok;
            }
            if (!selected) continue;
            bc.node_ids.push_back(n);
            for (int c : comps) bc.dof_ids.push_back(fem.dim * n + c);
        }
        if (bc.node_ids.empty() || bc.dof_ids.empty()) throw std::runtime_error("BC selects no nodes or dofs");
        std::sort(bc.node_ids.begin(), bc.node_ids.end());
        std::sort(bc.dof_ids.begin(), bc.dof_ids.end());
        bc.node_ids.erase(std::unique(bc.node_ids.begin(), bc.node_ids.end()), bc.node_ids.end());
        bc.dof_ids.erase(std::unique(bc.dof_ids.begin(), bc.dof_ids.end()), bc.dof_ids.end());
        bc.nNode = static_cast<int>(bc.node_ids.size());
        bc.nDof = static_cast<int>(bc.dof_ids.size());
    }
}

void init_mat(const std::vector<Material>& Mat, Analysis& A) {
    if (lower(A.mat_type) != "elastic") throw std::runtime_error("Only elastic init_mat is implemented in stage2");
    for (auto& region : A.regions) {
        std::vector<int> mat_ids = region.mat_ids;
        if (lower(region.status) == "void") mat_ids = {1};
        region.mat_paras.nMat = static_cast<int>(mat_ids.size());
        region.mat_paras.mat_ids = mat_ids;
        region.mat_paras.val_ids.clear();
        region.mat_paras.E_inf.clear();
        region.mat_paras.nu.clear();
        region.mat_paras.rho.clear();
        for (int id : mat_ids) {
            auto it = std::find_if(Mat.begin(), Mat.end(), [id](const Material& m) { return m.id == id; });
            if (it == Mat.end()) throw std::runtime_error("Material id not found");
            region.mat_paras.E_inf.push_back(it->E_inf != 0.0 ? it->E_inf : it->E);
            region.mat_paras.nu.push_back(it->nu);
            region.mat_paras.rho.push_back(it->rho);
        }
        if (lower(region.status) == "design_val") {
            for (std::size_t i = 1; i < mat_ids.size(); ++i) region.mat_paras.val_ids.push_back(mat_ids[i] - 1);
        } else {
            region.mat_paras.val_ids.push_back(std::max(0, mat_ids.front() - 1));
        }
    }
}

void init_interp(const Interp& interp, Analysis& A) {
    for (auto& region : A.regions) {
        if (lower(region.status) == "design_val") {
            region.interp = interp;
        } else {
            region.interp.elastic_interp = [](const std::vector<double>& para, double) { return para.front(); };
            region.interp.mass_density_interp = [](const std::vector<double>& para, double) { return para.front(); };
        }
    }
}

void attach_bcs_to_analysis_by_id(Analysis& A,
                                  const std::vector<BoundaryCondition>& Dirich,
                                  const std::vector<BoundaryCondition>& Neum) {
    A.Dirich.clear();
    A.Neum.clear();
    for (int idx : ids_by_id(Dirich, A.Dirich_id)) {
        BoundaryCondition bc = Dirich[static_cast<std::size_t>(idx)];
        bc.fun_value = [bc, &A](int istep) { return bc_value(bc, A, istep); };
        A.Dirich.push_back(std::move(bc));
    }
    for (int idx : ids_by_id(Neum, A.Neum_id)) {
        BoundaryCondition bc = Neum[static_cast<std::size_t>(idx)];
        bc.fun_value = [bc, &A](int istep) { return bc_value(bc, A, istep); };
        A.Neum.push_back(std::move(bc));
    }
}

Eigen::MatrixXd init_design_variables(const FEM& fem,
                                      const std::vector<Region>&,
                                      const std::vector<Material>& Mat,
                                      double x0,
                                      std::vector<int>& x_label) {
    const int nvar = std::max(1, static_cast<int>(Mat.size()) - 1);
    Eigen::MatrixXd x = Eigen::MatrixXd::Constant(fem.NElem, nvar, x0);
    x_label.assign(static_cast<std::size_t>(fem.NElem), 0);
    return x;
}

double bc_value(const BoundaryCondition& BC, const Analysis& A, int istep) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const std::string tt = lower(BC.time_type);
    if (tt == "constant") return BC.amp;
    if (tt == "sin") {
        const double t = istep * A.dt;
        return BC.amp * std::sin(2.0 * pi * BC.nCycle * t / A.Tmax + BC.phase);
    }
    if (tt == "linear") {
        return BC.amp * static_cast<double>(istep) / static_cast<double>(A.Nstep);
    }
    if (tt == "sum_sim") {
        const double t = (istep - 1) * A.dt;
        const double freq[] = {90.0, 63.0, 112.5, 135.0, 153.0, 18.0, 117.0};
        const double scale[] = {0.5, 0.8, 0.2, 0.70, 0.55, 0.5, 0.5};
        const double phi[] = {0.0, 1.3, 2.1, 4.0, 5.2, 0.0, 2.0};
        const double rise_time = 0.001;
        const double envelope = std::min(t / rise_time, 1.0);
        double sum = 0.0;
        for (int i = 0; i < 7; ++i) sum += scale[i] * std::sin(2.0 * pi * freq[i] * t + phi[i]);
        return BC.amp * envelope * sum;
    }
    throw std::runtime_error("Unsupported BC time_type");
}
