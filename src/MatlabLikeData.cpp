#include "MatlabLikeData.hpp"

#include <algorithm>
#include <cctype>
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

int dof_id_from_node_component(const FEM& fem, int node_local, int component) {
    return fem.dim * node_local + component;
}

const HMSet& find_hmset_by_name(const FEM& fem, const std::string& name) {
    auto it = fem.hm_sets_by_name.find(name);
    if (it != fem.hm_sets_by_name.end()) return it->second;
    const std::string lname = lower(name);
    it = fem.hm_sets_by_name.find(lname);
    if (it != fem.hm_sets_by_name.end()) return it->second;
    throw std::runtime_error("HMSET not found: " + name);
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

void RangeRegion::assign_elements_to_regions(std::vector<Region>& regions, const FEM& fem) {
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

void HMSetRegion::assign_elements_to_regions(std::vector<Region>& regions, const FEM& fem) {
    std::vector<int> assigned(static_cast<std::size_t>(fem.NElem), 0);
    for (auto& region : regions) {
        const std::string set_name = !region.hmset_name.empty() ? region.hmset_name : region.status;
        const HMSet& set = find_hmset_by_name(fem, set_name);
        if (set.kind != 2) throw std::runtime_error("HMSET is not an element set: " + set_name);
        region.elem_ids = set.local_ids;
        if (region.elem_ids.empty()) throw std::runtime_error("Region HMSET selects no elements: " + set_name);
        for (int e : region.elem_ids) {
            if (e < 0 || e >= fem.NElem) throw std::runtime_error("Region HMSET has invalid element local id");
            if (assigned[static_cast<std::size_t>(e)] != 0) throw std::runtime_error("Overlapping HMSET regions");
            assigned[static_cast<std::size_t>(e)] = 1;
        }
        region.nElem = static_cast<int>(region.elem_ids.size());
    }
}

void RangeBoundaryCondition::assign_nodes_and_dofs_to_bcs(std::vector<BoundaryCondition>& BCs, const FEM& fem) {
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

void HMSetBoundaryCondition::assign_nodes_and_dofs_to_bcs(std::vector<BoundaryCondition>& BCs, const FEM& fem) {
    for (auto& bc : BCs) {
        const std::string set_name = !bc.hmset_name.empty() ? bc.hmset_name : bc.kind;
        const HMSet& set = find_hmset_by_name(fem, set_name);
        if (set.kind != 1) throw std::runtime_error("HMSET is not a node set: " + set_name);
        const auto comps = active_components(bc, fem.dim);
        bc.node_ids = set.local_ids;
        bc.dof_ids.clear();
        for (int n : bc.node_ids) {
            if (n < 0 || n >= fem.NNode) throw std::runtime_error("BC HMSET has invalid node local id");
            for (int c : comps) bc.dof_ids.push_back(dof_id_from_node_component(fem, n, c));
        }
        if (bc.node_ids.empty() || bc.dof_ids.empty()) throw std::runtime_error("BC HMSET selects no nodes or dofs: " + set_name);
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
                                      const std::vector<Region>& regions,
                                      const std::vector<Material>& Mat,
                                      double x0,
                                      std::vector<int>& x_label) {

    const int nvar = std::max(1, static_cast<int>(Mat.size()) - 1);

    if (x0 < 0.0 || x0 > 1.0) {
        throw std::runtime_error("init_design_variables: x0 must be between 0 and 1");
    }

    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(fem.NElem, nvar);
    x_label.assign(static_cast<std::size_t>(fem.NElem), 0);

    for (const Region& region : regions) {
        for (int val_id : region.mat_paras.val_ids) {
            if (val_id <= 0 || val_id > nvar) {
                throw std::runtime_error("init_design_variables: region val_id is outside x columns");
            }
        }

        for (int elem_id : region.elem_ids) {
            if (elem_id < 0 || elem_id >= fem.NElem) {
                throw std::runtime_error("init_design_variables: invalid element id in region");
            }
        }

        if (region.status == "solid") {
            for (int elem_id : region.elem_ids) {
                x_label[static_cast<std::size_t>(elem_id)] = 1;

                for (int val_id : region.mat_paras.val_ids) {
                    x(elem_id, val_id - 1) = 1.0;
                }
            }
        } else if (region.status == "void") {
            for (int elem_id : region.elem_ids) {
                x_label[static_cast<std::size_t>(elem_id)] = -1;
                x.row(elem_id).setZero();
            }
        } else if (region.status == "design_val") {
            for (int elem_id : region.elem_ids) {
                for (int val_id : region.mat_paras.val_ids) {
                    x(elem_id, val_id - 1) = x0;
                }
            }
        } else {
            throw std::runtime_error("init_design_variables: unsupported region status: " + region.status);
        }
    }
    return x;
}

double bc_value(const BoundaryCondition& BC, const Analysis& A, int istep) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const std::string tt = lower(BC.time_type);
    if (tt == "constant") return BC.amp;
    if (tt == "sin") {
        const double t = istep * A.dt;
        return  (BC.base_value + BC.amp* std::sin(2.0 * pi * BC.nCycle * t / A.Tmax + BC.phase))/BC.nDof;
    }
    if (tt == "linear") {
        return (BC.amp/BC.nDof) * static_cast<double>(istep) / static_cast<double>(A.Nstep);
    }
    if (tt == "sum_sin" || tt == "sum_sim") 
    {
    // old implementation of sum_sin, commented out for reference
    const double t = istep * A.dt;
    constexpr int N = 7;
    const double freq_factor = 0.5;// previously was 0.9
    const double freq[N] = {100.0, 70.0, 125.0, 150.0, 170.0, 120.0, 130.0};
    const double scale[N] = {0.5, 0.8, 0.2, 0.70, 0.55, 0.5, 0.5};
    const double phi[N] = {0.0, 1.3, 2.1, 4.0, 5.2, 0.0, 2.0};
    const double rise_time = 0.001;
    const double envelope = std::min(t / rise_time, 1.0);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        const double f = freq_factor * freq[i];
        sum += scale[i] * std::sin(2.0 * pi * f * t + phi[i]);
    }
    return (BC.base_value + BC.amp*envelope* sum)/BC.nDof;

    }
    // =====================================================================
    if (tt == "sum_sin1" || tt == "sum_sim1") 
    {
    const double t = istep * A.dt;
    constexpr int N = 6;
    const double freq_factor = 0.35;// previously was 0.9
    const double freq[N] = {25.0, 55.0, 95.0, 150.0, 180.0, 200.0};
    const double scale[N] = {1.00, 0.82, 0.58, 0.44, 0.30, 0.30};
    const double phi[N] = {0.37, 2.11, 4.38, 1.42, 5.83, 2.84};
    const double rise_time = 0.001;
    const double envelope = std::min(t / rise_time, 1.0);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        const double f = freq_factor * freq[i];
        sum += scale[i] * std::sin(2.0 * pi * f * t + phi[i]);
    }
    return (BC.base_value + BC.amp* sum)/BC.nDof;
    
    }
    // ===========================================================================
    if (tt == "sum_sin2" || tt == "sum_sim2") 
    {
    const double t = istep * A.dt;
    constexpr int N = 6;
    const double freq_factor = 0.35;// previously was 0.9
    const double freq[N] = {25.0, 55.0, 95.0, 150.0, 180.0, 200.0};
    const double scale[N] = {1.00, 0.82, 0.58, 0.44, 0.30, 0.30};
    const double phi[N] = {0.37, 2.11, 4.38, 1.42, 5.83, 2.84};
    const double phi_scale = 0.5;
    const double rise_time = 0.001;
    const double envelope = std::min(t / rise_time, 1.0);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        const double f = freq_factor * freq[i];
        sum += scale[i] * std::sin(2.0 * pi * f * t + phi_scale* phi[i]);
    }
    return (BC.base_value + BC.amp* sum)/BC.nDof;
    
    }
    throw std::runtime_error("Unsupported BC time_type");
}
