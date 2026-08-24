#include "Analysis/Analysis.hpp"

#include <algorithm>
#include <cctype>
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

const HMSet& find_hmset_by_name(const FEM& fem, const std::string& name) {
    auto it = fem.hm_sets_by_name.find(name);
    if (it != fem.hm_sets_by_name.end()) return it->second;
    const std::string lname = lower(name);
    it = fem.hm_sets_by_name.find(lname);
    if (it != fem.hm_sets_by_name.end()) return it->second;
    throw std::runtime_error("HMSET not found: " + name);
}

std::vector<int> ids_by_id(const std::vector<std::shared_ptr<BoundaryCondition>>& all, const std::vector<int>& ids) {
    std::vector<int> out;
    for (int id : ids) {
        auto it = std::find_if(all.begin(), all.end(), [id](const std::shared_ptr<BoundaryCondition>& b) {
            return b && b->id == id;
        });
        if (it == all.end()) throw std::runtime_error("Boundary condition id not found");
        out.push_back(static_cast<int>(std::distance(all.begin(), it)));
    }
    return out;
}

} // namespace

void Analysis::solver(const FEM&, const Eigen::MatrixXd&, int) {
    throw std::runtime_error("Analysis::solver is not implemented for base Analysis");
}

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

void Analysis::init_mat(const std::vector<Material>& Mat) {
    if (lower(mat_type) != "elastic") throw std::runtime_error("Only elastic init_mat is implemented in stage2");
    for (auto& region : regions) {
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

void Analysis::init_interp(const Interp& interp) {
    for (auto& region : regions) {
        if (lower(region.status) == "design_val") {
            region.interp = interp;
        } else {
            region.interp.elastic_interp = [](const std::vector<double>& para, double) { return para.front(); };
            region.interp.mass_density_interp = [](const std::vector<double>& para, double) { return para.front(); };
        }
    }
}

void Analysis::attach_bcs_by_id(const std::vector<std::shared_ptr<BoundaryCondition>>& Dirich_all,
                                const std::vector<std::shared_ptr<BoundaryCondition>>& Neum_all) {
    Dirich.clear();
    Neum.clear();
    for (int idx : ids_by_id(Dirich_all, Dirich_id)) {
        Dirich.push_back(Dirich_all[static_cast<std::size_t>(idx)]);
    }
    for (int idx : ids_by_id(Neum_all, Neum_id)) {
        Neum.push_back(Neum_all[static_cast<std::size_t>(idx)]);
    }
}
