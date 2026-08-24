#include "BounadryCondition/BoundaryCondition.hpp"

#include <algorithm>
#include <stdexcept>

namespace {

bool in_range(double x, const std::array<double, 2>& r) {
    return x >= r[0] && x <= r[1];
}

std::vector<int> active_components(const BoundaryCondition& bc, int dim) {
    std::vector<int> comps;
    for (int d = 0; d < dim; ++d) {
        if (bc.flag[static_cast<std::size_t>(d)] != 0) comps.push_back(d);
    }
    if (comps.empty()) throw std::runtime_error("BC flag has no active component");
    return comps;
}

void ensure_bc_ranges(const RangeBoundaryCondition& bc, int dim) {
    if (bc.x_range.empty() || bc.y_range.empty()) throw std::runtime_error("BC x/y ranges are required");
    if (bc.x_range.size() != bc.y_range.size()) throw std::runtime_error("BC x/y range size mismatch");
    if (dim == 3 && bc.z_range.size() != bc.x_range.size()) throw std::runtime_error("BC z range size mismatch");
}

} // namespace

void RangeBoundaryCondition::assign_nodes_and_dofs_to_bcs(const FEM& fem) {
    ensure_bc_ranges(*this, fem.dim);
    const auto comps = active_components(*this, fem.dim);
    node_ids.clear();
    dof_ids.clear();
    for (int n = 0; n < fem.NNode; ++n) {
        const auto p = fem.Node[static_cast<std::size_t>(n)];
        bool selected = false;
        for (std::size_t r = 0; r < x_range.size(); ++r) {
            bool ok = in_range(p[0], x_range[r]) && in_range(p[1], y_range[r]);
            if (fem.dim == 3) ok = ok && in_range(p[2], z_range[r]);
            selected = selected || ok;
        }
        if (!selected) continue;
        node_ids.push_back(n);
        for (int c : comps) dof_ids.push_back(fem.dim * n + c);
    }
    if (node_ids.empty() || dof_ids.empty()) throw std::runtime_error("BC selects no nodes or dofs");
    std::sort(node_ids.begin(), node_ids.end());
    std::sort(dof_ids.begin(), dof_ids.end());
    node_ids.erase(std::unique(node_ids.begin(), node_ids.end()), node_ids.end());
    dof_ids.erase(std::unique(dof_ids.begin(), dof_ids.end()), dof_ids.end());
    nNode = static_cast<int>(node_ids.size());
    nDof = static_cast<int>(dof_ids.size());
}
