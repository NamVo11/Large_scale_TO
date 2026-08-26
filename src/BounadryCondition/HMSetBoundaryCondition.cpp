#include "BounadryCondition/BoundaryCondition.hpp"

#include <algorithm>
#include <stdexcept>

namespace {

int dof_id_from_node_component(const FEM& fem, int node_local, int component) {
    return fem.dim * node_local + component;
}

const HMSet& find_hmset_by_name(const FEM& fem, const std::string& name) {
    const auto* unstructured = dynamic_cast<const UnstructuredMesh*>(&fem);
    if (!unstructured) {
        throw std::runtime_error("BoundaryCondition source=hmset requires UnstructuredMesh: " + name);
    }
    return unstructured->hmset_by_name(name);
}

std::vector<int> active_components(const BoundaryCondition& bc, int dim) {
    std::vector<int> comps;
    for (int d = 0; d < dim; ++d) {
        if (bc.flag[static_cast<std::size_t>(d)] != 0) comps.push_back(d);
    }
    if (comps.empty()) throw std::runtime_error("BC flag has no active component");
    return comps;
}

} // namespace

void HMSetBoundaryCondition::assign_nodes_and_dofs_to_bcs(const FEM& fem) {
    const std::string set_name = !hmset_name.empty() ? hmset_name : kind;
    const HMSet& set = find_hmset_by_name(fem, set_name);
    if (set.kind != 1) throw std::runtime_error("HMSET is not a node set: " + set_name);
    const auto comps = active_components(*this, fem.dim);
    node_ids = set.local_ids;
    dof_ids.clear();
    for (int n : node_ids) {
        if (n < 0 || n >= fem.NNode) throw std::runtime_error("BC HMSET has invalid node local id");
        for (int c : comps) dof_ids.push_back(dof_id_from_node_component(fem, n, c));
    }
    if (node_ids.empty() || dof_ids.empty()) throw std::runtime_error("BC HMSET selects no nodes or dofs: " + set_name);
    std::sort(node_ids.begin(), node_ids.end());
    std::sort(dof_ids.begin(), dof_ids.end());
    node_ids.erase(std::unique(node_ids.begin(), node_ids.end()), node_ids.end());
    dof_ids.erase(std::unique(dof_ids.begin(), dof_ids.end()), dof_ids.end());
    nNode = static_cast<int>(node_ids.size());
    nDof = static_cast<int>(dof_ids.size());
}
