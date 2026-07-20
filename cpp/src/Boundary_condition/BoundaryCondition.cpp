
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