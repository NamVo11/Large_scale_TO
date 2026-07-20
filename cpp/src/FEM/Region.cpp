

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
