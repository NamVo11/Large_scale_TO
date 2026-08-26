#include "Region/Region.hpp"

#include "FEM/FEM.hpp"
#include "myjson.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace {

bool in_range(double x, const std::array<double, 2>& range) {
    return x >= range[0] && x <= range[1];
}

const HMSet& find_hmset_by_name(const FEM& fem, const std::string& name) {
    const auto* unstructured = dynamic_cast<const UnstructuredMesh*>(&fem);
    if (!unstructured) {
        throw std::runtime_error("Region source=hmset requires UnstructuredMesh: " + name);
    }
    return unstructured->hmset_by_name(name);
}

} // namespace

void Region::finalize_assignment(const FEM& fem, std::vector<int>& assigned) {
    std::sort(elem_ids.begin(), elem_ids.end());
    elem_ids.erase(std::unique(elem_ids.begin(), elem_ids.end()), elem_ids.end());
    if (elem_ids.empty()) throw std::runtime_error("Region selects no elements: " + name);

    for (int e : elem_ids) {
        if (e < 0 || e >= fem.NElem) throw std::runtime_error("Region has invalid element local id: " + name);
        if (assigned[static_cast<std::size_t>(e)] != 0) {
            throw std::runtime_error("Overlapping regions around element " + std::to_string(e) + ": " + name);
        }
        assigned[static_cast<std::size_t>(e)] = id == 0 ? 1 : id;
    }
    nElem = static_cast<int>(elem_ids.size());
}

bool RangeRegion::contains_element_center(const FEM& fem, int elem_id) const {
    if (x_range.empty() || y_range.empty()) {
        throw std::runtime_error("Region box needs x/y ranges: " + name);
    }
    if (x_range.size() != y_range.size()) {
        throw std::runtime_error("Region x/y range size mismatch: " + name);
    }
    if (fem.dim == 3 && z_range.size() != x_range.size()) {
        throw std::runtime_error("Region z range size mismatch: " + name);
    }

    const auto& center = fem.elem_center[static_cast<std::size_t>(elem_id)];
    for (std::size_t r = 0; r < x_range.size(); ++r) {
        bool inside = in_range(center[0], x_range[r]) && in_range(center[1], y_range[r]);
        if (fem.dim == 3) inside = inside && in_range(center[2], z_range[r]);
        if (inside) return true;
    }
    return false;
}

void RangeRegion::assign_elements(const FEM& fem, const std::vector<int>& assigned) {
    elem_ids.clear();
    const std::string type = myjson::lower_ascii(range_type.empty() ? "all" : range_type);
    if (type == "all") {
        elem_ids.resize(static_cast<std::size_t>(fem.NElem));
        std::iota(elem_ids.begin(), elem_ids.end(), 0);
    } else if (type == "box") {
        for (int e = 0; e < fem.NElem; ++e) {
            if (contains_element_center(fem, e)) elem_ids.push_back(e);
        }
    } else if (type == "other") {
        for (int e = 0; e < fem.NElem; ++e) {
            if (assigned[static_cast<std::size_t>(e)] == 0) elem_ids.push_back(e);
        }
    } else {
        throw std::runtime_error("Unknown Region.range_type: " + type);
    }
}

void HMSetRegion::assign_elements(const FEM& fem) {
    const std::string set_name = !hmset_name.empty() ? hmset_name : status;
    const HMSet& set = find_hmset_by_name(fem, set_name);
    if (set.kind != 2) throw std::runtime_error("HMSET is not an element set: " + set_name);
    elem_ids = set.local_ids;
}

std::vector<Region> InitializeRegions(const myjson::json& regions_json, const FEM& fem) {
    if (!regions_json.is_array() || regions_json.empty()) throw std::runtime_error("JSON Regions is empty");

    std::vector<Region> regions;
    std::vector<int> assigned(static_cast<std::size_t>(fem.NElem), 0);
    for (const auto& item : regions_json) {
        const std::string source = myjson::lower_ascii(myjson::string_value(item, "source", "range"));
        if (source == "hmset") {
            HMSetRegion region(fem, item);
            region.finalize_assignment(fem, assigned);
            regions.push_back(static_cast<Region&&>(std::move(region)));
        } else if (source == "range") {
            RangeRegion region(fem, item, assigned);
            region.finalize_assignment(fem, assigned);
            regions.push_back(static_cast<Region&&>(std::move(region)));
        } else {
            throw std::runtime_error("Unknown Region.source: " + source);
        }
    }

    if (std::any_of(assigned.begin(), assigned.end(), [](int v) { return v == 0; })) {
        throw std::runtime_error("Some elements are not assigned to any region");
    }
    return regions;
}
