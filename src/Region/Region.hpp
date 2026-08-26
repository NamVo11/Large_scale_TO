#pragma once

#include "myjson.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

struct FEM;

struct Region {
    int id = 0;
    std::string name;
    std::string source = "range";
    std::vector<int> mat_ids;
    std::string status = "design_val";

    std::vector<int> elem_ids;
    int nElem = 0;

    Region() = default;

    explicit Region(const myjson::json& j) {
        id = myjson::int_value(j, "id", id);
        name = myjson::string_value(j, "name", name);
        source = myjson::string_value(j, "source", source);
        status = myjson::string_value(j, "status", status);
        if (j.contains("mat_ids")) j.at("mat_ids").get_to(mat_ids);
    }

    void finalize_assignment(const FEM& fem, std::vector<int>& assigned);
};

struct RangeRegion : Region {
    std::string range_type = "all";
    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;

    RangeRegion(const FEM& fem, const myjson::json& j, const std::vector<int>& assigned)
        : Region(j) {
        range_type = myjson::string_value(j, "range_type", range_type);
        if (j.contains("x_range")) j.at("x_range").get_to(x_range);
        if (j.contains("y_range")) j.at("y_range").get_to(y_range);
        if (j.contains("z_range")) j.at("z_range").get_to(z_range);
        assign_elements(fem, assigned);
    }

    void assign_elements(const FEM& fem, const std::vector<int>& assigned);
    bool contains_element_center(const FEM& fem, int elem_id) const;
};

struct HMSetRegion : Region {
    std::string hmset_name;

    HMSetRegion(const FEM& fem, const myjson::json& j)
        : Region(j) {
        hmset_name = myjson::string_value(j, "hmset_name", hmset_name);
        assign_elements(fem);
    }

    void assign_elements(const FEM& fem);
};

std::vector<Region> InitializeRegions(const myjson::json& regions_json, const FEM& fem);
