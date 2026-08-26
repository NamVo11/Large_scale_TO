#pragma once

#include "BounadryCondition/Function.hpp"
#include "FEM/FEM.hpp"
#include "myjson.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Analysis;

struct BoundaryCondition {
    int id = 0;
    std::string name;
    std::string source = "range";
    std::string kind;
    std::array<int, 3> flag{0, 0, 0};

    std::vector<int> node_ids;
    std::vector<int> dof_ids;
    int nNode = 0;
    int nDof = 0;
    std::shared_ptr<Function> function;

    BoundaryCondition() = default;

    explicit BoundaryCondition(const myjson::json& j) {
        id = myjson::int_value(j, "id", id);
        name = myjson::string_value(j, "name", name);
        source = myjson::string_value(j, "source", source);
        kind = myjson::string_value(j, "kind", kind);
        if (j.contains("flag")) j.at("flag").get_to(flag);
        function = InitializeFunction(j.contains("Function") ? j.at("Function") : j);
    }

    double value(const Analysis& analysis, int istep, double scale) const;
};

struct RangeBoundaryCondition : BoundaryCondition {
    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;

    RangeBoundaryCondition(const FEM& fem, const myjson::json& j)
        : BoundaryCondition(j) {
        if (j.contains("x_range")) j.at("x_range").get_to(x_range);
        if (j.contains("y_range")) j.at("y_range").get_to(y_range);
        if (j.contains("z_range")) j.at("z_range").get_to(z_range);
        assign_nodes_and_dofs_to_bcs(fem);
    }

    void assign_nodes_and_dofs_to_bcs(const FEM& fem);
};

struct HMSetBoundaryCondition : BoundaryCondition {
    std::string hmset_name;

    HMSetBoundaryCondition(const FEM& fem, const myjson::json& j)
        : BoundaryCondition(j) {
        hmset_name = myjson::string_value(j, "hmset_name", hmset_name);
        assign_nodes_and_dofs_to_bcs(fem);
    }

    void assign_nodes_and_dofs_to_bcs(const FEM& fem);
};

inline std::vector<std::shared_ptr<BoundaryCondition>>
InitializeBoundaryConditions(const myjson::json& bcs_json, const FEM& fem) {
    std::vector<std::shared_ptr<BoundaryCondition>> bcs;
    for (const auto& item : bcs_json) {
        const std::string source = myjson::lower_ascii(myjson::string_value(item, "source", "range"));
        if (source == "hmset") {
            bcs.push_back(std::make_shared<HMSetBoundaryCondition>(fem, item));
        } else if (source == "range") {
            bcs.push_back(std::make_shared<RangeBoundaryCondition>(fem, item));
        } else {
            throw std::runtime_error("Unknown BoundaryCondition.source: " + source);
        }
    }
    return bcs;
}
