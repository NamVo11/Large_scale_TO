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
    int id;
    std::string name;
    std::string source;
    std::string kind;
    std::array<int, 3> flag;

    std::vector<int> node_ids;
    std::vector<int> dof_ids;
    int nNode = 0;
    int nDof = 0;
    std::shared_ptr<Function> function;

    BoundaryCondition() = delete;

    explicit BoundaryCondition(const myjson::json& j) {
        id = j.at("id").get<int>();
        name = myjson::string_value(j, "name", name);
        source = j.at("source").get<std::string>();
        kind = j.at("kind").get<std::string>();
        j.at("flag").get_to(flag);
        function = InitializeFunction(j.contains("Function") ? j.at("Function") : j);
    }

    double value(const Analysis& analysis, int istep, double scale) const;
};

struct RangeBoundaryCondition : BoundaryCondition {

    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;

    RangeBoundaryCondition(const FEM& fem, const myjson::json& j) : BoundaryCondition(j) {

        j.at("x_range").get_to(x_range);
        j.at("y_range").get_to(y_range);
        if (fem.dim == 3) j.at("z_range").get_to(z_range);
        else if (j.contains("z_range")) j.at("z_range").get_to(z_range);
        
        assign_nodes_and_dofs_to_bcs(fem);
    }

    void assign_nodes_and_dofs_to_bcs(const FEM& fem);
};

struct HMSetBoundaryCondition : BoundaryCondition {

    std::string hmset_name;

    HMSetBoundaryCondition(const FEM& fem, const myjson::json& j) : BoundaryCondition(j) {
        
        hmset_name = j.at("hmset_name").get<std::string>();
        assign_nodes_and_dofs_to_bcs(fem);
    }

    void assign_nodes_and_dofs_to_bcs(const FEM& fem);
};

inline std::vector<std::shared_ptr<BoundaryCondition>>
InitializeBoundaryConditions(const myjson::json& bcs_json, const FEM& fem) {
    std::vector<std::shared_ptr<BoundaryCondition>> bcs;
    for (const auto& item : bcs_json) {
        const std::string source = myjson::lower_ascii(item.at("source").get<std::string>());
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
