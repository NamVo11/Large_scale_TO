#pragma once

#include "myjson.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct MaterialParas {
    int id = 0;
    std::string type = "linear_elastic";
    int nMat = 0;
    std::vector<int> mat_ids;
    std::vector<int> val_ids;

    MaterialParas() = default;

    MaterialParas(const myjson::json&,
             const std::vector<int>& region_mat_ids,
             const std::string& region_status) {
        mat_ids = myjson::lower_ascii(region_status) == "void" ? std::vector<int>{1} : region_mat_ids;
        if (mat_ids.empty()) throw std::runtime_error("Material: region mat_ids is empty");
        nMat = static_cast<int>(mat_ids.size());
    }

    virtual ~MaterialParas() = default;

    static std::shared_ptr<MaterialParas> InitializeMaterialParas(const myjson::json& materials_json, 
                                                                  const std::vector<int>& region_mat_ids,
                                                                  const std::string& region_status);

protected:
    static const myjson::json& material_json_by_id(const myjson::json& materials_json, int id) {
        for (const auto& item : materials_json) {
            if (item.at("id").get<int>() == id) return item;
        }
        throw std::runtime_error("Material id not found: " + std::to_string(id));
    }

};
