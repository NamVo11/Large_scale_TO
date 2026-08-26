#pragma once

#include "myjson.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace material_detail {

inline const myjson::json& material_json_by_id(const myjson::json& materials_json, int id) {
    for (const auto& item : materials_json) {
        if (item.at("id").get<int>() == id) return item;
    }
    throw std::runtime_error("Material id not found: " + std::to_string(id));
}

inline std::string material_type_by_id(const myjson::json& materials_json, int id) {
    return myjson::lower_ascii(material_json_by_id(materials_json, id).at("type").get<std::string>());
}

} // namespace material_detail

struct Material {
    int id = 0;
    std::string type = "linear_elastic";
    std::string status = "design_val";
    int nMat = 0;
    std::vector<int> mat_ids;
    std::vector<int> val_ids;
    std::vector<double> E_values;
    std::vector<double> nu_values;
    std::vector<double> rho_values;

    Material() = default;

    Material(const myjson::json&,
             const std::vector<int>& region_mat_ids,
             const std::string& region_status) {
        status = region_status;
        mat_ids = myjson::lower_ascii(region_status) == "void" ? std::vector<int>{1} : region_mat_ids;
        if (mat_ids.empty()) throw std::runtime_error("Material: region mat_ids is empty");
        nMat = static_cast<int>(mat_ids.size());
    }

    virtual ~Material() = default;

    bool is_design() const { return myjson::lower_ascii(status) == "design_val"; }
    double nu() const {
        if (nu_values.empty()) throw std::runtime_error("Material: nu_values is empty");
        return nu_values.front();
    }

protected:
    void check_uniform_nu() const {
        if (nu_values.empty()) throw std::runtime_error("Material: nu_values is empty");
        const double first = nu_values.front();
        for (double current : nu_values) {
            if (std::abs(current - first) > 1e-12) {
                throw std::runtime_error("All materials in one Region must have the same nu");
            }
        }
    }
};

struct LinearElasticMaterial : Material {
    LinearElasticMaterial() {
        type = "linear_elastic";
    }

    LinearElasticMaterial(const myjson::json& materials_json,
                          const std::vector<int>& region_mat_ids,
                          const std::string& region_status)
        : Material(materials_json, region_mat_ids, region_status) {
        type = "linear_elastic";
        E_values.clear();
        nu_values.clear();
        rho_values.clear();
        val_ids.clear();

        const std::string expected_type = material_detail::material_type_by_id(materials_json, mat_ids.front());
        for (int mat_id : mat_ids) {
            const myjson::json& item = material_detail::material_json_by_id(materials_json, mat_id);
            const std::string current_type = material_detail::material_type_by_id(materials_json, mat_id);
            if (current_type != expected_type) {
                throw std::runtime_error("All materials in one Region must have the same type");
            }
            E_values.push_back(item.at("E").get<double>());
            nu_values.push_back(item.at("nu").get<double>());
            rho_values.push_back(item.at("rho").get<double>());
        }
        check_uniform_nu();

        if (is_design()) {
            for (std::size_t i = 1; i < mat_ids.size(); ++i) val_ids.push_back(mat_ids[i] - 1);
        } else {
            val_ids.push_back(std::max(0, mat_ids.front() - 1));
        }
    }
};

struct LinearViscoelasticMaterial : Material {
    LinearViscoelasticMaterial() {
        type = "linear_viscoelastic";
    }

    LinearViscoelasticMaterial(const myjson::json& materials_json,
                               const std::vector<int>& region_mat_ids,
                               const std::string& region_status)
        : Material(materials_json, region_mat_ids, region_status) {
        type = "linear_viscoelastic";
        E_values.clear();
        nu_values.clear();
        rho_values.clear();
        val_ids.clear();

        const std::string expected_type = material_detail::material_type_by_id(materials_json, mat_ids.front());
        for (int mat_id : mat_ids) {
            const myjson::json& item = material_detail::material_json_by_id(materials_json, mat_id);
            const std::string current_type = material_detail::material_type_by_id(materials_json, mat_id);
            if (current_type != expected_type) {
                throw std::runtime_error("All materials in one Region must have the same type");
            }
            E_values.push_back(item.contains("E_inf") ? item.at("E_inf").get<double>() : item.at("E").get<double>());
            nu_values.push_back(item.at("nu").get<double>());
            rho_values.push_back(item.at("rho").get<double>());
        }
        check_uniform_nu();

        if (is_design()) {
            for (std::size_t i = 1; i < mat_ids.size(); ++i) val_ids.push_back(mat_ids[i] - 1);
        } else {
            val_ids.push_back(std::max(0, mat_ids.front() - 1));
        }
    }
};

inline std::shared_ptr<Material> InitializeMaterial(const myjson::json& materials_json,
                                                   const std::vector<int>& region_mat_ids,
                                                   const std::string& region_status) {
    const std::vector<int> ids = myjson::lower_ascii(region_status) == "void" ? std::vector<int>{1} : region_mat_ids;
    if (ids.empty()) throw std::runtime_error("InitializeMaterial: region mat_ids is empty");
    const std::string type = material_detail::material_type_by_id(materials_json, ids.front());
    if (type == "elastic" || type == "linear_elastic") {
        return std::make_shared<LinearElasticMaterial>(materials_json, region_mat_ids, region_status);
    }
    if (type == "viscoelastic" || type == "linear_viscoelastic") {
        return std::make_shared<LinearViscoelasticMaterial>(materials_json, region_mat_ids, region_status);
    }
    throw std::runtime_error("Unknown Material.type: " + type);
}
