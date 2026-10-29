#pragma once

#include "Material/MaterialParas.hpp"
#include <cmath>

// Input-only placeholder for the future history-dependent constitutive model.
struct ViscoelasticParas final : MaterialParas {
    std::vector<double> E_values;
    std::vector<double> nu_values;
    std::vector<double> rho_values;

    ViscoelasticParas(const myjson::json& materials_json,
                      const std::vector<int>& region_mat_ids,
                      const std::string& region_status)
        : MaterialParas(materials_json, region_mat_ids, region_status) {
        type = myjson::lower_ascii(material_json_by_id(materials_json, mat_ids.front()).at("type").get<std::string>());
        for (int mat_id : mat_ids) {
            const auto& item = material_json_by_id(materials_json, mat_id);
            if (myjson::lower_ascii(item.at("type").get<std::string>()) != type)
                throw std::runtime_error("All materials in one Region must have the same type");
            E_values.push_back(item.contains("E_inf") ? item.at("E_inf").get<double>() : item.at("E").get<double>());
            nu_values.push_back(item.at("nu").get<double>());
            rho_values.push_back(item.at("rho").get<double>());
        }
        if (nu_values.empty()) throw std::runtime_error("ViscoelasticParas: nu_values is empty");
        for (double nu : nu_values)
            if (std::abs(nu - nu_values.front()) > 1e-12)
                throw std::runtime_error("All materials in one Region must have the same nu");
        if (myjson::lower_ascii(region_status) == "design_val") {
            for (std::size_t i = 1; i < mat_ids.size(); ++i) val_ids.push_back(mat_ids[i] - 1);
        } else {
            val_ids.push_back(std::max(0, mat_ids.front() - 1));
        }
    }
};
