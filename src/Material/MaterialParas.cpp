#include "Material/MaterialParas.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"
#include "Material/StructuralMaterial/ViscoElastic/ViscoelasticParas.hpp"

std::shared_ptr<MaterialParas> MaterialParas::InitializeMaterialParas(
    const myjson::json& materials_json, const std::vector<int>& region_mat_ids,
    const std::string& region_status) {
    const std::vector<int> ids = myjson::lower_ascii(region_status) == "void"
        ? std::vector<int>{1} : region_mat_ids;
    if (ids.empty()) throw std::runtime_error("InitializeMaterialParas: region mat_ids is empty");

    const std::string type = myjson::lower_ascii(
        material_json_by_id(materials_json, ids.front()).at("type").get<std::string>());
    if (type == "elastic" || type == "linear_elastic" || type == "st_venant" || type == "stvenant")
        return std::make_shared<ElasticParas>(materials_json, region_mat_ids, region_status);
    if (type == "viscoelastic" || type == "linear_viscoelastic")
        return std::make_shared<ViscoelasticParas>(materials_json, region_mat_ids, region_status);
    throw std::runtime_error("Unknown Material.type: " + type);
}
