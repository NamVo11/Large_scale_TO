#pragma once

#include "Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.hpp"
#include "Analysis/StaticLinearElastic/StaticLinearElasticAVM.hpp"
#include "Optimization/Constraint/Constraint.hpp"
#include "Optimization/Objective/Objective.hpp"
#include "Optimization/Volume/VolumeFractionConstraint.hpp"
#include "myjson.hpp"

#include <memory>
#include <stdexcept>
#include <string>

inline std::string group_function_key(const myjson::json& j) {
    const std::string sens_function = myjson::lower_ascii(myjson::string_value(j, "sens_function"));
    if (!sens_function.empty()) return sens_function;
    std::string function_name = myjson::lower_ascii(myjson::string_value(j, "function_name"));
    if (function_name.empty() && j.contains("terms") && j.at("terms").is_array() && !j.at("terms").empty()) {
        function_name = myjson::lower_ascii(myjson::string_value(j.at("terms").front(), "function_name"));
    }
    return function_name;
}

inline std::unique_ptr<GroupFunctionalBase> InitializeGroupFunctional(const myjson::json& j,
                                                                     const std::string& linear_backend) {
    const std::string key = group_function_key(j);
    const std::string backend = myjson::lower_ascii(linear_backend.empty() ? "petsc" : linear_backend);

    if (key == "dynamic_elastic_base" ||
        key == "dynamic_compliance_avm" ||
        key == "dynamic_linear_elastic_avm") {
        if (backend == "eigen") return std::make_unique<EigenDynamicLinearElasticAVM>(j);
        return std::make_unique<PetscDynamicLinearElasticAVM>(j);
    }

    if (key == "static_elastic_base" ||
        key == "static_compliance_avm" ||
        key == "static_linear_elastic_avm") {
        if (backend == "eigen") return std::make_unique<EigenStaticLinearElasticAVM>(j);
        return std::make_unique<PetscStaticLinearElasticAVM>(j);
    }

    if (key == "volume_fraction" || key == "volume_sens_base") {
        return std::make_unique<VolumeFractionConstraint>(j);
    }

    throw std::runtime_error("Unknown group functional: " + key);
}

inline double constraint_limit_from_json(const myjson::json& j) {
    return myjson::double_value(j, "limit", 0.0);
}
