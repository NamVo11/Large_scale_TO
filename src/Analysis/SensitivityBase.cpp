#include "Analysis/SensitivityBase.hpp"

#include "Analysis/Analysis.hpp"
#include "Analysis/DynamicLinearElastic/DynamicLinearElasticAVM.hpp"
#include "Analysis/StaticLinearElastic/StaticLinearElasticAVM.hpp"
#include "Optimization/Volume/VolumeFractionConstraint.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"

#include <stdexcept>
#include <string>

void GroupFunctionalBase::add_scaled(Eigen::VectorXd& dst,
                                     const Eigen::VectorXd& src,
                                     double scale) const {
    if (src.size() == 0) return;
    if (dst.size() == 0) {
        dst = scale * src;
        return;
    }
    if (dst.size() != src.size()) throw std::runtime_error("GroupFunctionalBase::add_scaled: derivative size mismatch");
    dst.noalias() += scale * src;
}

std::unique_ptr<GroupFunctionalBase> InitializeGroupFunctional(const myjson::json& j,
                                                              const std::string& linear_backend,
                                                              const std::vector<std::shared_ptr<Analysis>>& analysis_all,
                                                              const myjson::json& run_json) {
    const std::string key = myjson::lower_ascii(j.at("sens_function").get<std::string>());
    const std::string backend = myjson::lower_ascii(linear_backend.empty() ? "petsc" : linear_backend);

    std::unique_ptr<GroupFunctionalBase> group;
    if (key == "dynamic_elastic_base" ||
        key == "dynamic_compliance_avm" ||
        key == "dynamic_linear_elastic_avm") {
        if (backend == "eigen") {
            group = std::make_unique<EigenDynamicLinearElasticAVM>(j, analysis_all, run_json);
        } else {
            group = std::make_unique<PetscDynamicLinearElasticAVM>(j, analysis_all, run_json);
        }
    } else if (key == "static_elastic_base" ||
               key == "static_compliance_avm" ||
               key == "static_linear_elastic_avm") {
        if (backend == "eigen") {
            group = std::make_unique<EigenStaticLinearElasticAVM>(j, analysis_all, run_json);
        } else {
            group = std::make_unique<PetscStaticLinearElasticAVM>(j, analysis_all, run_json);
        }
    } else if (key == "volume_fraction" || key == "volume_sens_base") {
        group = std::make_unique<VolumeFractionConstraint>(j, analysis_all, run_json);
    } else {
        throw std::runtime_error("Unknown group functional: " + key);
    }

    return group;
}
