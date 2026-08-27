#include "ROM/ROM_method/ROM_Base.hpp"

#include "ROM/Error_Estimator/RelativeResidualEstimator.hpp"
#include "ROM/ROM_method/FOM.hpp"
#include "ROM/ROM_method/MyStrategy.hpp"
#include "ROM/Reduced_Basis_Algorithm/IncrementalSVD.hpp"
#include "ROM/Reduced_Basis_Algorithm/MyIncrementalQR1.hpp"
#include "ROM/Reduced_Basis_Algorithm/MyPOD_SVD1.hpp"

#include <cstdlib>
#include <stdexcept>

namespace {

std::string update_name_value(const myjson::json& j, const char* key, const std::string& fallback) {
    if (!j.contains(key) || j.at(key).is_null()) return fallback;
    if (j.at(key).is_boolean()) return j.at(key).get<bool>() ? fallback : "false";
    return j.at(key).get<std::string>();
}

std::shared_ptr<UpdateBase> InitializeUpdateMethod(const std::string& update_name,
                                                   int rank,
                                                   bool offline_slot) {
    const std::string update = myjson::lower_ascii(update_name);
    if (update == "false" || update == "none" || update == "off") {
        if (offline_slot) return std::make_shared<MyPOD_SVD1>(0.0, 1e-10, 0);
        return std::make_shared<IncrementalSVD>(10000, 1e-10, 1e-12, 0);
    }
    if (update == "pod_svd" || update == "pod" || update == "svd") {
        return std::make_shared<MyPOD_SVD1>(0.0, 1e-10, rank);
    }
    if (update == "isvd" || update == "incrementalsvd" || update == "incremental_svd") {
        return std::make_shared<IncrementalSVD>(10000, 1e-10, 1e-12, rank);
    }
    if (update == "iqr" || update == "incrementalqr" || update == "incremental_qr") {
        return std::make_shared<MyIncrementalQR1>(1e-20, rank);
    }
    throw std::runtime_error("Unknown ROM update method: " + update_name);
}

} // namespace

std::shared_ptr<ROMBase> InitializeFOMSolver(const std::string& fom_solver_type,
                                             const std::string& linear_backend) {
    const std::string backend = myjson::lower_ascii(linear_backend);
    const std::string solver = myjson::lower_ascii(fom_solver_type);
    if (backend == "petsc") {
        if (solver == "direct") {
            return std::make_shared<PetscFOM_direct>();
        }
        if (solver == "pcg" || solver == "iterative" || solver == "recycling_pcg") {
            return std::make_shared<PetscFOM_iterative>("pcg");
        }
        throw std::runtime_error("Unknown PETSc FOM solver: " + fom_solver_type);
    }
    if (solver == "direct") return std::make_shared<EigenFOM_direct>();
    if (solver == "pcg" || solver == "iterative" || solver == "recycling_pcg") {
        return std::make_shared<EigenFOM_iterative>("pcg");
    }
    throw std::runtime_error("Unknown FOM solver: " + fom_solver_type);
}

std::shared_ptr<ROMBase> InitializeDynamicROM(const std::string& method_name,
                                              const std::string& fom_solver_type,
                                              double residual_tol,
                                              int rank1,
                                              int max_rank,
                                              const std::string& offline_update,
                                              const std::string& online_update,
                                              const std::string& linear_backend) {
    const std::string method = myjson::lower_ascii(method_name);
    if (method != "mystrategy") throw std::runtime_error("Unknown ROM strategy: " + method_name);

    std::vector<std::shared_ptr<UpdateBase>> basis_list;
    basis_list.push_back(InitializeUpdateMethod(offline_update, rank1, true));
    basis_list.push_back(InitializeUpdateMethod(online_update, max_rank, false));
    const bool use_offline_update = myjson::lower_ascii(offline_update) != "false";
    const bool use_online_update = myjson::lower_ascii(online_update) != "false";
    const bool use_incremental_projection = use_online_update &&
                                            (myjson::lower_ascii(online_update) == "iqr" ||
                                             myjson::lower_ascii(online_update) == "incrementalqr" ||
                                             myjson::lower_ascii(online_update) == "incremental_qr");
    return std::make_shared<MyStrategy>(
                                        basis_list,
                                        residual_tol,
                                        0,
                                        InitializeFOMSolver(fom_solver_type, linear_backend),
                                        std::make_shared<RelativeResidualEstimator>(residual_tol),
                                        use_offline_update,
                                        use_online_update,
                                        use_incremental_projection);
}

std::shared_ptr<ROMBase> InitializeROM(const myjson::json& rom_json,
                                       const myjson::json& run_json) {
                                        
    const std::string linear_backend = run_json.at("linear_backend").get<std::string>();
    const std::string solve_mode = myjson::lower_ascii(rom_json.at("solve_mode").get<std::string>());
    const std::string fom_solver_type = rom_json.at("fom_solver_type").get<std::string>();
    const std::string pod_backend = rom_json.at("pod_backend").get<std::string>();
    const std::string online_backend = rom_json.at("online_backend").get<std::string>();
    if (!pod_backend.empty()) setenv("STAGE2_POD_BACKEND", pod_backend.c_str(), 1);
    if (!online_backend.empty()) setenv("STAGE2_ONLINE_BACKEND", online_backend.c_str(), 1);

    if (solve_mode != "rom") return InitializeFOMSolver(fom_solver_type, linear_backend);
    const std::string offline_update = update_name_value(rom_json, "offline_update", "pod_svd");
    const std::string online_update = update_name_value(rom_json, "online_update", "isvd");
    return InitializeDynamicROM(rom_json.at("method_name").get<std::string>(),
                                fom_solver_type,
                                rom_json.at("rom_tol").get<double>(),
                                myjson::int_value(rom_json, "rank1", 0),
                                myjson::int_value(rom_json, "max_rank", 0),
                                offline_update,
                                online_update,
                                linear_backend);
}
