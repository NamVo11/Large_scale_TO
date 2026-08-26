#include "ROM/ROM_method/ROM_Base.hpp"

#include "ROM/Error_Estimator/RelativeResidualEstimator.hpp"
#include "ROM/ROM_method/FOM_direct.hpp"
#include "ROM/ROM_method/FOM_iterative.hpp"
#include "ROM/ROM_method/MyStrategy.hpp"
#include "ROM/ROM_method/MyStrategy2.hpp"
#include "ROM/ROM_method/PetscFOM_direct.hpp"
#include "ROM/ROM_method/PetscFOM_iterative.hpp"
#include "ROM/Reduced_Basis_Algorithm/IncrementalSVD.hpp"
#include "ROM/Reduced_Basis_Algorithm/MyIncrementalQR1.hpp"
#include "ROM/Reduced_Basis_Algorithm/MyPOD_SVD1.hpp"

#include <cstdlib>
#include <stdexcept>

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
    if (solver == "direct") return std::make_shared<FOM_direct>();
    if (solver == "pcg" || solver == "iterative" || solver == "recycling_pcg") {
        return std::make_shared<FOM_iterative>("pcg");
    }
    throw std::runtime_error("Unknown FOM solver: " + fom_solver_type);
}

std::shared_ptr<ROMBase> InitializeDynamicROM(const std::string& method_name,
                                              const std::string& fom_solver_type,
                                              double residual_tol,
                                              int rank1,
                                              int max_rank,
                                              bool online,
                                              const std::string& linear_backend) {
    const std::string method = myjson::lower_ascii(method_name);
    if (method != "mystrategy" && method != "mystrategy2") {
        throw std::runtime_error("Eigen backend currently implements MyStrategy/MyStrategy2");
    }

    std::vector<std::shared_ptr<UpdateBase>> basis_list;
    basis_list.push_back(std::make_shared<MyPOD_SVD1>(0.0, 1e-10, rank1));
    if (method == "mystrategy2") {
        basis_list.push_back(std::make_shared<MyIncrementalQR1>(1e-20, max_rank));
        return std::make_shared<MyStrategy2>(
            basis_list,
            residual_tol,
            0,
            InitializeFOMSolver(fom_solver_type, linear_backend),
            std::make_shared<RelativeResidualEstimator>(residual_tol),
            online);
    }
    basis_list.push_back(std::make_shared<IncrementalSVD>(10000, 1e-10, 1e-12, max_rank));
    return std::make_shared<MyStrategy>(
        basis_list,
        residual_tol,
        0,
        InitializeFOMSolver(fom_solver_type, linear_backend),
        std::make_shared<RelativeResidualEstimator>(residual_tol),
        online);
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
    return InitializeDynamicROM(rom_json.at("method_name").get<std::string>(),
                                fom_solver_type,
                                rom_json.at("rom_tol").get<double>(),
                                rom_json.at("rank1").get<int>(),
                                rom_json.at("max_rank").get<int>(),
                                rom_json.at("online").get<bool>(),
                                linear_backend);
}
