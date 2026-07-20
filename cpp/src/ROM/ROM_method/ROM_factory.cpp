#include "ROM/ROM_method/ROM_Base.hpp"

#include "ROM/Error_Estimator/RelativeResidualEstimator.hpp"
#include "ROM/Reduced_Basis_Agorithm/IncrementalSVD.hpp"
#include "ROM/Reduced_Basis_Agorithm/MyIncrementalQR1.hpp"
#include "ROM/Reduced_Basis_Agorithm/MyPOD_SVD1.hpp"
#include "ROM/ROM_method/FOM_direct.hpp"
#include "ROM/ROM_method/FOM_iterative.hpp"
#include "ROM/ROM_method/MyStrategy.hpp"
#include "ROM/Utils/rom_petsc_utils.hpp"

#include <stdexcept>

std::shared_ptr<ROMBase> make_fom_solver(const std::string& fom_solver_type) {
    const std::string solver = lower_string(fom_solver_type);
    if (solver == "direct") return std::make_shared<FOM_direct>();
    if (solver == "pcg" || solver == "iterative" || solver == "recycling_pcg") {
        return std::make_shared<FOM_iterative>("pcg");
    }
    throw std::runtime_error("Unknown FOM solver: " + fom_solver_type);
}

std::shared_ptr<ROMBase> make_dynamic_ROM(const std::string& method_name,
                                          const std::string& fom_solver_type,
                                          double residual_tol,
                                          int rank1,
                                          int max_rank,
                                          bool online) {
    const std::string method = lower_string(method_name);
    if (method != "mystrategy" && method != "mystrategy2") {
        throw std::runtime_error("Stage2 currently implements MyStrategy/MyStrategy2 in C++ ROM tree");
    }

    std::vector<std::shared_ptr<UpdateBase>> basis_list;
    basis_list.push_back(std::make_shared<MyPOD_SVD1>(0.0, 1e-10, rank1));
    if (method == "mystrategy2") {
        basis_list.push_back(std::make_shared<MyIncrementalQR1>(1e-20, max_rank));
        return std::make_shared<MyStrategy2>(
            basis_list,
            residual_tol,
            0,
            make_fom_solver(fom_solver_type),
            std::make_shared<RelativeResidualEstimator>(residual_tol),
            online);
    }
    basis_list.push_back(std::make_shared<IncrementalSVD>(10000, 1e-10, 1e-12, max_rank));
    return std::make_shared<MyStrategy>(
        basis_list,
        residual_tol,
        0,
        make_fom_solver(fom_solver_type),
        std::make_shared<RelativeResidualEstimator>(residual_tol),
        online);
}

