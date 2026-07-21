#include "Eigen_backend/ROM/ROM_method/ROM_Base.hpp"

#include "Eigen_backend/ROM/Error_Estimator/RelativeResidualEstimator.hpp"
#include "Eigen_backend/ROM/ROM_method/FOM_direct.hpp"
#include "Eigen_backend/ROM/ROM_method/FOM_iterative.hpp"
#include "Eigen_backend/ROM/ROM_method/MyStrategy.hpp"
#include "Eigen_backend/ROM/Utils/rom_eigen_utils.hpp"
#include "MatlabLikeData.hpp"
#include "ROM/Common/Reduced_Basis_Agorithm/IncrementalSVD.hpp"
#include "ROM/Common/Reduced_Basis_Agorithm/MyIncrementalQR1.hpp"
#include "ROM/Common/Reduced_Basis_Agorithm/MyPOD_SVD1.hpp"

#include <stdexcept>

namespace eigen_backend {

std::shared_ptr<ROMBase> make_fom_solver(const std::string& fom_solver_type) {
    const std::string solver = lower_string(fom_solver_type);
    if (solver == "direct") return std::make_shared<FOM_direct>();
    if (solver == "pcg" || solver == "iterative" || solver == "recycling_pcg") {
        return std::make_shared<FOM_iterative>("pcg");
    }
    throw std::runtime_error("Unknown Eigen FOM solver: " + fom_solver_type);
}

std::shared_ptr<ROMBase> make_dynamic_ROM(const std::string& method_name,
                                          const std::string& fom_solver_type,
                                          double residual_tol,
                                          int rank1,
                                          int max_rank,
                                          bool online) {
    const std::string method = lower_string(method_name);
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

std::shared_ptr<ROMBase> make_eigen_rom_from_analysis(const Analysis& analysis) {
    if (!analysis.rom_use_mor) return make_fom_solver(analysis.fom_solver_type);
    return make_dynamic_ROM(analysis.rom_method_name,
                            analysis.fom_solver_type,
                            analysis.rom_tol,
                            analysis.rom_rank1,
                            analysis.rom_max_rank,
                            analysis.rom_online);
}

std::shared_ptr<ROMBase> make_eigen_rom_from_objective(const ObjectiveGroup& object) {
    if (!object.rom_use_mor) return make_fom_solver(object.fom_solver_type);
    return make_dynamic_ROM(object.rom_method_name,
                            object.fom_solver_type,
                            object.rom_tol,
                            object.rom_rank1,
                            object.rom_max_rank,
                            object.rom_online);
}

} // namespace eigen_backend
