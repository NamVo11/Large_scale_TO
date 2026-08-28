#include "Optimization/Volume/VolumeFunctional.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/FEM.hpp"

#include <stdexcept>

void VolumeFunctionalBase::initialize(const FEM& fem,
                                      Analysis& analysis,
                                      const Eigen::MatrixXd& xPhys) {
    calculate(fem, analysis, xPhys);
    initial_value = value;
    if (initial_value == 0.0) {
        throw std::runtime_error("Volume functional has zero initial_value: " + function_name);
    }
    scale = weight / initial_value;
}

void VolumeFractionTerm::calculate(const FEM& fem,
                                   Analysis&,
                                   const Eigen::MatrixXd& xPhys) {
    const int nvar = static_cast<int>(xPhys.cols());
    const int val_id = mat_id - 2;
    if (val_id < 0 || val_id >= nvar) throw std::runtime_error("volume_fraction: invalid mat_id");
    if (static_cast<int>(fem.volume.size()) != fem.NElem) {
        throw std::runtime_error("volume_fraction: fem volume is not initialized");
    }

    const Eigen::Map<const Eigen::VectorXd> volume_vec(fem.volume.data(), fem.NElem);
    const double total_volume = volume_vec.sum();
    if (!(total_volume > 0.0)) throw std::runtime_error("volume_fraction: non-positive total volume");

    value = volume_vec.dot(xPhys.col(val_id)) / total_volume;
}

Eigen::VectorXd VolumeFractionTerm::dfdx(const FEM& fem,
                                         Analysis&,
                                         const Eigen::MatrixXd& xPhys,
                                         int val_id) {
    const int nvar = static_cast<int>(xPhys.cols());
    const int term_val_id = mat_id - 2;
    if (term_val_id < 0 || term_val_id >= nvar) throw std::runtime_error("volume_fraction: invalid mat_id");
    if (val_id != term_val_id) return Eigen::VectorXd();
    if (static_cast<int>(fem.volume.size()) != fem.NElem) {
        throw std::runtime_error("volume_fraction: fem volume is not initialized");
    }

    const Eigen::Map<const Eigen::VectorXd> volume_vec(fem.volume.data(), fem.NElem);
    const double total_volume = volume_vec.sum();
    if (!(total_volume > 0.0)) throw std::runtime_error("volume_fraction: non-positive total volume");
    return volume_vec / total_volume;
}
