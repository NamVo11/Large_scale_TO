#include "Material/StructuralMaterial/StructuralMaterial.hpp"

#include <cmath>
#include <stdexcept>

const Eigen::VectorXd& StructuralMaterial::strain() const { return strain_; }
const Eigen::VectorXd& StructuralMaterial::stress() const { return stress_; }
const Eigen::MatrixXd& StructuralMaterial::C_matrix() const { return C_matrix_; }
void StructuralMaterial::commit() {}
void StructuralMaterial::rollback() {}

void StructuralMaterial::set_isotropic_tangent(double E, double nu)
{
    // Validate material parameters
    if (!(E > 0.0) ||
        !(nu > -1.0 && nu < 0.5))
    {
        throw std::invalid_argument("StructuralMaterial: invalid E or nu");
    }

    // Lamé parameters
    const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu = E / (2.0 * (1.0 + nu));

    // 1. Plane Stress
    if (state_ == StressState::PlaneStress) {

        const double c = E / (1.0 - nu * nu);

        C_matrix_ = Eigen::MatrixXd::Zero(3, 3);

        C_matrix_(0,0) = c;
        C_matrix_(0,1) = c * nu;

        C_matrix_(1,0) = c * nu;
        C_matrix_(1,1) = c;

        C_matrix_(2,2) = mu;
    }

    // 2. Plane Strain
    else if (state_ == StressState::PlaneStrain) {

        C_matrix_ = Eigen::MatrixXd::Zero(3, 3);

        C_matrix_(0,0) = lambda + 2.0 * mu;
        C_matrix_(0,1) = lambda;

        C_matrix_(1,0) = lambda;
        C_matrix_(1,1) = lambda + 2.0 * mu;

        C_matrix_(2,2) = mu;
    }

    // 3. Three-Dimensional
    else {

        C_matrix_ = Eigen::MatrixXd::Zero(6, 6);

        C_matrix_(0,0) = lambda + 2.0 * mu;
        C_matrix_(0,1) = lambda;
        C_matrix_(0,2) = lambda;

        C_matrix_(1,0) = lambda;
        C_matrix_(1,1) = lambda + 2.0 * mu;
        C_matrix_(1,2) = lambda;

        C_matrix_(2,0) = lambda;
        C_matrix_(2,1) = lambda;
        C_matrix_(2,2) = lambda + 2.0 * mu;

        C_matrix_(3,3) = mu;
        C_matrix_(4,4) = mu;
        C_matrix_(5,5) = mu;
    }
}
