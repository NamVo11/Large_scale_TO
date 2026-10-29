#include "Material/StructuralMaterial/StVenant/StVenant.hpp"

#include <stdexcept>

void StVenant::cal_constitutive_law(const KinematicData& k,
                                    double E, double nu) {
    if (k.formulation!=KinematicFormulation::TotalLagrange)
        throw std::invalid_argument("StVenant requires total-Lagrange kinematics");
    set_isotropic_tangent(E, nu);
    if (k.strain.size()!=C_matrix_.rows()) throw std::invalid_argument("StVenant strain size mismatch");
    strain_=k.strain;
    stress_=C_matrix_*strain_;
}
