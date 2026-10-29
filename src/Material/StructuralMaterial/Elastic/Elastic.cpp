#include "Material/StructuralMaterial/Elastic/Elastic.hpp"

#include <stdexcept>

void Elastic::cal_constitutive_law(const KinematicData& k,
                                  double E, double nu) {
    if (k.formulation!=KinematicFormulation::SmallStrain)
        throw std::invalid_argument("Elastic requires small-strain kinematics");
    set_isotropic_tangent(E, nu);
    if (k.strain.size()!=C_matrix_.rows()) throw std::invalid_argument("Elastic strain size mismatch");
    strain_=k.strain;
    stress_=C_matrix_*strain_;
}
