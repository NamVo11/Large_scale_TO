#pragma once
#include "Material/StructuralMaterial/StructuralMaterial.hpp"

// small strain linear elasticity. Mot object / Gauss point.
class Elastic final : public SmallStrainMaterial {
public:
    explicit Elastic(StressState state) : SmallStrainMaterial(state) {}
    // Linear assembly currently computes only C_matrix_; strain and stress
    // must be recovered from the solved displacement before they are used.
    // Gọi bởi: StructuralElementSmallStrain::create_local_matrix().
    void cal_constitutive_law(const Eigen::MatrixXd& B,
                              const Eigen::VectorXd& u_e,
                              double young_modulus,
                              double poisson_ratio) override;
};
