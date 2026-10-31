#pragma once
#include "Material/StructuralMaterial/StructuralMaterial.hpp"

// St. Venant-Kirchhoff (geometrically nonlinear, elastic). Mot object / Gauss point.
class StVenant final : public FiniteStrainMaterial {
public:
    explicit StVenant(StressState state) : FiniteStrainMaterial(state) {}
    // Mục đích: tính constitutive response (Green strain E; 2-PK S = C:E; consistent C).
    // Trả về: void; ghi vao strain_, stress_, C_matrix_ của instance.
    // Gọi bởi: StructuralElementTotalLagrange::create_local_matrix().
    void cal_constitutive_law(const Eigen::MatrixXd& dNdX,
                              const Eigen::VectorXd& u_e,
                              double young_modulus,
                              double poisson_ratio) override;
};
