#pragma once
#include "Material/StructuralMaterial/StructuralMaterial.hpp"

// Neo-Hookean hyperelastic. Mot object / Gauss point.
class NeoHookean final : public FiniteStrainMaterial {
public:
    explicit NeoHookean(StressState state) : FiniteStrainMaterial(state) {}
    // Mục đích: tính constitutive response (F (J=det F>0); 2-PK S và material tangent được suy ra từ strain energy).
    // Trả về: void; ghi vao strain_, stress_, C_matrix_ của instance.
    // Gọi bởi: StructuralElementTotalLagrange::create_local_matrix().
    void cal_constitutive_law(const Eigen::MatrixXd& dNdX,
                              const Eigen::VectorXd& u_e,
                              double young_modulus,
                              double poisson_ratio) override;
};
