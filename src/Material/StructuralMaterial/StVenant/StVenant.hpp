#pragma once
#include "Material/StructuralMaterial/StructuralMaterial.hpp"

// St. Venant-Kirchhoff (geometrically nonlinear, elastic). Mot object / Gauss point.
class StVenant final : public StructuralMaterial {
public:
    explicit StVenant(StressState state) : StructuralMaterial(state) {}
    // Mục đích: tính constitutive response (Green strain E; 2-PK S = C:E; consistent C).
    // Trả về: void; ghi vao strain_, stress_, C_matrix_ của instance.
    // Gọi bởi: StructuralElementTotalLagrange::compute_response().
    void cal_constitutive_law(const KinematicData& kinematics,
                              double E, double nu) override;
};
