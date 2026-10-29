#pragma once
#include "Material/StructuralMaterial/StructuralMaterial.hpp"

// small strain linear elasticity. Mot object / Gauss point.
class Elastic final : public StructuralMaterial {
public:
    explicit Elastic(StressState state) : StructuralMaterial(state) {}
    // Mục đích: tính constitutive response (eps = B*u_e; sigma = C:eps; C không phụ thuộc u).
    // Trả về: void; ghi vao strain_, stress_, C_matrix_ của instance.
    // Gọi bởi: StructuralElementSmallStrain::compute_response().
    void cal_constitutive_law(const KinematicData& kinematics,
                              double E, double nu) override;
};
