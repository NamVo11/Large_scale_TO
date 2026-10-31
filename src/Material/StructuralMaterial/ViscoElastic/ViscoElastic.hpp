#pragma once
#include "Material/StructuralMaterial/StructuralMaterial.hpp"

// viscoelastic history-dependent. Mot object / Gauss point.
class ViscoElastic final : public SmallStrainMaterial {
public:
    explicit ViscoElastic(StressState state) : SmallStrainMaterial(state) {}
    // Mục đích: tính constitutive response (strain/strain-rate/time-step history và algorithmic tangent).
    // Trả về: void; ghi vao strain_, stress_, C_matrix_ của instance.
    // Gọi bởi: StructuralElementSmallStrain hoac dynamic structural element::create_local_matrix().
    void cal_constitutive_law(const Eigen::MatrixXd& B,
                              const Eigen::VectorXd& u_e,
                              double young_modulus,
                              double poisson_ratio) override;

    // Mục đích: commit trial viscoelastic history sau hội tụ.
    // Trả về: void; copy/update history noi bộ trong material object.
    // Gọi bởi: Analysis khi converged load/time step.
    void commit() override;

    // Mục đích: rollback trial viscoelastic history khi Newton fail/line search reject.
    // Trả về: void; restore history từ committed copy.
    // Gọi bởi: Analysis khi step bị reject.
    void rollback() override;
};
