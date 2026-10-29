#pragma once
#include "Material/Material.hpp"
#include <Eigen/Dense>

// MỘT instance cho MỘT Gauss point.
// One object owns response and internal history, not duplicated input/effective parameters.
// Khong tach MaterialPointState hay MaterialResponse thanh object riêng.
class StructuralMaterial : public Material {
public:
    enum class StressState { PlaneStress, PlaneStrain, ThreeDimensional };
    explicit StructuralMaterial(StressState state) : state_(state) {}
    virtual ~StructuralMaterial() = default;

    // Mục đích: tính strain/stress/tangent và lưu VAO chính material object.
    // Trả về: void; kết quả lay qua stress(), strain(), C_matrix().
    // Gọi bởi: StructuralElementSmallStrain/TotalLagrange loop theo Gauss point.
    virtual void cal_constitutive_law(const KinematicData& kinematics,
                                      double E, double nu) = 0;

    // Mục đích: lay strain da tính tại Gauss point.
    // Trả về: const ref vector strain (small strain hoac Green-Lagrange E).
    // Gọi bởi: output, postprocess và unit tests.
    const Eigen::VectorXd& strain() const;

    // Mục đích: lay stress da tính tại Gauss point.
    // Trả về: const ref vector stress (Cauchy cho small-strain, 2-PK cho TL).
    // Gọi bởi: StructuralElement tích phân f_int và K_geo; postprocess.
    const Eigen::VectorXd& stress() const;

    // Mục đích: lay constitutive consistent tangent tại Gauss point.
    // Trả về: const ref ma trận C (Voigt consistent với Bmatrix).
    // Gọi bởi: StructuralElement tích phân K_material.
    const Eigen::MatrixXd& C_matrix() const;

    // Mục đích: xac nhan trial-history sau KHI load step hội tụ (nếu history-dependent).
    // Trả về: void; với elastic/hyperelastic có thể no-op.
    // Gọi bởi: Analysis sau Newton convergence, KHÔNG gọi mỗi iteration.
    virtual void commit();

    // Mục đích: hoan lại history ve committed state nếu thứ Newton bị từ choi.
    // Trả về: void; elastic/hyperelastic có thể no-op.
    // Gọi bởi: Analysis khi Newton fail / line-search reject.
    virtual void rollback();

protected:
    StressState state_;
    void set_isotropic_tangent(double E, double nu);
    Eigen::VectorXd strain_;
    Eigen::VectorXd stress_;
    Eigen::MatrixXd C_matrix_;
    // Internal history của visco/plastic được khai bao tại derived class.
};
