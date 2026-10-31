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

// Small-strain materials receive the B matrix already computed by the element.
class SmallStrainMaterial : public StructuralMaterial {
public:
    explicit SmallStrainMaterial(StressState state) : StructuralMaterial(state) {}
    virtual void cal_constitutive_law(const Eigen::MatrixXd& B,
                                      const Eigen::VectorXd& u_e,
                                      double young_modulus,
                                      double poisson_ratio) = 0;
};

// Finite-strain materials own the deformation gradient at their Gauss point.
class FiniteStrainMaterial : public StructuralMaterial {
public:
    explicit FiniteStrainMaterial(StressState state) : StructuralMaterial(state) {}
    virtual void cal_constitutive_law(const Eigen::MatrixXd& dNdX,
                                      const Eigen::VectorXd& u_e,
                                      double young_modulus,
                                      double poisson_ratio) = 0;
    const Eigen::Matrix3d& deformation_gradient() const { return F_; }

protected:
    Eigen::Matrix3d F_ = Eigen::Matrix3d::Identity();
};
