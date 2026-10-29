#pragma once

#include "FEM/Element/Bmatrix/StructuralBmatrix.hpp"

// Total Lagrange: B_L(u_e), B_NL va Sbar dung cho material/geometric tangents.
class StructuralBmatrixTotalLagrange : public StructuralBmatrix {
public:
    ~StructuralBmatrixTotalLagrange() override = default;

    // Muc dich: tao B_NL su dung trong K_geo.
    // Tra ve: B_NL, kich thuoc 4 x nDOF (2D) hoac 9 x nDOF (3D).
    // Goi boi: StructuralElementTotalLagrange::compute_response() o moi GP.
    virtual Eigen::MatrixXd create_NL(const Shape& shape) const = 0;

    // Muc dich: chuyen stress Second Piola-Kirchhoff sang block matrix Sbar.
    // Tra ve: Sbar co kich thuoc 4x4 (2D) hoac 9x9 (3D).
    // Goi boi: StructuralElementTotalLagrange khi tich phan K_geo.
    virtual Eigen::MatrixXd create_Sbar(const Eigen::VectorXd& stress) const = 0;
};

class StructuralBmatrixTotalLagrange2D final : public StructuralBmatrixTotalLagrange {
public:
    // Muc dich: tao B_L(F, dNdX) tai Gauss point hien tai.
    // Tra ve: B_L kich thuoc 3 x (2*nNode).
    // Goi boi: StructuralElementTotalLagrange::compute_response().
    Eigen::MatrixXd create(const Shape& shape,
                           const Eigen::VectorXd& u_e) const override;

    // Muc dich: tinh F va Green-Lagrange strain E (TL 2D).
    // Tra ve: KinematicData(formulation=TotalLagrange).
    // Goi boi: StructuralElementTotalLagrange -> Material::cal_constitutive_law().
    KinematicData create_kinematics(const Shape& shape,
                                    const Eigen::VectorXd& u_e) const override;

    // Muc dich: tao B_NL, Sbar cho geometric tangent 2D.
    // Tra ve: B_NL 4 x nDOF, Sbar 4x4.
    // Goi boi: StructuralElementTotalLagrange::compute_response().
    Eigen::MatrixXd create_NL(const Shape& shape) const override;
    Eigen::MatrixXd create_Sbar(const Eigen::VectorXd& stress) const override;
};

class StructuralBmatrixTotalLagrange3D final : public StructuralBmatrixTotalLagrange {
public:
    // Muc dich: tao B_L(F, dNdX) tai Gauss point hien tai.
    // Tra ve: B_L kich thuoc 6 x (3*nNode).
    // Goi boi: StructuralElementTotalLagrange::compute_response().
    Eigen::MatrixXd create(const Shape& shape,
                           const Eigen::VectorXd& u_e) const override;

    // Muc dich: tinh F va Green-Lagrange strain E (TL 3D).
    // Tra ve: KinematicData(formulation=TotalLagrange).
    // Goi boi: StructuralElementTotalLagrange -> Material::cal_constitutive_law().
    KinematicData create_kinematics(const Shape& shape,
                                    const Eigen::VectorXd& u_e) const override;

    // Muc dich: tao B_NL, Sbar cho geometric tangent 3D.
    // Tra ve: B_NL 9 x nDOF, Sbar 9x9.
    // Goi boi: StructuralElementTotalLagrange::compute_response().
    Eigen::MatrixXd create_NL(const Shape& shape) const override;
    Eigen::MatrixXd create_Sbar(const Eigen::VectorXd& stress) const override;
};
