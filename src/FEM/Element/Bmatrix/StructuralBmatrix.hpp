#pragma once

#include "FEM/Element/Bmatrix/Bmatrix.hpp"

// Base class cho cac Bmatrix cua structural mechanics.
class StructuralBmatrix : public Bmatrix {
public:
    ~StructuralBmatrix() override = default;
};

class StructuralBmatrix2D final : public StructuralBmatrix {
public:
    // Muc dich: tao B small-strain 2D tu Shape::dNdX().
    // Tra ve: matrix 3 x (2*nNode), Voigt {xx,yy,xy_engineering}.
    // Goi boi: StructuralElementSmallStrain::compute_response() o moi GP.
    Eigen::MatrixXd create(const Shape& shape,
                           const Eigen::VectorXd& u_e) const override;

    // Muc dich: tinh eps = B * u_e (small strain 2D).
    // Tra ve: KinematicData chua eps, formulation=SmallStrain.
    // Goi boi: StructuralElementSmallStrain -> material(ip).cal_constitutive_law().
    KinematicData create_kinematics(const Shape& shape,
                                    const Eigen::VectorXd& u_e) const override;
};

class StructuralBmatrix3D final : public StructuralBmatrix {
public:
    // Muc dich: tao B small-strain 3D tu Shape::dNdX().
    // Tra ve: matrix 6 x (3*nNode), theo Voigt convention thong nhat.
    // Goi boi: StructuralElementSmallStrain::compute_response() o moi GP.
    Eigen::MatrixXd create(const Shape& shape,
                           const Eigen::VectorXd& u_e) const override;

    // Muc dich: tinh eps = B * u_e (small strain 3D).
    // Tra ve: KinematicData chua eps, formulation=SmallStrain.
    // Goi boi: StructuralElementSmallStrain -> material(ip).cal_constitutive_law().
    KinematicData create_kinematics(const Shape& shape,
                                    const Eigen::VectorXd& u_e) const override;
};
