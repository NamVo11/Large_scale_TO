#pragma once

#include "FEM/Shape/Shape.hpp"
#include "Material/Material.hpp"
#include <Eigen/Dense>

// Bmatrix doc gradient cua Gauss point HIEN TAI tu Shape (working buffer),
// khong can GaussPointData, khong tu truy cap global FEM nodes.
class Bmatrix {
public:
    virtual ~Bmatrix() = default;

    // Muc dich: tao strain-displacement B tu Shape::dNdX() va u_e.
    // Tra ve: B matrix; SmallStrain: B, TotalLagrange: B_L(u_e).
    // Goi boi: StructuralElementSmallStrain/TotalLagrange::compute_response().
    // Dieu kien: Shape::update_at_gauss_point(ip, X_e) da duoc goi.
    virtual Eigen::MatrixXd create(const Shape& shape,
                                   const Eigen::VectorXd& u_e) const = 0;

    // Muc dich: tinh small strain hoac Green-Lagrange strain/deformation gradient.
    // Tra ve: KinematicData dua cho constitutive law.
    // Goi boi: StructuralElement::compute_response() -> Material::cal_constitutive_law().
    virtual KinematicData create_kinematics(const Shape& shape,
                                            const Eigen::VectorXd& u_e) const = 0;
};
