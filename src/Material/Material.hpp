#pragma once
#include <Eigen/Dense>

enum class KinematicFormulation { SmallStrain, TotalLagrange };

// Du lieu kinematics đầu vào cho constitutive law (được Bmatrix tính).
// Khong phai state object; Material instance TU lưu response/history của no.
struct KinematicData {
    KinematicFormulation formulation;
    Eigen::VectorXd strain; // eps hoac Green-Lagrange E theo Voigt convention.
    Eigen::Matrix3d F;      // deformation gradient; TL can, small strain co the bo qua.
};

// Response/history interface; each Gauss point owns one instance.
class Material {
public:
    virtual ~Material() = default;
};
