#include "FEM/Element/Bmatrix/StructuralBmatrix.hpp"

#include <stdexcept>

namespace {
void check(const Shape& shape, const Eigen::VectorXd& u, int dim) {
    if (shape.num_dimension()!=dim || u.size()!=dim*shape.numnode())
        throw std::invalid_argument("Bmatrix: shape/displacement dimension mismatch");
}
}

Eigen::MatrixXd StructuralBmatrix2D::create(const Shape& shape, const Eigen::VectorXd& u) const {
    check(shape,u,2);
    Eigen::MatrixXd B=Eigen::MatrixXd::Zero(3,u.size());
    const auto& g=shape.dNdX();
    for (int a=0;a<shape.numnode();++a) {
        B(0,2*a)=g(a,0); B(1,2*a+1)=g(a,1);
        B(2,2*a)=g(a,1); B(2,2*a+1)=g(a,0);
    }
    return B;
}

KinematicData StructuralBmatrix2D::create_kinematics(const Shape& shape, const Eigen::VectorXd& u) const {
    KinematicData k{KinematicFormulation::SmallStrain, create(shape,u)*u, Eigen::Matrix3d::Identity()};
    return k;
}

Eigen::MatrixXd StructuralBmatrix3D::create(const Shape& shape, const Eigen::VectorXd& u) const {
    check(shape,u,3);
    Eigen::MatrixXd B=Eigen::MatrixXd::Zero(6,u.size());
    const auto& g=shape.dNdX();
    for (int a=0;a<shape.numnode();++a) {
        const int c=3*a;
        B(0,c)=g(a,0); B(1,c+1)=g(a,1); B(2,c+2)=g(a,2);
        B(3,c)=g(a,1); B(3,c+1)=g(a,0);
        B(4,c+1)=g(a,2); B(4,c+2)=g(a,1);
        B(5,c)=g(a,2); B(5,c+2)=g(a,0);
    }
    return B;
}

KinematicData StructuralBmatrix3D::create_kinematics(const Shape& shape, const Eigen::VectorXd& u) const {
    KinematicData k{KinematicFormulation::SmallStrain, create(shape,u)*u, Eigen::Matrix3d::Identity()};
    return k;
}
