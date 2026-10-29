#include "FEM/Element/Bmatrix/StructuralBmatrixTotalLagrange.hpp"

#include <stdexcept>

namespace {
Eigen::Matrix3d deformation_gradient(const Shape& shape, const Eigen::VectorXd& u, int dim) {
    if (shape.num_dimension()!=dim || u.size()!=dim*shape.numnode())
        throw std::invalid_argument("TotalLagrange Bmatrix: dimensions mismatch");
    Eigen::Matrix3d F=Eigen::Matrix3d::Identity();
    for (int a=0;a<shape.numnode();++a)
        for (int i=0;i<dim;++i)
            for (int j=0;j<dim;++j)
                F(i,j)+=u[dim*a+i]*shape.dNdX()(a,j);
    if (!(F.topLeftCorner(dim,dim).determinant()>0.0))
        throw std::runtime_error("TotalLagrange: non-positive deformation Jacobian");
    return F;
}

KinematicData kinematics(const Shape& shape, const Eigen::VectorXd& u, int dim) {
    const Eigen::Matrix3d F=deformation_gradient(shape,u,dim);
    const Eigen::Matrix3d E=0.5*(F.transpose()*F-Eigen::Matrix3d::Identity());
    Eigen::VectorXd strain(dim==2 ? 3:6);
    if (dim==2) strain << E(0,0),E(1,1),2*E(0,1);
    else strain << E(0,0),E(1,1),E(2,2),2*E(0,1),2*E(1,2),2*E(2,0);
    return {KinematicFormulation::TotalLagrange,strain,F};
}

Eigen::MatrixXd linear_B(const Shape& shape, const Eigen::VectorXd& u, int dim) {
    const auto F=deformation_gradient(shape,u,dim);
    const auto& g=shape.dNdX();
    Eigen::MatrixXd B=Eigen::MatrixXd::Zero(dim==2 ? 3:6,u.size());
    for (int a=0;a<shape.numnode();++a) for (int i=0;i<dim;++i) {
        const int c=dim*a+i;
        B(0,c)=F(i,0)*g(a,0);
        B(1,c)=F(i,1)*g(a,1);
        B(2,c)=dim==2 ? F(i,1)*g(a,0)+F(i,0)*g(a,1) : F(i,2)*g(a,2);
        if (dim==3) {
            B(3,c)=F(i,1)*g(a,0)+F(i,0)*g(a,1);
            B(4,c)=F(i,2)*g(a,1)+F(i,1)*g(a,2);
            B(5,c)=F(i,0)*g(a,2)+F(i,2)*g(a,0);
        }
    }
    return B;
}

Eigen::MatrixXd nonlinear_B(const Shape& shape, int dim) {
    Eigen::MatrixXd B=Eigen::MatrixXd::Zero(dim*dim,dim*shape.numnode());
    for (int a=0;a<shape.numnode();++a)
        for (int i=0;i<dim;++i)
            for (int j=0;j<dim;++j)
                B(dim*i+j,dim*a+i)=shape.dNdX()(a,j);
    return B;
}

Eigen::MatrixXd stress_bar(const Eigen::VectorXd& s, int dim) {
    if (s.size()!=(dim==2 ? 3:6)) throw std::invalid_argument("Stress Voigt size mismatch");
    Eigen::Matrix3d S=Eigen::Matrix3d::Zero();
    S(0,0)=s[0]; S(1,1)=s[1]; S(0,1)=S(1,0)=s[dim==2 ? 2:3];
    if (dim==3) {
        S(2,2)=s[2]; S(1,2)=S(2,1)=s[4]; S(2,0)=S(0,2)=s[5];
    }
    Eigen::MatrixXd out=Eigen::MatrixXd::Zero(dim*dim,dim*dim);
    for (int i=0;i<dim;++i) out.block(dim*i,dim*i,dim,dim)=S.topLeftCorner(dim,dim);
    return out;
}
}

Eigen::MatrixXd StructuralBmatrixTotalLagrange2D::create(const Shape& s,const Eigen::VectorXd& u) const { return linear_B(s,u,2); }
KinematicData StructuralBmatrixTotalLagrange2D::create_kinematics(const Shape& s,const Eigen::VectorXd& u) const { return kinematics(s,u,2); }
Eigen::MatrixXd StructuralBmatrixTotalLagrange2D::create_NL(const Shape& s) const { return nonlinear_B(s,2); }
Eigen::MatrixXd StructuralBmatrixTotalLagrange2D::create_Sbar(const Eigen::VectorXd& s) const { return stress_bar(s,2); }

Eigen::MatrixXd StructuralBmatrixTotalLagrange3D::create(const Shape& s,const Eigen::VectorXd& u) const { return linear_B(s,u,3); }
KinematicData StructuralBmatrixTotalLagrange3D::create_kinematics(const Shape& s,const Eigen::VectorXd& u) const { return kinematics(s,u,3); }
Eigen::MatrixXd StructuralBmatrixTotalLagrange3D::create_NL(const Shape& s) const { return nonlinear_B(s,3); }
Eigen::MatrixXd StructuralBmatrixTotalLagrange3D::create_Sbar(const Eigen::VectorXd& s) const { return stress_bar(s,3); }
