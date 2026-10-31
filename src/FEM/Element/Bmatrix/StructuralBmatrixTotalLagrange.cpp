#include "FEM/Element/Bmatrix/StructuralBmatrixTotalLagrange.hpp"

#include <stdexcept>

Eigen::MatrixXd StructuralBmatrixTotalLagrange::create(
    const Shape& shape, const Eigen::Matrix3d& F) const {
    if (shape.num_dimension()!=dim_)
        throw std::invalid_argument("Total-Lagrange Bmatrix: shape dimension mismatch");
    const auto& g=shape.dNdX();
    Eigen::MatrixXd B=Eigen::MatrixXd::Zero(dim_==2 ? 3 : 6,dim_*shape.numnode());
    for (int a=0;a<shape.numnode();++a)
        for (int i=0;i<dim_;++i) {
            const int c=dim_*a+i;
            B(0,c)=F(i,0)*g(a,0);
            B(1,c)=F(i,1)*g(a,1);
            B(2,c)=dim_==2
                ? F(i,1)*g(a,0)+F(i,0)*g(a,1)
                : F(i,2)*g(a,2);
            if (dim_==3) {
                B(3,c)=F(i,1)*g(a,0)+F(i,0)*g(a,1);
                B(4,c)=F(i,2)*g(a,1)+F(i,1)*g(a,2);
                B(5,c)=F(i,0)*g(a,2)+F(i,2)*g(a,0);
            }
        }
    return B;
}

Eigen::MatrixXd StructuralBmatrixTotalLagrange::create_NL(
    const Shape& shape) const {
    if (shape.num_dimension()!=dim_)
        throw std::invalid_argument("Total-Lagrange BNL: shape dimension mismatch");
    Eigen::MatrixXd B=Eigen::MatrixXd::Zero(dim_*dim_,dim_*shape.numnode());
    for (int a=0;a<shape.numnode();++a)
        for (int i=0;i<dim_;++i)
            for (int j=0;j<dim_;++j)
                B(dim_*i+j,dim_*a+i)=shape.dNdX()(a,j);
    return B;
}

Eigen::MatrixXd StructuralBmatrixTotalLagrange::create_Sbar(
    const Eigen::VectorXd& stress) const {
    if (stress.size()!=(dim_==2 ? 3 : 6))
        throw std::invalid_argument("Total-Lagrange stress Voigt size mismatch");
    Eigen::Matrix3d S=Eigen::Matrix3d::Zero();
    S(0,0)=stress[0];
    S(1,1)=stress[1];
    S(0,1)=S(1,0)=stress[dim_==2 ? 2 : 3];
    if (dim_==3) {
        S(2,2)=stress[2];
        S(1,2)=S(2,1)=stress[4];
        S(2,0)=S(0,2)=stress[5];
    }
    Eigen::MatrixXd out=Eigen::MatrixXd::Zero(dim_*dim_,dim_*dim_);
    for (int i=0;i<dim_;++i)
        out.block(dim_*i,dim_*i,dim_,dim_)=S.topLeftCorner(dim_,dim_);
    return out;
}
