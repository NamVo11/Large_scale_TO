#include "Material/StructuralMaterial/StVenant/StVenant.hpp"

#include <stdexcept>

void StVenant::cal_constitutive_law(const Eigen::MatrixXd& dNdX,
                                    const Eigen::VectorXd& u_e,
                                    double young_modulus,
                                    double poisson_ratio) {
    const int dim=state_==StressState::ThreeDimensional ? 3 : 2;
    if (dNdX.cols()!=dim || u_e.size()!=dim*dNdX.rows())
        throw std::invalid_argument("StVenant gradient/displacement size mismatch");
    F_.setIdentity();
    for (Eigen::Index a=0;a<dNdX.rows();++a)
        for (int i=0;i<dim;++i)
            for (int j=0;j<dim;++j)
                F_(i,j)+=u_e[dim*a+i]*dNdX(a,j);
    if (!(F_.topLeftCorner(dim,dim).determinant()>0.0))
        throw std::runtime_error("StVenant: non-positive deformation Jacobian");
    const Eigen::Matrix3d green_strain=
        0.5*(F_.transpose()*F_-Eigen::Matrix3d::Identity());
    strain_.resize(dim==2 ? 3 : 6);
    if (dim==2)
        strain_ << green_strain(0,0),green_strain(1,1),2.0*green_strain(0,1);
    else
        strain_ << green_strain(0,0),green_strain(1,1),green_strain(2,2),
                   2.0*green_strain(0,1),2.0*green_strain(1,2),2.0*green_strain(2,0);
    set_isotropic_tangent(young_modulus, poisson_ratio);
    stress_=C_matrix_*strain_;
}
