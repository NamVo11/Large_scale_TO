#include "FEM/Shape/Quad4.hpp"

std::string Quad4::name() const { return "Quad4"; }

int Quad4::numnode() const { return 4; }

int Quad4::num_dimension() const { return 2; }

const std::vector<IntegrationPoint>& Quad4::integration_rule() const {
    return NumericalIntegration::quad4();
}
void Quad4::create_shape_function(const Eigen::Vector3d& xi, Eigen::VectorXd& N, Eigen::MatrixXd& dN) const {
    const double r=xi[0], s=xi[1];
    N.resize(4); dN.resize(4,2);
    const int sign[4][2]={{-1,-1},{1,-1},{1,1},{-1,1}};
    for (int a=0;a<4;++a) {
        N[a]=0.25*(1+sign[a][0]*r)*(1+sign[a][1]*s);
        dN(a,0)=0.25*sign[a][0]*(1+sign[a][1]*s);
        dN(a,1)=0.25*sign[a][1]*(1+sign[a][0]*r);
    }
}
