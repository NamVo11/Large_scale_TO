#include "FEM/Shape/Penta6.hpp"

std::string Penta6::name() const { return "Penta6"; }
int Penta6::numnode() const { return 6; }
int Penta6::num_dimension() const { return 3; }
const std::vector<IntegrationPoint>& Penta6::integration_rule() const {
    return NumericalIntegration::penta6();
}
void Penta6::create_shape_function(const Eigen::Vector3d& xi, Eigen::VectorXd& N, Eigen::MatrixXd& dN) const {
    const double r=xi[0],s=xi[1],t=xi[2];
    const double L[3]={1-r-s,r,s};
    const double dL[3][2]={{-1,-1},{1,0},{0,1}};
    N.resize(6); dN.resize(6,3);
    for (int layer=0;layer<2;++layer) {
        const double z=layer ? 1.0 : -1.0;
        for (int a=0;a<3;++a) {
            const int i=3*layer+a;
            N[i]=0.5*L[a]*(1+z*t);
            dN(i,0)=0.5*dL[a][0]*(1+z*t);
            dN(i,1)=0.5*dL[a][1]*(1+z*t);
            dN(i,2)=0.5*z*L[a];
        }
    }
}
