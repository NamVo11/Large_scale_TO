#include "FEM/Shape/Hexa8.hpp"

std::string Hexa8::name() const { return "Hexa8"; }
int Hexa8::numnode() const { return 8; }
int Hexa8::num_dimension() const { return 3; }
const std::vector<IntegrationPoint>& Hexa8::integration_rule() const {
    return NumericalIntegration::hexa8();
}
void Hexa8::create_shape_function(const Eigen::Vector3d& xi, Eigen::VectorXd& N, Eigen::MatrixXd& dN) const {
    static const int s[8][3]={{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
                               {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
    N.resize(8); dN.resize(8,3);
    for (int a=0;a<8;++a) {
        N[a]=0.125*(1+s[a][0]*xi[0])*(1+s[a][1]*xi[1])*(1+s[a][2]*xi[2]);
        for (int j=0;j<3;++j) {
            const int k=(j+1)%3, l=(j+2)%3;
            dN(a,j)=0.125*s[a][j]*(1+s[a][k]*xi[k])*(1+s[a][l]*xi[l]);
        }
    }
}
