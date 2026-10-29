#include "FEM/Shape/Tetra4.hpp"

std::string Tetra4::name() const { return "Tetra4"; }
int Tetra4::numnode() const { return 4; }
int Tetra4::num_dimension() const { return 3; }
const std::vector<IntegrationPoint>& Tetra4::integration_rule() const {
    return NumericalIntegration::tetra4();
}
void Tetra4::create_shape_function(const Eigen::Vector3d& xi, Eigen::VectorXd& N, Eigen::MatrixXd& dN) const {
    N.resize(4); N << 1-xi[0]-xi[1]-xi[2],xi[0],xi[1],xi[2];
    dN.resize(4,3); dN << -1,-1,-1, 1,0,0, 0,1,0, 0,0,1;
}
