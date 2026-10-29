#include "FEM/Shape/Tria3.hpp"

std::string Tria3::name() const { return "Tria3"; }
int Tria3::numnode() const { return 3; }
int Tria3::num_dimension() const { return 2; }
const std::vector<IntegrationPoint>& Tria3::integration_rule() const {
    return NumericalIntegration::tria3();
}
void Tria3::create_shape_function(const Eigen::Vector3d& xi, Eigen::VectorXd& N, Eigen::MatrixXd& dN) const {
    N.resize(3); N << 1-xi[0]-xi[1],xi[0],xi[1];
    dN.resize(3,2); dN << -1,-1, 1,0, 0,1;
}
