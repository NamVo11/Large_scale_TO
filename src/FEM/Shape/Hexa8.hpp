#pragma once
#include "FEM/Shape/Shape.hpp"

// Hexa8: hinh hoc va shape functions; doc lap voi formulation / material.
class Hexa8 final : public Shape {
public:
    // Muc dich: cung cap ten, so nodes, so chieu cua Hexa8.
    // Tra ve: name, numnode, num_dimension.
    // Goi boi: element factory, Shape/StructuralElement dimension checks.
    std::string name() const override;
    int numnode() const override;
    int num_dimension() const override;

    // Muc dich: lay Gauss quadrature rule hexa8 dung chung cho moi Hexa8.
    // Tra ve: const ref den static, immutable vector IntegrationPoint.
    // Goi boi: Shape::update_at_gauss_point(), num_integral_point().
    // Rule duoc luu duy nhat trong NumericalIntegration::hexa8().
    const std::vector<IntegrationPoint>& integration_rule() const override;

    // Muc dich: tinh N va dNdXi tai mot natural coordinate xi.
    // Tra ve: void; ghi vao N, dNdXi (working buffers cua Shape).
    // Goi boi: Shape::update_at_gauss_point().
    void create_shape_function(const Eigen::Vector3d& xi,
                               Eigen::VectorXd& N,
                               Eigen::MatrixXd& dNdXi) const override;
};
