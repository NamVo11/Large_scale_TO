#pragma once

#include "FEM/NumericalIntegration/NumericalIntegration.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <string>
#include <vector>

// Shape chi xu ly hinh hoc/shape functions; KHONG phu thuoc SmallStrain / TotalLagrange.
// Thiet ke theo TOP_general_v1: chi 1 bo working buffers cho Gauss point hien tai.
// Khong luu std::vector<GaussPointData> cho moi element.
class Shape {
public:
    virtual ~Shape() = default;

    // Muc dich: ten loai hinh hoc, vd. Quad4, Hexa8.
    // Tra ve: ten shape.
    // Goi boi: element factory / debug.
    virtual std::string name() const = 0;

    // Muc dich: cho biet so nodes va so chieu cua shape.
    // Tra ve: so nut / so chieu (int).
    // Goi boi: StructuralElement::initialize_geometry(), kich thuoc u_e.
    virtual int numnode() const = 0;
    virtual int num_dimension() const = 0;

    // Muc dich: lay Gauss rule cho loai shape, KHONG copy rule moi lan goi.
    // Tra ve: const ref den danh sach (natural coordinates, weight) dung chung
    //         giua cac elements co cung Shape va integration order.
    // Goi boi: Shape::update_at_gauss_point(), num_integral_point(),
    //           FEM::InitializeMaterials() khi tạo một material tại mỗi Gauss point.
    virtual const std::vector<IntegrationPoint>& integration_rule() const = 0;

    // Muc dich: tinh N va dN/dXi tai 1 toa do tu nhien; khong tinh material.
    // Tra ve: void, ghi N(nNode) va dNdXi(nNode x dim) vao arguments.
    // Goi boi: Shape::update_at_gauss_point().
    virtual void create_shape_function(const Eigen::Vector3d& xi,
                                       Eigen::VectorXd& N,
                                       Eigen::MatrixXd& dNdXi) const = 0;

    // Muc dich: tai diem GP 'ip', tinh N, dNdXi, dNdX, detJ0 tu reference
    //           coordinates X_e; GHI DE bo working buffers cua Shape.
    // Tra ve: void; ket qua duoc lay qua N(), dNdX(), detJ0(), factor().
    // Goi boi: StructuralElement::create_local_matrix() trong Gauss loop;
    //           initialize_geometry() de kiem tra detJ0 va khoi tao buffer.
    // Luu y: KHONG cache cac GP, KHONG nhan displacement; khong goi dong thoi
    //         tren cung mot Shape object tu nhieu threads.
    void update_at_gauss_point(std::size_t ip,
                               const Eigen::MatrixXd& reference_node_coordinates);

    // Muc dich: lay so Gauss points cua shape.
    // Tra ve: so integration points (size_t).
    // Goi boi: StructuralElement khi tao materials_ va lap tich phan.
    std::size_t num_integral_point() const;

    // Muc dich: tra ve shape functions / reference gradients cua GP VUA tinh.
    // Tra ve: const ref, khong copy cac vector/matrix.
    // Goi boi: StructuralElement (mass, body forces), Bmatrix.
    const Eigen::VectorXd& N() const;
    const Eigen::MatrixXd& dNdX() const;

    // Muc dich: lay Jacobian reference va he so tich phan GP VUA tinh.
    // Tra ve: detJ0; factor = weight * detJ0 (CHUA nhan thickness).
    // Goi boi: StructuralElement khi tich phan fint, tangent va mass.
    double detJ0() const;
    double factor() const;

protected:
    // Mot bo working buffers tren MOI Shape object, KHONG phai moi Gauss point.
    Eigen::VectorXd N_;
    Eigen::MatrixXd dNdXi_;
    Eigen::MatrixXd dNdX_;
    double detJ0_ = 0.0;
    double factor_ = 0.0;
};
