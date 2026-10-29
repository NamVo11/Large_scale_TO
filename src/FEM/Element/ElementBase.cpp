#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/StructuralElement/StructuralElementSmallStrain.hpp"
#include "FEM/Element/StructuralElement/StructuralElementTotalLagrange.hpp"
#include "FEM/Shape/Quad4.hpp"
#include "FEM/Shape/Tria3.hpp"
#include "FEM/Shape/Hexa8.hpp"
#include "FEM/Shape/Tetra4.hpp"
#include "FEM/Shape/Penta6.hpp"

#include <cmath>
#include <stdexcept>

std::unique_ptr<Shape> ElementBase::make_shape(ElementType type) {
    switch (type) {
        case ElementType::Q4: return std::make_unique<Quad4>();
        case ElementType::TRI3: return std::make_unique<Tria3>();
        case ElementType::H8: return std::make_unique<Hexa8>();
        case ElementType::T4: return std::make_unique<Tetra4>();
        case ElementType::P6: return std::make_unique<Penta6>();
        default: throw std::invalid_argument("Unsupported element shape");
    }
}

std::string ElementBase::name() const { return shape_->name(); }

Eigen::MatrixXd ElementBase::reference_coordinates(const FEM& fem) const {
    Eigen::MatrixXd X(node_count(),dim);
    for (std::size_t a=0;a<node_count();++a)
        for (int j=0;j<dim;++j) X(a,j) = fem.Node.at(node_ids[a])[j];
    return X;
}

std::array<double,3> ElementBase::center(const FEM& fem) const {
    std::array<double,3> c{0,0,0};
    for (int n:node_ids) for (int j=0;j<3;++j) c[j]+=fem.Node.at(n)[j];
    for (double& v:c) v/=static_cast<double>(node_count());
    return c;
}

Eigen::MatrixXd ElementBase::shape_function_matrix(const Eigen::Vector3d& xi) const {
    Eigen::VectorXd N; Eigen::MatrixXd dN;
    shape_->create_shape_function(xi,N,dN);
    return N.transpose();
}

std::vector<double> ElementBase::signed_detJ_values(const FEM& fem, const std::vector<int>& order) const {
    const auto& nodes=order.empty() ? node_ids:order;
    if (nodes.size()!=static_cast<std::size_t>(shape_->numnode())) return {};
    Eigen::MatrixXd X(nodes.size(),dim);
    for (std::size_t a=0;a<nodes.size();++a)
        for (int j=0;j<dim;++j) X(a,j)=fem.Node.at(nodes[a])[j];
    std::vector<double> values;
    const auto& rule = shape_->integration_rule(); // chỉ tính các điểm tích phân lần đầu, lưu vào biến cục bộ rule của hàm tương ứng của class NumericalIntegration.
                                                   // Nếu có nhiêu hơn 1 element cùng loại shape và cùng order, thì các element này sẽ dùng chung rule, không cần tính lại.
                                                   // Như vậy mỗi loại element sẽ có 1 rule riêng, được lưu trong hàm tương ứng của NumericalIntegration.
    for (const auto& gp:rule) {
        Eigen::VectorXd N;
        Eigen::MatrixXd dN;
        shape_->create_shape_function(gp.xi,N,dN);
        values.push_back((X.transpose()*dN).determinant());
    }
    return values;
}

double ElementBase::volume(const FEM& fem) const {
    if (volume_is_cached) return volume_value;
    const auto determinants=signed_detJ_values(fem);
    const auto& rule = shape_->integration_rule();
    volume_value=0.0; detJ_is_positive=true;
    for (std::size_t ip=0;ip<determinants.size();++ip) {
        detJ_is_positive &= determinants[ip]>0.0;
        volume_value+=std::abs(determinants[ip])*rule[ip].weight;
    }
    volume_is_cached=true;
    return volume_value;
}

double ElementBase::mesh_size(const FEM& fem) const {
    return std::pow(volume(fem),1.0/static_cast<double>(dim));
}

LocalDenseMatrix ElementBase::element_stiffness(const FEM& fem,double nu,double thickness) const {
    return compute_element_matrices(fem,nu,thickness).ke;
}
LocalDenseMatrix ElementBase::element_mass(const FEM& fem,double thickness) const {
    return compute_element_matrices(fem,0.3,thickness).me;
}

std::vector<ElementOrientationPermutation> ElementBase::orientation_check_permutations() const {
    if (dynamic_cast<const Quad4*>(shape_.get())) return {
        {"as_file_order 1-2-3-4",{0,1,2,3}},
        {"reverse_order 1-4-3-2",{0,3,2,1}},
        {"rotate_only 2-3-4-1",{1,2,3,0}},
        {"swap_1_2 2-1-3-4",{1,0,2,3}}};
    if (dynamic_cast<const Hexa8*>(shape_.get())) return {
        {"as_file_order",{0,1,2,3,4,5,6,7}},
        {"reverse_order",{0,3,2,1,4,7,6,5}}};
    return {};
}

std::shared_ptr<ElementBase> ElementBase::make_element_object(const FEM& fem,int e,
                                                              const std::string& element_type) {
    const auto formulation = myjson::lower_ascii(element_type);
    if (formulation == "small_strain")
        return std::make_shared<StructuralElementSmallStrain>(fem,e);
    if (formulation == "finite_strain")
        return std::make_shared<StructuralElementTotalLagrange>(fem,e);
    throw std::invalid_argument("Unsupported FEM.element_type: " + element_type);
}
