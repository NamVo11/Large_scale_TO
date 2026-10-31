#include "FEM/Element/StructuralElement/StructuralElement.hpp"

#include <stdexcept>

void StructuralElement::initialize_geometry(const Eigen::MatrixXd& X) {
    const std::size_t npoint = shape_->num_integral_point();
    for (std::size_t ip=0;ip<npoint;++ip)
        shape_->update_at_gauss_point(ip,X);
}

double StructuralElement::integration_factor(double thickness) const {
    if (dim==2 && !(thickness>0.0)) throw std::invalid_argument("Positive thickness required");
    return shape_->factor()*(dim==2 ? thickness:1.0);
}

void StructuralElement::commit_materials() { for (auto& m:materials_) m->commit(); }
void StructuralElement::rollback_materials() { for (auto& m:materials_) m->rollback(); }
StructuralMaterial& StructuralElement::material(std::size_t ip) { return *materials_.at(ip); }
const StructuralMaterial& StructuralElement::material(std::size_t ip) const { return *materials_.at(ip); }
