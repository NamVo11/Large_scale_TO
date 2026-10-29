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

Eigen::MatrixXd StructuralElement::compute_mass(const Eigen::MatrixXd& X,double thickness) {
    if (materials_.size()!=shape_->num_integral_point())
        throw std::runtime_error("StructuralElement materials not initialized");
    if (!(rho_interp >= 0.0)) throw std::invalid_argument("StructuralElement: invalid density");
    Eigen::MatrixXd M=Eigen::MatrixXd::Zero(dof_count(),dof_count());
    for (std::size_t ip=0;ip<materials_.size();++ip) {
        shape_->update_at_gauss_point(ip,X);
        const double factor=integration_factor(thickness)*rho_interp;
        for (int a=0;a<shape_->numnode();++a) for (int b=0;b<shape_->numnode();++b)
            for (int j=0;j<dim;++j)
                M(dim*a+j,dim*b+j)+=factor*shape_->N()[a]*shape_->N()[b];
    }
    return M;
}

void StructuralElement::commit_materials() { for (auto& m:materials_) m->commit(); }
void StructuralElement::rollback_materials() { for (auto& m:materials_) m->rollback(); }
StructuralMaterial& StructuralElement::material(std::size_t ip) { return *materials_.at(ip); }
const StructuralMaterial& StructuralElement::material(std::size_t ip) const { return *materials_.at(ip); }
