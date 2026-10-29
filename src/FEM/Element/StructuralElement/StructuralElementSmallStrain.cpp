#include "FEM/Element/StructuralElement/StructuralElementSmallStrain.hpp"

#include "Material/StructuralMaterial/Elastic/Elastic.hpp"

#include <stdexcept>

void StructuralElementSmallStrain::initialize_material_point(
    StructuralMaterial::StressState stress_state) {
    if (!material_paras)
        throw std::logic_error("Small-strain element: material parameters are missing");
    if (!materials_.empty())
        throw std::logic_error("Small-strain element: material points already initialized");
    if (material_paras->type != "elastic" && material_paras->type != "linear_elastic")
        throw std::invalid_argument("Small-strain element does not support material type: " + material_paras->type);

    const std::size_t npoint = shape_->num_integral_point();
    materials_.reserve(npoint);
    for (std::size_t ip = 0; ip < npoint; ++ip)
        materials_.emplace_back(std::make_unique<Elastic>(stress_state));
}

ElementResponse StructuralElementSmallStrain::compute_response(const Eigen::VectorXd& u,
                                                                const Eigen::MatrixXd& X,
                                                                double thickness,bool need_tangent) {
    if (u.size()!=static_cast<Eigen::Index>(dof_count()))
        throw std::invalid_argument("Element displacement size mismatch");
    ElementResponse out{Eigen::VectorXd::Zero(dof_count()), Eigen::MatrixXd()};
    if (need_tangent) out.K_tangent=Eigen::MatrixXd::Zero(dof_count(),dof_count());
    const std::size_t npoint = shape_->num_integral_point();
    for (std::size_t ip=0;ip<npoint;++ip) {
        shape_->update_at_gauss_point(ip,X);
        const auto B=bmatrix_->create(*shape_,u);
        materials_[ip]->cal_constitutive_law(bmatrix_->create_kinematics(*shape_,u), E_interp, nu_interp);
        const double w=integration_factor(thickness);
        out.f_int.noalias()+=w*B.transpose()*materials_[ip]->stress();
        if (need_tangent) out.K_tangent.noalias()+=w*B.transpose()*materials_[ip]->C_matrix()*B;
    }
    return out;
}

ElementMatrix StructuralElementSmallStrain::compute_element_matrices(const FEM& fem,double nu,double thickness) const {
    auto& self=const_cast<StructuralElementSmallStrain&>(*this);
    const double previous_E = self.E_interp;
    const double previous_nu = self.nu_interp;
    const double previous_rho = self.rho_interp;
    self.E_interp = 1.0;
    self.nu_interp = nu;
    self.rho_interp = 1.0;
    try {
        const auto X=reference_coordinates(fem);
        const auto u=Eigen::VectorXd::Zero(dof_count());
        ElementMatrix out;
        out.ndof=static_cast<int>(dof_count());
        out.ke=self.compute_response(u,X,thickness,true).K_tangent;
        out.me=self.compute_mass(X,thickness);
        self.E_interp = previous_E;
        self.nu_interp = previous_nu;
        self.rho_interp = previous_rho;
        return out;
    } catch (...) {
        self.E_interp = previous_E;
        self.nu_interp = previous_nu;
        self.rho_interp = previous_rho;
        throw;
    }
}
