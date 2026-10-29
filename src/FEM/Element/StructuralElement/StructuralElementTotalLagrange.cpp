#include "FEM/Element/StructuralElement/StructuralElementTotalLagrange.hpp"

#include "FEM/Element/Bmatrix/StructuralBmatrixTotalLagrange.hpp"
#include "Material/StructuralMaterial/StVenant/StVenant.hpp"

#include <stdexcept>

void StructuralElementTotalLagrange::initialize_material_point(
    StructuralMaterial::StressState stress_state) {
    if (!material_paras)
        throw std::logic_error("Finite-strain element: material parameters are missing");
    if (!materials_.empty())
        throw std::logic_error("Finite-strain element: material points already initialized");
    if (material_paras->type != "elastic" && material_paras->type != "linear_elastic" &&
        material_paras->type != "st_venant" && material_paras->type != "stvenant")
        throw std::invalid_argument("Finite-strain element does not support material type: " + material_paras->type);

    const std::size_t npoint = shape_->num_integral_point();
    materials_.reserve(npoint);
    for (std::size_t ip = 0; ip < npoint; ++ip)
        materials_.emplace_back(std::make_unique<StVenant>(stress_state));
}

ElementResponse StructuralElementTotalLagrange::compute_response(const Eigen::VectorXd& u,
                                                                  const Eigen::MatrixXd& X,
                                                                  double thickness,bool need_tangent) {
    if (u.size()!=static_cast<Eigen::Index>(dof_count()))
        throw std::invalid_argument("Element displacement size mismatch");

    auto* tl=dynamic_cast<StructuralBmatrixTotalLagrange*>(bmatrix_.get());

    ElementResponse out{Eigen::VectorXd::Zero(dof_count()), Eigen::MatrixXd()};

    if (need_tangent) out.K_tangent=Eigen::MatrixXd::Zero(dof_count(),dof_count());

    const std::size_t npoint = shape_->num_integral_point();
    for (std::size_t ip=0;ip<npoint;++ip) {
        shape_->update_at_gauss_point(ip,X);
        const auto B=tl->create(*shape_,u);
        materials_[ip]->cal_constitutive_law(tl->create_kinematics(*shape_,u), E_interp, nu_interp);
        const double w=integration_factor(thickness);
        out.f_int.noalias()+=w*B.transpose()*materials_[ip]->stress();
        if (need_tangent) {
            out.K_tangent.noalias()+=w*B.transpose()*materials_[ip]->C_matrix()*B;
            const auto Bnl=tl->create_NL(*shape_);
            out.K_tangent.noalias()+=w*Bnl.transpose()*tl->create_Sbar(materials_[ip]->stress())*Bnl;
        }
    }
    return out;
}

ElementMatrix StructuralElementTotalLagrange::compute_element_matrices(const FEM&,double,double) const {
    throw std::logic_error("TotalLagrange element cannot be assembled as a linear unit-modulus matrix");
}
