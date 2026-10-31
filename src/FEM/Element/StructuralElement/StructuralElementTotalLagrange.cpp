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

ElementMatrix StructuralElementTotalLagrange::create_local_matrix(
    const FEM& fem, const Eigen::VectorXd& u, bool need_K, bool need_mass) {
    if (u.size()!=static_cast<Eigen::Index>(dof_count()))
        throw std::invalid_argument("Element displacement size mismatch");
    if (materials_.size()!=shape_->num_integral_point())
        throw std::logic_error("Finite-strain material points are not initialized");
    if (need_mass && !(rho_interp>=0.0))
        throw std::invalid_argument("Finite-strain density must be nonnegative");

    ElementMatrix out;
    out.ndof=static_cast<int>(dof_count());
    out.f_int=Eigen::VectorXd::Zero(out.ndof);
    if (need_K) out.ke=LocalDenseMatrix::Zero(out.ndof,out.ndof);
    if (need_mass) out.me=LocalDenseMatrix::Zero(out.ndof,out.ndof);

    const auto X=reference_coordinates(fem);
    for (std::size_t ip=0;ip<materials_.size();++ip) {
        shape_->update_at_gauss_point(ip,X);
        auto& material=dynamic_cast<FiniteStrainMaterial&>(*materials_[ip]);
        material.cal_constitutive_law(shape_->dNdX(),u,E_interp,nu_interp);
        const auto B=bmatrix_->create(*shape_,material.deformation_gradient());
        const double w=integration_factor(fem.thickness);
        out.f_int.noalias()+=w*B.transpose()*material.stress();
        if (need_K) {
            out.ke.noalias()+=w*B.transpose()*material.C_matrix()*B;
            const auto Bnl=bmatrix_->create_NL(*shape_);
            out.ke.noalias()+=w*Bnl.transpose()*bmatrix_->create_Sbar(material.stress())*Bnl;
        }
        if (need_mass) {
            const double mass_factor=w*rho_interp;
            for (int a=0;a<shape_->numnode();++a)
                for (int b=0;b<shape_->numnode();++b)
                    for (int j=0;j<dim;++j)
                        out.me(dim*a+j,dim*b+j)+= mass_factor*shape_->N()[a]*shape_->N()[b];
        }
    }
    return out;
}
