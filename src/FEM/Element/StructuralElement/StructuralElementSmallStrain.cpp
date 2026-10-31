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

ElementMatrix StructuralElementSmallStrain::create_local_matrix(
    const FEM& fem, const Eigen::VectorXd& u, bool need_K, bool need_mass) {
    const double stiffness=fem.uniform_element_matrices ? 1.0 : E_interp;
    const double density=fem.uniform_element_matrices ? 1.0 : rho_interp;
    return integrate_local_matrix(fem,u,stiffness,density,need_K,need_mass);
}

ElementMatrix StructuralElementSmallStrain::create_unit_local_matrix(
    const FEM& fem, const Eigen::VectorXd& u, bool need_K, bool need_mass) {
    return integrate_local_matrix(fem,u,1.0,1.0,need_K,need_mass);
}

ElementMatrix StructuralElementSmallStrain::integrate_local_matrix(
    const FEM& fem, const Eigen::VectorXd& u, double stiffness, double density,
    bool need_K, bool need_mass) {
    if (u.size()!=static_cast<Eigen::Index>(dof_count()))
        throw std::invalid_argument("Element displacement size mismatch");
    if (materials_.size()!=shape_->num_integral_point())
        throw std::logic_error("Small-strain material points are not initialized");
    ElementMatrix out;
    out.ndof=static_cast<int>(dof_count());
    if (need_K) out.ke=LocalDenseMatrix::Zero(out.ndof,out.ndof);
    if (need_mass) {
        if (!(density>=0.0)) throw std::invalid_argument("Small-strain density must be nonnegative");
        out.me=LocalDenseMatrix::Zero(out.ndof,out.ndof);
    }
    if (!need_K && !need_mass) return out;
    const auto X=reference_coordinates(fem);
    for (std::size_t ip=0;ip<materials_.size();++ip) {
        shape_->update_at_gauss_point(ip,X);
        const double w=integration_factor(fem.thickness);
        if (need_K) {
            const auto B=bmatrix_->create(*shape_,u);
            auto& material=dynamic_cast<SmallStrainMaterial&>(*materials_[ip]);
            material.cal_constitutive_law(B,u,stiffness,nu_interp);
            out.ke.noalias()+=w*B.transpose()*materials_[ip]->C_matrix()*B;
        }
        if (need_mass) {
            const double mass_factor=w*density;
            for (int a=0;a<shape_->numnode();++a)
                for (int b=0;b<shape_->numnode();++b)
                    for (int j=0;j<dim;++j)
                        out.me(dim*a+j,dim*b+j)+=
                            mass_factor*shape_->N()[a]*shape_->N()[b];
        }
    }
    return out;
}
