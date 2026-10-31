#include "Analysis/StaticLinearElastic/StaticLinearElasticAVM.hpp"

#include "Analysis/Analysis.hpp"
#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/StructuralElement/StructuralElement.hpp"
#include "FEM/Element/StructuralElement/StructuralElementSmallStrain.hpp"
#include "Material/StructuralMaterial/Elastic/ElasticParas.hpp"
#include "ROM/ROM_method/ROM_Base.hpp"

#include <petscsys.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <omp.h>

namespace {

int nvar_of(const Eigen::MatrixXd& x) {
    return static_cast<int>(x.cols());
}

void zero_fixed(Eigen::VectorXd& v, const std::vector<int>& fixed) {
    for (int id : fixed) v(id) = 0.0;
}

Eigen::VectorXd sized_or_zero(const Eigen::VectorXd& values, int size, const std::string& label) {
    if (values.size() == 0) return Eigen::VectorXd::Zero(size);
    if (values.size() != size) throw std::runtime_error(label + ": vector size mismatch");
    return values;
}

Eigen::MatrixXd explicit_gradient_static(const FEM& fem,
                                         Analysis& analysis,
                                         const Eigen::MatrixXd& xPhys,
                                         StaticLinearElasticAVM& object,
                                         const Eigen::VectorXd& lambda) {
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    int sens_element_threads = 1;
    if (const char* env = std::getenv("SENS_ELEMENT_THREADS")) {
        sens_element_threads = std::max(1, std::atoi(env));
    }
    if (size != 1) sens_element_threads = 1;

    const int nvar = nvar_of(xPhys);
    Eigen::MatrixXd df_dx = Eigen::MatrixXd::Zero(fem.NElem, nvar);
    const Eigen::VectorXd u = analysis.U.col(analysis.Nstep);

    for (const auto& region_ptr : fem.Regions) {
        const Region& region = *region_ptr;
        if (region.status != "design_val") continue;
        if (region.elem_ids.empty()) continue;
        const auto region_element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(region.elem_ids.front())));
        if (!region_element || !region_element->material_paras) {
            throw std::runtime_error("Element material is not initialized");
        }
        const auto& region_material = dynamic_cast<const ElasticParas&>(*region_element->material_paras);
        if (region_material.val_ids.empty()) throw std::runtime_error("explicit_gradient_static: no val_ids");
        if (region_material.E_values.empty()) throw std::runtime_error("explicit_gradient_static: no elastic parameters");
        if (!analysis.interp) throw std::runtime_error("Analysis interpolation is not initialized");

        const double region_nu = region_material.nu_values.front();
        ElementMatrix uniform_em;
        if (fem.uniform_element_matrices) {
            if (region.elem_ids.empty()) throw std::runtime_error("explicit_gradient_static: region has no elements");
            region_element->nu_interp = region_nu;
            uniform_em = region_element->create_local_matrix(fem, Eigen::VectorXd::Zero(region_element->dof_count()), true, false);
        }

        const int region_nelem = static_cast<int>(region.elem_ids.size());
        #pragma omp parallel for schedule(static) num_threads(sens_element_threads) if(sens_element_threads > 1)
        for (int local_e = 0; local_e < region_nelem; ++local_e) {
            const int elem_id = region.elem_ids[static_cast<std::size_t>(local_e)];
            ElementMatrix local_em;
            const ElementMatrix* em_ptr = &uniform_em;
            if (!fem.uniform_element_matrices) {
                auto& sensitivity_element = dynamic_cast<StructuralElementSmallStrain&>(*fem.ElementObjects.at(static_cast<std::size_t>(elem_id)));
                sensitivity_element.nu_interp = region_nu;
                local_em = sensitivity_element.create_unit_local_matrix(fem, Eigen::VectorXd::Zero(sensitivity_element.dof_count()), true, false);
                em_ptr = &local_em;
            }
            const ElementMatrix& em = *em_ptr;
            const auto& dofs = fem.ElementDofs.at(static_cast<std::size_t>(elem_id));

            Eigen::Matrix<double, 24, 1> u_elem_buffer;
            Eigen::Matrix<double, 24, 1> lambda_elem_buffer;
            Eigen::Matrix<double, 24, 1> Ku_elem_buffer;
            for (int i = 0; i < em.ndof; ++i) {
                const int dof_i = static_cast<int>(dofs[static_cast<std::size_t>(i)]);
                u_elem_buffer(i) = u(dof_i);
                lambda_elem_buffer(i) = lambda(dof_i);
            }
            const auto u_elem = u_elem_buffer.head(em.ndof);
            const auto lambda_elem = lambda_elem_buffer.head(em.ndof);
            auto Ku_elem = Ku_elem_buffer.head(em.ndof);
            Ku_elem.noalias() = em.ke * u_elem;
            const double lambda_dot_Ku = lambda_elem.dot(Ku_elem);

            const auto element = std::dynamic_pointer_cast<StructuralElement>(fem.ElementObjects.at(static_cast<std::size_t>(elem_id)));
            if (!element || !element->material_paras) throw std::runtime_error("Element material is not initialized");
            const auto& material = dynamic_cast<const ElasticParas&>(*element->material_paras);
            for (int val_id : material.val_ids) {
                if (val_id <= 0 || val_id > nvar) throw std::runtime_error("explicit_gradient_static: val_id outside xPhys");
                const double x = xPhys(elem_id, val_id - 1);
                const double dE = analysis.interp->sensitivity(material.E_values, x,
                    myjson::lower_ascii(element->status) == "design_val", val_id, analysis.interp->modulus_penalty());
                df_dx(elem_id, val_id - 1) -= dE * lambda_dot_Ku;
            }
        }
    }

    for (int val_id = 0; val_id < nvar; ++val_id) {
        const Eigen::VectorXd explicit_dfdx = object.dfdx(fem, analysis, xPhys, val_id);
        if (explicit_dfdx.size() == 0) continue;
        if (explicit_dfdx.size() != df_dx.rows()) {
            throw std::runtime_error("explicit_gradient_static: explicit dfdx size mismatch");
        }
        df_dx.col(val_id).noalias() += explicit_dfdx;
    }
    if (rank == 0) {
        std::cout << "  explicit_gradient_static: threads=" << sens_element_threads << "\n";
    }
    return df_dx;
}

} // namespace

Eigen::MatrixXd EigenStaticLinearElasticAVM::AVM(const FEM& fem,
                                                Analysis& analysis_base,
                                                const Eigen::MatrixXd&,
                                                const std::vector<int>& x_label,
                                                const Filtering& filtering,
                                                int Iter) {
    auto& analysis = dynamic_cast<EigenStaticLinearElastic&>(analysis_base);
    if (!ROM) throw std::runtime_error("EigenStaticLinearElasticAVM requires ROM initialized");
    if (analysis.K_inf.rows() == 0) {
        throw std::runtime_error("EigenStaticLinearElasticAVM requires Eigen forward stiffness matrix");
    }

    Eigen::VectorXd rhs = sized_or_zero(dfdu(fem, analysis, filtering.xPhys, analysis.Nstep),
                                        fem.ndof,
                                        "EigenStaticLinearElasticAVM::dfdu");
    zero_fixed(rhs, analysis.SPCdofs);
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(rhs.size());
    if (analysis.factorized_K_initial) {
        ROM->set_factorized_solver(analysis.K_inf, analysis.factorized_K_initial);
    }
    ROM->InitializeROMContext(analysis, Iter, 1, analysis.Nstep, 1.0, "adjoint", "static_adjoint", true);
    const SolveStats stats = ROM->main(analysis.K_inf, rhs, lambda);
    ROM->set_factorized_solver(analysis.K_inf, nullptr);
    (void)stats;

    Eigen::MatrixXd out = filtering.sensitivity(
        explicit_gradient_static(fem, analysis, filtering.xPhys, *this, lambda));
    for (Eigen::Index e = 0; e < out.rows(); ++e) {
        if (x_label[static_cast<std::size_t>(e)] != 0) out.row(e).setZero();
    }
    return out;
}
