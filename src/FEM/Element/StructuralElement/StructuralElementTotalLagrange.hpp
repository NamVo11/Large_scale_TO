#pragma once
#include "FEM/Element/StructuralElement/StructuralElement.hpp"
#include "FEM/Element/Bmatrix/StructuralBmatrixTotalLagrange.hpp"

// TotalLagrange dung StructuralBmatrixTotalLagrange2D/3D, geometric stiffness.
class StructuralElementTotalLagrange final : public StructuralElement {
public:
    StructuralElementTotalLagrange(const FEM& fem, int e) : StructuralElement(fem, e) {
        if (dim == 2) {
            bmatrix_ = std::make_unique<StructuralBmatrixTotalLagrange2D>();
        } else if (dim == 3) {
            bmatrix_ = std::make_unique<StructuralBmatrixTotalLagrange3D>();
        } else {
            throw std::invalid_argument("Total-Lagrange element: unsupported dimension");
        }
    }
    void initialize_material_point(StructuralMaterial::StressState stress_state) override;
    // Muc dich: Gauss loop -> stress 2-PK, f_int, K_material + K_geometric.
    // Tra ve: ElementResponse tai current u_e.
    // Goi boi: Newton nonlinear FOM; ROM/ECSW sampled element evaluation.
    ElementResponse compute_response(const Eigen::VectorXd& u_e,
                                     const Eigen::MatrixXd& X_e,
                                     double thickness,
                                     bool need_tangent = true) override;

    // Muc dich: bridge virtual API cu (co the throw not-supported for TL).
    // Tra ve: ElementMatrix, hoac throw co thong bao ro.
    // Goi boi: legacy linear assembler neu bi goi nham; nonlinear khong dung.
    ElementMatrix compute_element_matrices(const FEM& fem,
                                           double nu,
                                           double thickness) const override;

};
