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
    ElementMatrix create_local_matrix(const FEM& fem,
                                      const Eigen::VectorXd& u_e,
                                      bool need_K,
                                      bool need_mass) override;

private:
    std::unique_ptr<StructuralBmatrixTotalLagrange> bmatrix_;
};
