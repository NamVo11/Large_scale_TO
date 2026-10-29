#pragma once
#include "FEM/Element/StructuralElement/StructuralElement.hpp"

// SmallStrain dung StructuralBmatrix2D/3D, B khong phu thuoc u_e.
class StructuralElementSmallStrain final : public StructuralElement {
public:
    StructuralElementSmallStrain(const FEM& fem, int e) : StructuralElement(fem, e) {
        if (dim == 2) {
            bmatrix_ = std::make_unique<StructuralBmatrix2D>();
        } else if (dim == 3) {
            bmatrix_ = std::make_unique<StructuralBmatrix3D>();
        } else {
            throw std::invalid_argument("Small-strain element: unsupported dimension");
        }
    }
    void initialize_material_point(StructuralMaterial::StressState stress_state) override;
    // Muc dich: Gauss loop -> f_int^e=sum B^T sigma dV, K=sum B^T C B dV.
    // Tra ve: ElementResponse {f_int, K_tangent}.
    // Goi boi: SmallStrain Analysis, unit tests, ROM if applicable.
    ElementResponse compute_response(const Eigen::VectorXd& u_e,
                                     const Eigen::MatrixXd& X_e,
                                     double thickness,
                                     bool need_tangent = true) override;

    // Muc dich: giu bridge linear assembly API hien tai, Ke chuan hoa E=1.
    // Tra ve: ElementMatrix {unit-modulus Ke, unit-density Me}.
    // Goi boi: Eigen/PETSc linear assemblers trong migration.
    // Chu y: legacy assembler tiep tuc nhan Eeff/rhoeff NGOAI element.
    ElementMatrix compute_element_matrices(const FEM& fem,
                                           double nu,
                                           double thickness) const override;

};
