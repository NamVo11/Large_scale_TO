#pragma once
#include "FEM/Element/StructuralElement/StructuralElement.hpp"
#include "FEM/Element/Bmatrix/StructuralBmatrix.hpp"

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
    ElementMatrix create_local_matrix(const FEM& fem,
                                      const Eigen::VectorXd& u_e,
                                      bool need_K,
                                      bool need_mass) override;
    // Unit kernels required by linear sensitivities on nonuniform meshes.
    ElementMatrix create_unit_local_matrix(const FEM& fem,
                                           const Eigen::VectorXd& u_e,
                                           bool need_K,
                                           bool need_mass);

private:
    std::unique_ptr<StructuralBmatrix> bmatrix_;
    ElementMatrix integrate_local_matrix(const FEM& fem,
                                         const Eigen::VectorXd& u_e,
                                         double stiffness,
                                         double density,
                                         bool need_K,
                                         bool need_mass);
};
