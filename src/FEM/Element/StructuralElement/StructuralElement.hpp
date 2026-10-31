#pragma once

#include "FEM/Element/ElementBase.hpp"
#include "FEM/Shape/Shape.hpp"
#include "Material/MaterialParas.hpp"
#include "Material/StructuralMaterial/StructuralMaterial.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <vector>

// Tuong tu TOP_general_v1: moi element so huu 1 Shape, 1 Bmatrix,
// va nhieu StructuralMaterial objects (1 object / Gauss point).
class StructuralElement : public ElementBase {
public:
    StructuralElement(const FEM& fem, int e) : ElementBase(fem, e) {}
    virtual ~StructuralElement() = default;

    // Muc dich: kiem tra hinh hoc reference va detJ0, khoi tao working buffers
    //           cua Shape; KHONG precompute/cache du lieu moi Gauss point.
    // Tra ve: void; Shape van chi co 1 bo buffer hien tai.
    // Goi boi: FEM::precompute_elements() sau khi co reference X_e.
    void initialize_geometry(const Eigen::MatrixXd& X_e);

    // FEM assigns MaterialParas first; each concrete element creates its own GP materials.
    virtual void initialize_material_point(StructuralMaterial::StressState stress_state) = 0;

    // Uses this element's interpolated E and rho. Small-strain elements use
    // unit coefficients when FEM requests shared uniform matrices.
    virtual ElementMatrix create_local_matrix(const FEM& fem,
                                              const Eigen::VectorXd& u_e,
                                              bool need_K,
                                              bool need_mass) = 0;

    // Current interpolated values belong to this element, never to region-shared MaterialParas.
    double E_interp = 0.0;
    double nu_interp = 0.0;
    double rho_interp = 0.0;

    // Muc dich: chap nhan history tai moi Material sau khi load/time step hoi tu.
    // Tra ve: void.
    // Goi boi: Nonlinear Analysis sau Newton convergence.
    void commit_materials();

    // Muc dich: rollback material trial histories neu step bi reject.
    // Tra ve: void.
    // Goi boi: Nonlinear Analysis khi Newton fail / line-search reject.
    void rollback_materials();

    // Muc dich: truy cap Material object tai Gauss point ip.
    // Tra ve: reference StructuralMaterial, khong copy.
    // Goi boi: Element integration, post-processing, GP-level verification.
    StructuralMaterial& material(std::size_t ip);
    const StructuralMaterial& material(std::size_t ip) const;

    std::shared_ptr<MaterialParas> material_paras;

protected:
    double integration_factor(double thickness) const;
    std::vector<std::unique_ptr<StructuralMaterial>> materials_;
};
