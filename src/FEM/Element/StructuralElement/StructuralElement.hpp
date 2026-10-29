#pragma once

#include "FEM/Element/ElementBase.hpp"
#include "FEM/Element/Bmatrix/StructuralBmatrix.hpp"
#include "FEM/Shape/Shape.hpp"
#include "Material/MaterialParas.hpp"
#include "Material/StructuralMaterial/StructuralMaterial.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <vector>

// Ket qua o cap ELEMENT, khong luu contribution K_g tai moi Gauss point.
struct ElementResponse {
    Eigen::VectorXd f_int;
    Eigen::MatrixXd K_tangent; // rong khi need_tangent=false
};

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

    // Muc dich: lap qua cac Gauss points, goi Shape::update_at_gauss_point(ip,X_e),
    //           Bmatrix, Material[ip], sau do tich phan f_int^e va K_t^e.
    // Tra ve: ElementResponse {f_int, K_tangent}.
    // Goi boi: Nonlinear FOM assembly / Newton; ROM + ECSW chi selected elements.
    // Luu y: X_e duoc cung cap moi lan evaluate de KHONG nhan ban reference
    //         nodal coordinates ben trong moi Shape; caller co the gom X_e 1 lan / element.
    virtual ElementResponse compute_response(const Eigen::VectorXd& u_e,
                                             const Eigen::MatrixXd& X_e,
                                             double thickness,
                                             bool need_tangent = true) = 0;

    // Muc dich: tich phan M_e = sum N^T * rho * N * detJ0 * weight.
    // Tra ve: dense element mass matrix (nDOF x nDOF).
    // Goi boi: Linear/Dynamic assembly; xay dung/cache mass khi can.
    Eigen::MatrixXd compute_mass(const Eigen::MatrixXd& X_e,
                                 double thickness);

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
    std::unique_ptr<StructuralBmatrix> bmatrix_;
    std::vector<std::unique_ptr<StructuralMaterial>> materials_;
};
