#pragma once
#include "FEM/FEM.hpp"
#include "FEM/Shape/Shape.hpp"
#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <vector>

struct ElementOrientationPermutation {
    std::string name;
    std::vector<int> order;
};

class ElementBase {
public:
    ElementBase(const FEM& fem, int elem_local_id) : local_id(elem_local_id),
          raw_id(elem_local_id >= 0 && elem_local_id < static_cast<int>(fem.element_ids.size())
                     ? fem.element_ids[elem_local_id] : elem_local_id + 1),
          dim(fem.dim),
          node_ids(fem.Element.at(elem_local_id)),
          dof_ids(fem.ElementDofs.at(elem_local_id)),
          shape_(make_shape(fem.ElementTypes.at(elem_local_id)))
        {
            if (!shape_ || shape_->numnode() != static_cast<int>(node_count()) ||
            shape_->num_dimension() != dim)
            throw std::invalid_argument("ElementBase: incompatible shape and mesh");
        }
    virtual ~ElementBase() = default;

    int local_id = -1;
    int raw_id = -1;
    int dim = 0;
    const std::vector<int>& node_ids;
    const std::vector<PetscInt>& dof_ids;
    std::string status = "design_val";
    mutable bool volume_is_cached = false;
    mutable double volume_value = 0.0;
    mutable bool detJ_is_positive = true;

    // Mục đích: lay ID element trong mang local FEM.
    // Trả về: int local element ID.
    // Gọi bởi: global assembly, output, ECSW element sampling.
    int local_index() const { return local_id; }

    // Mục đích: lay danh sach DOFs của element.
    // Trả về: const ref vector PETSc global DOF IDs.
    // Gọi bởi: Eigen/PETSc assembly; ROM để extract Phi_e và u_e.
    const std::vector<PetscInt>& dofs() const { return dof_ids; }

    // Mục đích: lay danh sach nodes của element.
    // Trả về: const ref vector FEM local node IDs.
    // Gọi bởi: FEM mesh/precompute geometry, output.
    const std::vector<int>& nodes() const { return node_ids; }

    virtual std::string name() const;
    virtual double volume(const FEM& fem) const;
    virtual double mesh_size(const FEM& fem) const;
    virtual Eigen::MatrixXd shape_function_matrix(const Eigen::Vector3d& xi) const;
    virtual LocalDenseMatrix element_stiffness(const FEM& fem, double nu, double thickness) const;
    virtual LocalDenseMatrix element_mass(const FEM& fem, double thickness) const;
    virtual std::vector<double> signed_detJ_values(const FEM& fem,
                                                   const std::vector<int>& node_order = {}) const;
    virtual std::vector<ElementOrientationPermutation> orientation_check_permutations() const;
    std::array<double,3> center(const FEM& fem) const;
    std::size_t node_count() const { return node_ids.size(); }
    std::size_t dof_count() const { return dof_ids.size(); }
    Eigen::MatrixXd reference_coordinates(const FEM& fem) const;

    // Mục đích: API tuong thich vói assembler LINEAR hien tại.
    // Trả về: normalized unit-modulus Ke và Me theo convention cu.
    // Gọi bởi: EigenStaticLinearElastic::assemble_K(), assemble_K_M(), Petsc counterparts.
    // CHÚ Ý: giữ Eeff nhan ở assembler trong giai đoạn migration, tranh double E.
    virtual ElementMatrix compute_element_matrices(const FEM& fem,
                                                   double nu,
                                                   double thickness) const = 0;

    static std::shared_ptr<ElementBase> make_element_object(const FEM& fem,
                                                            int elem_local_id,
                                                            const std::string& element_type);

protected:
    static std::unique_ptr<Shape> make_shape(ElementType type);
    std::unique_ptr<Shape> shape_;
};
