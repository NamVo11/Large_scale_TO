#pragma once

#include "FEM/FEM.hpp"

#include <Eigen/Dense>

struct ElementOrientationPermutation {
    std::string name;
    std::vector<int> order;
};

struct ElementBase {
    int local_id = -1;
    int raw_id = -1;
    ElementType type = ElementType::Unknown;
    int dim = 0;
    std::vector<int> node_ids;
    std::vector<PetscInt> dof_ids;
    mutable bool volume_is_cached = false;
    mutable double volume_value = 0.0;
    mutable bool detJ_is_positive = true;

    ElementBase(const FEM& fem, int elem_local_id);
    virtual ~ElementBase() = default;

    virtual std::string name() const = 0;
    virtual double volume(const FEM& fem) const = 0;
    virtual double mesh_size(const FEM& fem) const = 0;
    virtual Eigen::MatrixXd shape_function_matrix(const Eigen::Vector3d& natural_coord) const = 0;
    virtual LocalDenseMatrix element_stiffness(const FEM& fem, double nu, double thickness) const = 0;
    virtual LocalDenseMatrix element_mass(const FEM& fem, double thickness) const = 0;
    virtual std::vector<double> signed_detJ_values(const FEM& fem,
                                                   const std::vector<int>& node_order = {}) const;
    virtual std::vector<ElementOrientationPermutation> orientation_check_permutations() const;

    std::array<double, 3> center(const FEM& fem) const;
    std::size_t node_count() const { return node_ids.size(); }
    std::size_t dof_count() const { return dof_ids.size(); }

protected:
    static Eigen::Vector3d node_vec(const FEM& fem, int node_id);
    static double triangle_area(const Eigen::Vector3d& a,
                                const Eigen::Vector3d& b,
                                const Eigen::Vector3d& c);
    static double tetra_volume(const Eigen::Vector3d& a,
                               const Eigen::Vector3d& b,
                               const Eigen::Vector3d& c,
                               const Eigen::Vector3d& d);
    static double det3(const std::array<double, 9>& a);
    static std::array<double, 9> inv3(const std::array<double, 9>& a);
    static void add_btdb(LocalDenseMatrix& ke,
                         const std::vector<double>& B,
                         const std::vector<double>& D,
                         int strain_dim,
                         int ndof,
                         double weight);
};

struct Q4Element : ElementBase {
    using ElementBase::ElementBase;
    std::string name() const override { return "Q4"; }
    double volume(const FEM& fem) const override;
    double mesh_size(const FEM& fem) const override;
    Eigen::MatrixXd shape_function_matrix(const Eigen::Vector3d& natural_coord) const override;
    LocalDenseMatrix element_stiffness(const FEM& fem, double nu, double thickness) const override;
    LocalDenseMatrix element_mass(const FEM& fem, double thickness) const override;
    std::vector<double> signed_detJ_values(const FEM& fem, const std::vector<int>& node_order = {}) const override;
    std::vector<ElementOrientationPermutation> orientation_check_permutations() const override;
};

struct Tri3Element : ElementBase {
    using ElementBase::ElementBase;
    std::string name() const override { return "TRI3"; }
    double volume(const FEM& fem) const override;
    double mesh_size(const FEM& fem) const override;
    Eigen::MatrixXd shape_function_matrix(const Eigen::Vector3d& natural_coord) const override;
    LocalDenseMatrix element_stiffness(const FEM& fem, double nu, double thickness) const override;
    LocalDenseMatrix element_mass(const FEM& fem, double thickness) const override;
};

struct H8Element : ElementBase {
    using ElementBase::ElementBase;
    std::string name() const override { return "H8"; }
    double volume(const FEM& fem) const override;
    double mesh_size(const FEM& fem) const override;
    Eigen::MatrixXd shape_function_matrix(const Eigen::Vector3d& natural_coord) const override;
    LocalDenseMatrix element_stiffness(const FEM& fem, double nu, double thickness) const override;
    LocalDenseMatrix element_mass(const FEM& fem, double thickness) const override;
    std::vector<double> signed_detJ_values(const FEM& fem, const std::vector<int>& node_order = {}) const override;
    std::vector<ElementOrientationPermutation> orientation_check_permutations() const override;
};

struct T4Element : ElementBase {
    using ElementBase::ElementBase;
    std::string name() const override { return "T4"; }
    double volume(const FEM& fem) const override;
    double mesh_size(const FEM& fem) const override;
    Eigen::MatrixXd shape_function_matrix(const Eigen::Vector3d& natural_coord) const override;
    LocalDenseMatrix element_stiffness(const FEM& fem, double nu, double thickness) const override;
    LocalDenseMatrix element_mass(const FEM& fem, double thickness) const override;
};

struct P6Element : ElementBase {
    using ElementBase::ElementBase;
    std::string name() const override { return "P6"; }
    double volume(const FEM& fem) const override;
    double mesh_size(const FEM& fem) const override;
    Eigen::MatrixXd shape_function_matrix(const Eigen::Vector3d& natural_coord) const override;
    LocalDenseMatrix element_stiffness(const FEM& fem, double nu, double thickness) const override;
    LocalDenseMatrix element_mass(const FEM& fem, double thickness) const override;
};

std::shared_ptr<ElementBase> make_element_object(const FEM& fem, int elem_local_id);
