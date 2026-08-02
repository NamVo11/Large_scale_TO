#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <petscksp.h>
#include <Eigen/Dense>

struct Geometry {
    double Lx = 1200.0;
    double Ly = 400.0;
    double Lz = 120.0;
    int nelx = 120;
    int nely = 40;
    int nelz = 1;
};

struct ElementBase;

enum class ElementType {
    Unknown,
    Q4,
    TRI3,
    H8,
    T4,
    P6
};

std::string element_type_name(ElementType type);

struct HMSet {
    int id = 0;
    int kind = 0; // HyperMesh convention observed here: 1=node set, 2=element set.
    std::string name;
    std::vector<int> raw_ids;
    std::vector<int> local_ids;
};

using LocalDenseMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct ElementMatrix {
    int ndof = 0;
    LocalDenseMatrix ke;
    LocalDenseMatrix me;
};

struct FEM {
    std::string type = "plane_stress";
    double nu = 0.3;
    double Thickness = 1.0;

    std::vector<std::array<double, 3>> Node;
    std::vector<std::vector<int>> Element;
    std::vector<std::shared_ptr<ElementBase>> ElementObjects;
    std::vector<ElementType> ElementTypes;
    std::vector<std::vector<PetscInt>> ElementDofs;
    std::vector<std::array<double, 3>> elem_center;
    std::vector<double> volume;
    std::vector<double> mesh_size;
    std::vector<ElementMatrix> ElementMatrixCache;
    std::vector<int> nonpositive_detJ_element_ids;
    std::vector<int> node_ids;
    std::vector<int> element_ids;
    std::unordered_map<int, int> node_id_to_local;
    std::unordered_map<int, int> element_id_to_local;
    std::unordered_map<int, HMSet> hm_sets_by_id;
    std::unordered_map<std::string, HMSet> hm_sets_by_name;
    int dim = 2;
    std::string mesh_type = "2D_Q4";
    bool uniform_element_matrices = true;
    int NNode = 0;
    int NElem = 0;
    int ndof = 0;
    std::vector<int> AllDofs;
    int nelx = 0;
    int nely = 0;
    int nelz = 0;

    void build_element_objects();
    ElementMatrix compute_element_matrices(int elem_id, double nu, double thickness) const;
    void precompute_element_geometry();
    void precompute_element_dofs();
    void debug_check_element_orientation() const;
    void precompute_element_data() {
        precompute_element_geometry();
        precompute_element_dofs();
    }
};

struct StructuredMesh : FEM {
    StructuredMesh(const Geometry& geom,
                   const std::string& fem_type = "plane_stress",
                   double nu_in = 0.3,
                   double thickness_in = 1.0,
                   bool uniform_element_matrices_in = true) {
        type = fem_type;
        nu = nu_in;
        Thickness = thickness_in;
        uniform_element_matrices = uniform_element_matrices_in;
        build_from_geometry(geom);
        precompute_element_data();
    }

    void build_from_geometry(const Geometry& geom);
};

struct UnstructuredMesh : FEM {
    UnstructuredMesh(const std::string& fem_file,
                     const std::string& fem_type = "3d_brick",
                     double nu_in = 0.3,
                     double thickness_in = 1.0,
                     bool uniform_element_matrices_in = false) {
        type = fem_type;
        nu = nu_in;
        Thickness = thickness_in;
        uniform_element_matrices = uniform_element_matrices_in;
        read_hypermesh_fem(fem_file);
        precompute_element_data();
    }

    void read_hypermesh_fem(const std::string& fem_file);
};
