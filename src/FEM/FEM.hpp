#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <petscksp.h>
#include <Eigen/Dense>

#include "Region/Region.hpp"
#include "myjson.hpp"

struct Geometry {
    double Lx = 1200.0;
    double Ly = 400.0;
    double Lz = 120.0;
    int nelx = 120;
    int nely = 40;
    int nelz = 1;

    Geometry() = default;

    explicit Geometry(const myjson::json& j) {
        Lx = myjson::double_value(j, "Lx", Lx);
        Ly = myjson::double_value(j, "Ly", Ly);
        Lz = myjson::double_value(j, "Lz", Lz);
        nelx = myjson::int_value(j, "nelx", nelx);
        nely = myjson::int_value(j, "nely", nely);
        nelz = myjson::int_value(j, "nelz", nelz);
    }
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
    Eigen::VectorXd f_int;
};

struct FEM {
    std::string fem_type = "plane_stress";
    double thickness = 1.0;

    std::vector<std::array<double, 3>> Node;
    std::vector<std::vector<int>> Element;
    std::vector<std::shared_ptr<ElementBase>> ElementObjects;
    std::vector<std::shared_ptr<Region>> Regions;
    std::vector<ElementType> ElementTypes;
    std::vector<std::vector<PetscInt>> ElementDofs;
    std::vector<std::array<double, 3>> elem_center;
    std::vector<double> volume;
    std::vector<double> mesh_size;
    std::vector<int> nonpositive_detJ_element_ids;
    std::vector<int> node_ids;
    std::vector<int> element_ids;
    std::unordered_map<int, int> node_id_to_local;
    std::unordered_map<int, int> element_id_to_local;
    int dim = 2;
    bool uniform_element_matrices = true;
    int NNode = 0;
    int NElem = 0;
    int ndof = 0;
    std::vector<int> AllDofs;
    int nelx = 0;
    int nely = 0;
    int nelz = 0;

    virtual ~FEM() = default;
    void InitializeRegions(const myjson::json& regions_json);
    void InitializeMaterials(const myjson::json& materials_json);
    void precompute_elements(const std::string& element_type);
    void debug_check_element_orientation() const;
};

struct StructuredMesh : FEM {
    Geometry geometry;

    explicit StructuredMesh(const myjson::json& fem_json) : geometry(fem_json.at("Geom")) {
        fem_type = myjson::lower_ascii(fem_json.at("fem_type").get<std::string>());
        thickness = myjson::double_value(fem_json, "thickness", 1.0);
        uniform_element_matrices = myjson::bool_value(fem_json, "uniform_element_matrices", true);
        build_from_geometry(geometry);
        precompute_elements(fem_json.at("element_type").get<std::string>());
    }

    void build_from_geometry(const Geometry& geom);
};

struct UnstructuredMesh : FEM {
    std::unordered_map<int, HMSet> hm_sets_by_id;
    std::unordered_map<std::string, HMSet> hm_sets_by_name;

    explicit UnstructuredMesh(const myjson::json& fem_json) {
        fem_type = myjson::lower_ascii(fem_json.at("fem_type").get<std::string>());
        thickness = myjson::double_value(fem_json, "thickness", 1.0);
        uniform_element_matrices = myjson::bool_value(fem_json, "uniform_element_matrices", false);
        const std::string fem_file = myjson::string_value(fem_json, "fem_file", "");
        if (fem_file.empty())
            throw std::runtime_error("FEM.fem_file is required for unstructured/hypermesh mesh");
        read_hypermesh_fem(fem_file);
        precompute_elements(fem_json.at("element_type").get<std::string>());

        if (uniform_element_matrices) {
            std::cout << "====================NOTE===================\n";
            std::cout << "UnstructuredMesh but uniform element\n";
        }
    }

    const HMSet& hmset_by_name(const std::string& name) const;
    void read_hypermesh_fem(const std::string& fem_file);
};

inline std::unique_ptr<FEM> InitializeFEM(const myjson::json& fem_json) {
    const std::string mesh_type = myjson::lower_ascii(fem_json.at("mesh_type").get<std::string>());
    if (mesh_type == "hypermesh" || mesh_type == "unstructured") {
        return std::make_unique<UnstructuredMesh>(fem_json);
    }
    if (mesh_type == "structured") {
        return std::make_unique<StructuredMesh>(fem_json);
    }
    throw std::runtime_error("Unknown FEM.mesh_type: " + mesh_type);
}
