#pragma once

#include "FEM/FEM.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <petscksp.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class ROMBase;
class FactorizedSolver;
struct Optimization;
struct OutputData;

struct Material {
    int id = 0;
    std::string type = "elastic";
    double E_inf = 0.0;
    double E = 0.0;
    double nu = 0.3;
    double rho = 0.0;
};

struct MatParas {
    int nMat = 0;
    std::vector<int> mat_ids;
    std::vector<int> val_ids;
    std::vector<double> E_inf;
    std::vector<double> nu;
    std::vector<double> rho;
};

using InterpFunction = std::function<double(const std::vector<double>& para, double x)>;

struct Interp {
    double p_E_elastic = 8.0;
    double p_rho = 0.0;
    InterpFunction elastic_interp;
    InterpFunction mass_density_interp;
};

struct Region {
    int id = 0;
    std::string name;
    std::string source = "range";
    std::vector<int> mat_ids;
    std::string status = "design_val";
    std::string range_type = "all";
    std::string hmset_name;
    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;

    std::vector<int> elem_ids;
    int nElem = 0;
    MatParas mat_paras;
    Interp interp;
};

struct RangeRegion : Region {
    RangeRegion(const FEM& fem, Region region_in = Region{})
        : Region(std::move(region_in)) {
        std::vector<Region> tmp{*this};
        assign_elements_to_regions(tmp, fem);
        static_cast<Region&>(*this) = std::move(tmp.front());
    }

    static void assign_elements_to_regions(std::vector<Region>& regions, const FEM& fem);
};

struct HMSetRegion : Region {
    HMSetRegion(const FEM& fem, const std::string& set_name, Region region_in = Region{})
        : Region(std::move(region_in)) {
        hmset_name = set_name;
        range_type = "hmset";
        std::vector<Region> tmp{*this};
        assign_elements_to_regions(tmp, fem);
        static_cast<Region&>(*this) = std::move(tmp.front());
    }

    static void assign_elements_to_regions(std::vector<Region>& regions, const FEM& fem);
};

struct BoundaryCondition {
    int id = 0;
    std::string name;
    std::string source = "range";
    std::string kind;
    std::string hmset_name;
    std::array<int, 3> flag{0, 0, 0};
    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;
    double amp = 0.0;
    double base_value = 0.0;
    std::string time_type = "constant";
    double nCycle = 0.0;
    double power = 0.0;
    double phase = 0.0;

    std::vector<int> node_ids;
    std::vector<int> dof_ids;
    int nNode = 0;
    int nDof = 0;
    std::function<double(int)> fun_value;
};

struct RangeBoundaryCondition : BoundaryCondition {
    RangeBoundaryCondition(const FEM& fem, BoundaryCondition bc_in = BoundaryCondition{})
        : BoundaryCondition(std::move(bc_in)) {
        std::vector<BoundaryCondition> tmp{*this};
        assign_nodes_and_dofs_to_bcs(tmp, fem);
        static_cast<BoundaryCondition&>(*this) = std::move(tmp.front());
    }

    static void assign_nodes_and_dofs_to_bcs(std::vector<BoundaryCondition>& BCs, const FEM& fem);
};

struct HMSetBoundaryCondition : BoundaryCondition 
{
    HMSetBoundaryCondition(const FEM& fem, const std::string& set_name, BoundaryCondition bc_in = BoundaryCondition{})
        : BoundaryCondition(std::move(bc_in))
    {
        hmset_name = set_name;
        std::vector<BoundaryCondition> tmp{*this};
        assign_nodes_and_dofs_to_bcs(tmp, fem);
        static_cast<BoundaryCondition&>(*this) = std::move(tmp.front());
    }

    static void assign_nodes_and_dofs_to_bcs(std::vector<BoundaryCondition>& BCs, const FEM& fem);
};

struct AnalysisResult {
    Eigen::MatrixXd U;
    Eigen::MatrixXd V;
    Eigen::MatrixXd A;
    Eigen::MatrixXd F_ext;
    std::vector<int> SPCdofs;
    std::vector<int> free;
    std::vector<int> ksp_iterations;
    std::vector<double> ksp_residuals;
    Mat K_inf = nullptr;
    Mat M = nullptr;
    Mat K_eff = nullptr;
    KSP factorized_K_initial = nullptr;
    KSP factorized_K_eff = nullptr;
    Eigen::SparseMatrix<double> K_inf_eigen;
    Eigen::SparseMatrix<double> M_eigen;
    Eigen::SparseMatrix<double> K_eff_eigen;
    std::shared_ptr<FactorizedSolver> factorized_K_initial_eigen;
    std::shared_ptr<FactorizedSolver> factorized_K_eff_eigen;
    double assembly_time = 0.0;
    double solve_time = 0.0;
};

struct Analysis {
    int id = 1;
    std::string mat_type = "elastic";
    std::vector<int> Dirich_id;
    std::vector<int> Neum_id;
    std::string type = "dynamic";
    std::string method = "linear";
    double Tmax = 0.12;
    int Nstep = 300;
    double dt = Tmax / static_cast<double>(Nstep);
    bool save_factorized_K_initial = true;

    std::shared_ptr<ROMBase> ROM;
    std::string solver_name = "dynamic_elastic_linear";
    std::vector<Region> regions;
    std::vector<BoundaryCondition> Dirich;
    std::vector<BoundaryCondition> Neum;
    AnalysisResult results;

    virtual ~Analysis() = default;
    virtual void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0);
    std::vector<int> SPC() const {
        std::vector<int> dofs;
        for (const auto& bc : Dirich) dofs.insert(dofs.end(), bc.dof_ids.begin(), bc.dof_ids.end());
        std::sort(dofs.begin(), dofs.end());
        dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
        return dofs;
    }

    std::vector<int> free_dofs(int ndof) const {
        const std::vector<int> fixed = SPC();
        std::vector<char> is_fixed(static_cast<std::size_t>(ndof), 0);
        for (int d : fixed) {
            if (d >= 0 && d < ndof) is_fixed[static_cast<std::size_t>(d)] = 1;
        }
        std::vector<int> free;
        free.reserve(static_cast<std::size_t>(std::max(0, ndof - static_cast<int>(fixed.size()))));
        for (int d = 0; d < ndof; ++d) {
            if (!is_fixed[static_cast<std::size_t>(d)]) free.push_back(d);
        }
        return free;
    }
};

struct DynamicLinearElastic : Analysis {
    DynamicLinearElastic() {
        type = "dynamic";
        method = "linear";
        solver_name = "dynamic_elastic_linear";
    }

    DynamicLinearElastic& operator=(const Analysis& other) {
        Analysis::operator=(other);
        return *this;
    }
};

struct PetscDynamicLinearElastic : DynamicLinearElastic {
    PetscDynamicLinearElastic() = default;
    using DynamicLinearElastic::operator=;
    void assemble_K_M(const FEM& fem, const Eigen::MatrixXd& y_elem, Mat K, Mat M) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};

struct EigenDynamicLinearElastic : DynamicLinearElastic {
    EigenDynamicLinearElastic() = default;
    using DynamicLinearElastic::operator=;
    void assemble_K_M(const FEM& fem,
                      const Eigen::MatrixXd& y_elem,
                      Eigen::SparseMatrix<double>& K,
                      Eigen::SparseMatrix<double>& M) const;
    void solver(const FEM& fem, const Eigen::MatrixXd& y_elem, int Iter = 0) override;
};
