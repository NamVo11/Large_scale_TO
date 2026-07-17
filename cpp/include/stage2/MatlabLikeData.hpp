#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <petscmat.h>

namespace stage2 {

class ROMBase;

struct Geometry {
    double Lx = 1200.0;
    double Ly = 400.0;
    double Lz = 120.0;
    int nelx = 120;
    int nely = 40;
    int nelz = 1;
};

struct FEM {
    std::string type = "plane_stress";
    double nu = 0.3;
    double Thickness = 1.0;

    std::vector<std::array<double, 3>> Node;
    std::vector<std::vector<int>> Element;
    std::vector<std::vector<PetscInt>> ElementDofs;
    std::vector<std::array<double, 3>> elem_center;
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
};

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
    std::vector<int> mat_ids;
    std::string status = "design_val";
    std::string range_type = "all";
    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;

    std::vector<int> elem_ids;
    int nElem = 0;
    MatParas mat_paras;
    Interp interp;
};

struct BoundaryCondition {
    int id = 0;
    std::string kind;
    std::array<int, 3> flag{0, 0, 0};
    std::vector<std::array<double, 2>> x_range;
    std::vector<std::array<double, 2>> y_range;
    std::vector<std::array<double, 2>> z_range;
    double amp = 0.0;
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

struct AnalysisResult {
    std::vector<std::vector<double>> U;
    std::vector<std::vector<double>> V;
    std::vector<std::vector<double>> A;
    std::vector<std::vector<double>> F_ext;
    std::vector<int> SPCdofs;
    std::vector<int> free;
    std::vector<int> ksp_iterations;
    std::vector<double> ksp_residuals;
    Mat K_inf = nullptr;
    Mat M = nullptr;
    Mat K_eff = nullptr;
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

    std::shared_ptr<ROMBase> ROM;
    std::string solver_name = "dynamic_elastic_linear";
    std::vector<Region> regions;
    std::vector<BoundaryCondition> Dirich;
    std::vector<BoundaryCondition> Neum;
    AnalysisResult results;
};

struct OutputData {
    double saved_Iter = 500.0;
    std::string outDir_vtk;
    std::string outDir_mat;
    std::string outDir_mat_FOM;
    std::string densityDir;
};

struct Filtering {
    double rmin = 3.0;
    double beta = 1.0;
    int beta_loop = 50;
    double beta_max = 8.0;
    double T = 0.5;
};

struct Algorithm {
    std::string type = "MMA";
    double move = 0.1;
};

struct ObjectiveTerm {
    std::string function_name;
    double weight = 1.0;
    double value_initial = 0.0;
    double value = 0.0;
    std::vector<std::vector<double>> dfdu_all;
    std::vector<std::vector<double>> dfdv_all;
    std::vector<std::vector<double>> dfda_all;
    std::vector<std::vector<double>> dfdFext_all;
    std::vector<std::vector<double>> df_dx;
};

struct ObjectiveGroup {
    int analysis_id = 1;
    std::string sens_type;
    std::string sens_function;
    std::string type;
    std::shared_ptr<ROMBase> ROM;
    std::vector<ObjectiveTerm> terms;
    double value = 0.0;
    std::vector<std::vector<double>> d_explicit_dfdu_all;
    std::vector<std::vector<double>> d_explicit_dfdv_all;
    std::vector<std::vector<double>> d_explicit_dfda_all;
    std::vector<std::vector<double>> d_explicit_dfdFext_all;
    std::vector<std::vector<double>> d_explicit_df_dx;
};

struct Constraint {
    int analysis_id = 1;
    std::string function_name;
    double limit = 0.0;
    std::string sens_type;
    std::string sens_function;
    std::string type;
    int mat_id = 0;
    double value_initial = 0.0;
    double value = 0.0;
    std::vector<std::vector<double>> d_explicit_dfdu_all;
    std::vector<std::vector<double>> d_explicit_dfdv_all;
    std::vector<std::vector<double>> d_explicit_dfda_all;
    std::vector<std::vector<double>> d_explicit_dfdFext_all;
    std::vector<std::vector<double>> d_explicit_df_dx;
};

struct OptimizationHistory {
    std::vector<int> Iter;
    std::vector<double> objective;
    std::vector<double> constraint;
    std::vector<double> change;
    std::vector<double> rms_change;
    std::vector<double> forward_time;
    std::vector<double> objective_time;
    std::vector<double> constraint_time;
    std::vector<double> objective_sensitivity_time;
    std::vector<double> constraint_sensitivity_time;
    std::vector<double> mma_time;
    std::vector<double> iteration_time;
};

struct Optimization {
    std::vector<std::vector<double>> x0;
    std::vector<std::vector<double>> x;
    std::vector<int> x_label;
    int MaxIter = 300;
    double Tol = 0.0;
    bool plot_during_optimization = false;
    Filtering filtering;
    Algorithm algorithm;
    double debug_fd_epsilon = 1e-6;
    int debug_fd_max_checks = 20;
    bool save_density_each_iter = true;
    int save_density_interval = 1;
    std::vector<ObjectiveGroup> objective_groups;
    std::vector<Constraint> constraints;
    OptimizationHistory history;
};

double RAMP(const std::vector<double>& para, double x, double p);

FEM Mesh(const Geometry& geom, FEM fem);
void PrecomputeFEM(FEM& fem);
void assign_elements_to_regions(std::vector<Region>& regions, const FEM& fem);
void assign_nodes_and_dofs_to_bcs(std::vector<BoundaryCondition>& BCs, const FEM& fem);
void init_mat(const std::vector<Material>& Mat, Analysis& A);
void init_interp(const Interp& interp, Analysis& A);
void attach_bcs_to_analysis_by_id(Analysis& A,
                                  const std::vector<BoundaryCondition>& Dirich,
                                  const std::vector<BoundaryCondition>& Neum);
std::vector<std::vector<double>> init_design_variables(const FEM& fem,
                                                       const std::vector<Region>& regions,
                                                       const std::vector<Material>& Mat,
                                                       double x0,
                                                       std::vector<int>& x_label);

double bc_value(const BoundaryCondition& BC, const Analysis& A, int istep);

} // namespace stage2
