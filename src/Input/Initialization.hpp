#pragma once

#include "MatlabLikeData.hpp"
#include "Output/OutputData.hpp"

#include <memory>
#include <string>
#include <vector>

struct InputConfig {
    Geometry geom;

    std::string fem_type = "plane_stress";
    std::string mesh_source = "structured";
    std::string fem_file;
    int fem_uniform = 1;

    std::string region_source = "range";
    std::string bc_source = "range";
    std::string design_hmset_name = "design_val";
    std::string nondesign_hmset_name = "nondesign";
    std::string spc_hmset_name = "SPC";
    std::string load_hmset_name = "load";

    int nstep = 100;
    double tmax = 0.008;
    double x_initial = 0.5;
    double force_amp = 3333.0;
    double p_rho = 0.0;

    std::string solve_mode = "ROM";
    std::string linear_backend = "petsc";
    std::string method_name = "MyStrategy";
    std::string fom_solver_type = "direct";
    std::string pod_backend = "snapshot_mkl";
    std::string online_backend = "incremental_mkl";
    std::string sens_type = "AVM";
    double rom_tol = 0.02;
    int rom_rank1 = 10;
    int rom_max_rank = 20;
    int rom_online = 1;

    int run_optimization = 0;
    int max_iter = 5;
    double volume_limit = 0.4;
    double move = 0.1;
    double debug_fd_epsilon = 1e-6;
    int debug_fd_max_checks = 20;

    double filter_rmin = 3.0;
    double filter_beta = 1.0;
    int filter_beta_loop = 50;
    double filter_beta_max = 8.0;
    double filter_T = 0.5;

    int save_density_each_iter = 1;
    int save_density_interval = 1;
    int save_vtk_response_interval = 1;
    int save_vtk_opt_interval = 1;
    int factor_num_threads = 1;
    int sens_element_threads = 1;

    std::string out_dir = "case_study/Debug";
    std::string density_dir;

    double void_E = 1e-3;
    double void_rho = 1e-12;
    double solid_E = 210000.0;
    double solid_rho = 6e-9;
    double nu = 0.3;
    double thickness = 1.0;
    double interp_p_E_elastic = 8.0;

    std::array<int, 3> dirich_flag{1, 1, 1};
    std::array<int, 3> neum_flag{0, 1, 0};
    double dirich_x_min = 0.0;
    double dirich_x_max = 2.0;
    double dirich_y_min = 0.0;
    double dirich_y_max = 400.0;
    double dirich_z_min = -1e3;
    double dirich_z_max = 1e3;
    double dirich_amp = 0.0;
    std::string dirich_time_type = "constant";

    double load_x_min_offset = 5.0;
    double load_x_min = 1195.0;
    double load_x_max = 1200.0;
    double load_y_min = 190.0;
    double load_y_max = 210.0;
    double load_z_min = -1e3;
    double load_z_max = 1e3;
    std::string load_time_type = "sin";
    double load_nCycle = 1.0;
};

InputConfig InitializeInputConfig();
std::unique_ptr<FEM> InitializeFEM(const InputConfig& input);
OutputData InitializeOutput(const InputConfig& input);
std::vector<Material> InitializeMaterial(const InputConfig& input);
std::vector<Region> InitializeRegions(const FEM& fem, const InputConfig& input);
std::vector<BoundaryCondition> InitializeDirichlet(const FEM& fem, const InputConfig& input);
std::vector<BoundaryCondition> InitializeNeumann(const FEM& fem, const InputConfig& input);
std::unique_ptr<DynamicLinearElastic> InitializeAnalysis(const InputConfig& input,
                                                        const std::vector<Material>& materials,
                                                        const std::vector<Region>& regions,
                                                        const std::vector<BoundaryCondition>& dirich,
                                                        const std::vector<BoundaryCondition>& neum);
Optimization InitializeOptimization(const FEM& fem,
                                    const DynamicLinearElastic& analysis,
                                    const std::vector<Material>& materials,
                                    const InputConfig& input);

std::string lower_ascii(std::string s);
