#include "Input/Initialization.hpp"

#include "Optimization/Optimization.hpp"
#include "ROM/ROM.hpp"

#include <petscksp.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace {

template <typename T>
void get_int_option(const char* name, T& value) {
    PetscInt v = static_cast<PetscInt>(value);
    PetscBool set = PETSC_FALSE;
    PetscOptionsGetInt(nullptr, nullptr, name, &v, &set);
    if (set) value = static_cast<T>(v);
}

void get_real_option(const char* name, double& value) {
    PetscReal v = value;
    PetscBool set = PETSC_FALSE;
    PetscOptionsGetReal(nullptr, nullptr, name, &v, &set);
    if (set) value = static_cast<double>(v);
}

void get_string_option(const char* name, std::string& value) {
    char buf[PETSC_MAX_PATH_LEN] = {0};
    PetscBool set = PETSC_FALSE;
    PetscOptionsGetString(nullptr, nullptr, name, buf, sizeof(buf), &set);
    if (set) value = buf;
}

bool option_is_set(const char* name) {
    PetscBool set = PETSC_FALSE;
    PetscOptionsHasName(nullptr, nullptr, name, &set);
    return set == PETSC_TRUE;
}

void set_default_option_if_missing(const char* name, const char* value) {
    if (!option_is_set(name)) PetscOptionsSetValue(nullptr, name, value);
}

bool is_rom_mode(const InputConfig& input) {
    const std::string mode = lower_ascii(input.solve_mode);
    return mode == "rom";
}

std::shared_ptr<ROMBase> InitializeROM(const InputConfig& input) {
    if (!is_rom_mode(input)) return make_fom_solver(input.fom_solver_type, input.linear_backend);
    return make_dynamic_ROM(input.method_name,
                            input.fom_solver_type,
                            input.rom_tol,
                            input.rom_rank1,
                            input.rom_max_rank,
                            input.rom_online != 0,
                            input.linear_backend);
}

BoundaryCondition make_base_dirichlet(const InputConfig& input) {
    BoundaryCondition bc;
    bc.id = 1;
    bc.kind = "fixed_disp";
    bc.flag = input.dirich_flag;
    bc.x_range = {{{input.dirich_x_min, input.dirich_x_max}}};
    bc.y_range = {{{input.dirich_y_min, input.dirich_y_max}}};
    bc.z_range = {{{input.dirich_z_min, input.dirich_z_max}}};
    bc.amp = input.dirich_amp;
    bc.time_type = input.dirich_time_type;
    return bc;
}

BoundaryCondition make_base_neumann(const InputConfig& input) {
    BoundaryCondition bc;
    bc.id = 1;
    bc.kind = "nodal_force";
    bc.flag = input.neum_flag;
    bc.x_range = {{{input.load_x_min, input.load_x_max}}};
    bc.y_range = {{{input.load_y_min, input.load_y_max}}};
    bc.z_range = {{{input.load_z_min, input.load_z_max}}};
    bc.amp = input.force_amp;
    bc.time_type = input.load_time_type;
    bc.nCycle = input.load_nCycle;
    return bc;
}

} // namespace

std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

InputConfig InitializeInputConfig() {
    InputConfig input;
    input.geom.Lx = 1200.0;
    input.geom.Ly = 400.0;
    input.geom.Lz = 400.0;
    input.geom.nelx = 600;
    input.geom.nely = 200;
    input.geom.nelz = 1;

    set_default_option_if_missing("-ksp_type", "preonly");
    set_default_option_if_missing("-pc_type", "cholesky");
    set_default_option_if_missing("-pc_factor_mat_solver_type", "mkl_pardiso");

    get_int_option("-nelx", input.geom.nelx);
    get_int_option("-nely", input.geom.nely);
    get_int_option("-nelz", input.geom.nelz);
    get_real_option("-Lx", input.geom.Lx);
    get_real_option("-Ly", input.geom.Ly);
    get_real_option("-Lz", input.geom.Lz);
    get_int_option("-nstep", input.nstep);
    get_real_option("-tmax", input.tmax);
    get_real_option("-x0", input.x_initial);
    get_real_option("-force_amp", input.force_amp);
    get_real_option("-p_rho", input.p_rho);
    get_int_option("-run_optimization", input.run_optimization);
    get_int_option("-max_iter", input.max_iter);
    get_real_option("-volume_limit", input.volume_limit);
    get_real_option("-move", input.move);
    get_real_option("-debug_fd_epsilon", input.debug_fd_epsilon);
    get_int_option("-debug_fd_max_checks", input.debug_fd_max_checks);
    get_int_option("-save_density_each_iter", input.save_density_each_iter);
    get_int_option("-save_density_interval", input.save_density_interval);
    get_int_option("-density_plot_interval", input.save_density_interval);
    get_int_option("-save_vtk_response_interval", input.save_vtk_response_interval);
    get_int_option("-save_vtk_opt_interval", input.save_vtk_opt_interval);
    get_int_option("-fem_uniform", input.fem_uniform);
    get_int_option("-factor_num_threads", input.factor_num_threads);
    get_int_option("-sens_element_threads", input.sens_element_threads);
    get_real_option("-rom_tol", input.rom_tol);
    get_int_option("-rank1", input.rom_rank1);
    get_int_option("-max_rank", input.rom_max_rank);
    get_int_option("-online", input.rom_online);
    get_real_option("-filter_rmin", input.filter_rmin);
    get_real_option("-filter_beta", input.filter_beta);
    get_int_option("-filter_beta_loop", input.filter_beta_loop);
    get_real_option("-filter_beta_max", input.filter_beta_max);
    get_real_option("-filter_T", input.filter_T);
    get_real_option("-void_E", input.void_E);
    get_real_option("-void_rho", input.void_rho);
    get_real_option("-solid_E", input.solid_E);
    get_real_option("-solid_rho", input.solid_rho);
    get_real_option("-nu", input.nu);
    get_real_option("-thickness", input.thickness);
    get_real_option("-p_E_elastic", input.interp_p_E_elastic);
    get_int_option("-dirich_flag_x", input.dirich_flag[0]);
    get_int_option("-dirich_flag_y", input.dirich_flag[1]);
    get_int_option("-dirich_flag_z", input.dirich_flag[2]);
    get_int_option("-neum_flag_x", input.neum_flag[0]);
    get_int_option("-neum_flag_y", input.neum_flag[1]);
    get_int_option("-neum_flag_z", input.neum_flag[2]);
    get_real_option("-dirich_x_min", input.dirich_x_min);
    get_real_option("-dirich_x_max", input.dirich_x_max);
    get_real_option("-dirich_y_min", input.dirich_y_min);
    get_real_option("-dirich_y_max", input.dirich_y_max);
    get_real_option("-dirich_z_min", input.dirich_z_min);
    get_real_option("-dirich_z_max", input.dirich_z_max);
    get_real_option("-dirich_amp", input.dirich_amp);
    get_real_option("-load_x_min_offset", input.load_x_min_offset);
    input.load_x_min = input.geom.Lx - input.load_x_min_offset;
    input.load_x_max = input.geom.Lx;
    get_real_option("-load_x_min", input.load_x_min);
    get_real_option("-load_x_max", input.load_x_max);
    get_real_option("-load_y_min", input.load_y_min);
    get_real_option("-load_y_max", input.load_y_max);
    get_real_option("-load_z_min", input.load_z_min);
    get_real_option("-load_z_max", input.load_z_max);
    get_real_option("-load_nCycle", input.load_nCycle);

    get_string_option("-fem_type", input.fem_type);
    get_string_option("-linear_backend", input.linear_backend);
    get_string_option("-solve_mode", input.solve_mode);
    get_string_option("-method_name", input.method_name);
    get_string_option("-fom_solver_type", input.fom_solver_type);
    get_string_option("-pod_backend", input.pod_backend);
    get_string_option("-online_backend", input.online_backend);
    get_string_option("-sens_type", input.sens_type);
    get_string_option("-stage2_out", input.out_dir);
    get_string_option("-density_out", input.density_dir);
    get_string_option("-mesh_source", input.mesh_source);
    get_string_option("-fem_file", input.fem_file);
    get_string_option("-region_source", input.region_source);
    get_string_option("-bc_source", input.bc_source);
    get_string_option("-design_hmset", input.design_hmset_name);
    get_string_option("-nondesign_hmset", input.nondesign_hmset_name);
    get_string_option("-spc_hmset", input.spc_hmset_name);
    get_string_option("-load_hmset", input.load_hmset_name);
    get_string_option("-dirich_time_type", input.dirich_time_type);
    get_string_option("-load_time_type", input.load_time_type);

    setenv("FACTOR_NUM_THREADS", std::to_string(std::max(1, input.factor_num_threads)).c_str(), 1);
    setenv("SENS_ELEMENT_THREADS", std::to_string(std::max(1, input.sens_element_threads)).c_str(), 1);
    if (!input.pod_backend.empty()) setenv("STAGE2_POD_BACKEND", input.pod_backend.c_str(), 1);
    if (!input.online_backend.empty()) setenv("STAGE2_ONLINE_BACKEND", input.online_backend.c_str(), 1);

    return input;
}

std::unique_ptr<FEM> InitializeFEM(const InputConfig& input) {
    const std::string mesh_source_key = lower_ascii(input.mesh_source);
    if (mesh_source_key == "hypermesh" || mesh_source_key == "unstructured" || !input.fem_file.empty()) {
        if (input.fem_file.empty()) throw std::runtime_error("-fem_file is required for unstructured/hypermesh mesh_source");
        return std::make_unique<UnstructuredMesh>(input.fem_file,
                                                  input.fem_type,
                                                  input.nu,
                                                  input.thickness,
                                                  input.fem_uniform != 0);
    }
    return std::make_unique<StructuredMesh>(input.geom,
                                            input.fem_type,
                                            input.nu,
                                            input.thickness,
                                            input.fem_uniform != 0);
}

OutputData InitializeOutput(const InputConfig& input) {
    OutputData output;
    output.outDir_mat = input.out_dir;
    output.outDir_mat_FOM = input.out_dir;
    const std::filesystem::path case_dir = std::filesystem::path(input.out_dir).parent_path();
    output.outDir_vtk = (case_dir / "vtk").string();
    output.densityDir = input.density_dir.empty() ? input.out_dir + "/density" : input.density_dir;
    output.figureDir = (case_dir / "figure").string();
    output.save_density_each_iter = input.save_density_each_iter;
    output.save_density_interval = std::max(1, input.save_density_interval);
    output.save_vtk_response_interval = std::max(1, input.save_vtk_response_interval);
    output.save_vtk_opt_interval = std::max(1, input.save_vtk_opt_interval);
    return output;
}

std::vector<Material> InitializeMaterial(const InputConfig& input) {
    return {
        Material{1, "elastic", input.void_E, 0.0, input.nu, input.void_rho},
        Material{2, "elastic", input.solid_E, 0.0, input.nu, input.solid_rho},
    };
}

std::vector<Region> InitializeRegions(const FEM& fem, const InputConfig& input) {
    Region base;
    base.id = 1;
    base.mat_ids = {1, 2};
    base.status = "design_val";
    base.range_type = "all";

    std::vector<Region> regions;
    if (lower_ascii(input.region_source) == "hmset") {
        regions.push_back(HMSetRegion(fem, input.design_hmset_name, base));
    } else {
        regions.push_back(RangeRegion(fem, base));
    }
    return regions;
}

std::vector<BoundaryCondition> InitializeDirichlet(const FEM& fem, const InputConfig& input) {
    BoundaryCondition base = make_base_dirichlet(input);
    std::vector<BoundaryCondition> dirich;
    if (lower_ascii(input.bc_source) == "hmset") {
        dirich.push_back(HMSetBoundaryCondition(fem, input.spc_hmset_name, base));
    } else {
        dirich.push_back(RangeBoundaryCondition(fem, base));
    }
    return dirich;
}

std::vector<BoundaryCondition> InitializeNeumann(const FEM& fem, const InputConfig& input) {
    BoundaryCondition base = make_base_neumann(input);
    std::vector<BoundaryCondition> neum;
    if (lower_ascii(input.bc_source) == "hmset") {
        neum.push_back(HMSetBoundaryCondition(fem, input.load_hmset_name, base));
    } else {
        neum.push_back(RangeBoundaryCondition(fem, base));
    }
    return neum;
}

std::unique_ptr<DynamicLinearElastic> InitializeAnalysis(const InputConfig& input,
                                                        const std::vector<Material>& materials,
                                                        const std::vector<Region>& regions,
                                                        const std::vector<BoundaryCondition>& dirich,
                                                        const std::vector<BoundaryCondition>& neum) {
    std::unique_ptr<DynamicLinearElastic> analysis_owner;
    if (lower_ascii(input.linear_backend) == "eigen") {
        analysis_owner = std::make_unique<EigenDynamicLinearElastic>();
    } else {
        analysis_owner = std::make_unique<PetscDynamicLinearElastic>();
    }
    DynamicLinearElastic& analysis = *analysis_owner;
    analysis.id = 1;
    analysis.mat_type = "elastic";
    analysis.Dirich_id = {1};
    analysis.Neum_id = {1};
    analysis.Tmax = input.tmax;
    analysis.Nstep = input.nstep;
    analysis.dt = analysis.Tmax / static_cast<double>(analysis.Nstep);

    analysis.ROM = InitializeROM(input);
    analysis.regions = regions;

    Interp interp;
    interp.p_E_elastic = input.interp_p_E_elastic;
    interp.p_rho = input.p_rho;
    interp.elastic_interp = [p = interp.p_E_elastic](const std::vector<double>& para, double x) {
        return RAMP(para, x, p);
    };
    interp.mass_density_interp = [p = interp.p_rho](const std::vector<double>& para, double x) {
        return RAMP(para, x, p);
    };

    init_mat(materials, analysis);
    init_interp(interp, analysis);
    attach_bcs_to_analysis_by_id(analysis, dirich, neum);
    return analysis_owner;
}

Optimization InitializeOptimization(const FEM& fem,
                                    const DynamicLinearElastic& analysis,
                                    const std::vector<Material>& materials,
                                    const InputConfig& input) {
    Optimization opt;
    opt.x = init_design_variables(fem, analysis.regions, materials, input.x_initial, opt.x_label);
    opt.MaxIter = input.max_iter;
    opt.Tol = 0.0;
    opt.filtering.rmin = input.filter_rmin;
    opt.filtering.beta = input.filter_beta;
    opt.filtering.beta_loop = input.filter_beta_loop;
    opt.filtering.beta_max = input.filter_beta_max;
    opt.filtering.T = input.filter_T;
    opt.algorithm.type = "MMA";
    opt.algorithm.move = input.move;
    opt.debug_fd_epsilon = input.debug_fd_epsilon;
    opt.debug_fd_max_checks = input.debug_fd_max_checks;
    opt.save_density_each_iter = input.save_density_each_iter != 0;
    opt.save_density_interval = std::max(1, input.save_density_interval);
    opt.Output = InitializeOutput(input);

    ObjectiveGroup obj_group;
    obj_group.analysis_id = analysis.id;
    obj_group.sens_type = input.sens_type;
    obj_group.sens_function = "dynamic_elastic_base";
    obj_group.type = is_rom_mode(input) ? "separate_BC" : "full";
    obj_group.ROM = InitializeROM(input);

    ObjectiveTerm term;
    term.function_name = "L2_compliance_elastic";
    term.weight = 1.0;
    obj_group.terms.push_back(term);
    opt.objective_groups.push_back(obj_group);

    Constraint constr;
    constr.analysis_id = analysis.id;
    constr.function_name = "volume_fraction";
    constr.limit = input.volume_limit;
    constr.sens_type = "AVM";
    constr.sens_function = "volume_sens_base";
    constr.type = "full";
    constr.mat_id = 2;
    constr.value_initial = 1.0;
    opt.constraints.push_back(constr);

    return opt;
}
