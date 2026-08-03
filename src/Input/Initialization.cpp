#include "Input/Initialization.hpp"

#include "Optimization/Optimization.hpp"
#include "ROM/ROM.hpp"

#include <petscksp.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

std::string json_string(const json& j, const char* key, const std::string& fallback = "") {
    return j.contains(key) && !j.at(key).is_null() ? j.at(key).get<std::string>() : fallback;
}

double json_double(const json& j, const char* key, double fallback = 0.0) {
    return j.contains(key) && !j.at(key).is_null() ? j.at(key).get<double>() : fallback;
}

int json_int(const json& j, const char* key, int fallback = 0) {
    return j.contains(key) && !j.at(key).is_null() ? j.at(key).get<int>() : fallback;
}

bool json_bool(const json& j, const char* key, bool fallback = false) {
    return j.contains(key) && !j.at(key).is_null() ? j.at(key).get<bool>() : fallback;
}

json analysis_by_id(const json& analysis_all_json, int analysis_id) {
    for (const auto& analysis_json : analysis_all_json) {
        if (json_int(analysis_json, "id", 0) == analysis_id) return analysis_json;
    }
    throw std::runtime_error("Analysis id not found in JSON: " + std::to_string(analysis_id));
}

std::shared_ptr<ROMBase> InitializeROM(const json& rom_json, const json& run_json) {
    const std::string linear_backend = json_string(run_json, "linear_backend", "petsc");
    const std::string solve_mode = lower_ascii(json_string(rom_json, "solve_mode", "FOM"));
    const std::string fom_solver_type = json_string(rom_json, "fom_solver_type", "direct");
    const std::string pod_backend = json_string(rom_json, "pod_backend", "snapshot_mkl");
    const std::string online_backend = json_string(rom_json, "online_backend", "incremental_mkl");
    if (!pod_backend.empty()) setenv("STAGE2_POD_BACKEND", pod_backend.c_str(), 1);
    if (!online_backend.empty()) setenv("STAGE2_ONLINE_BACKEND", online_backend.c_str(), 1);

    if (solve_mode != "rom") return make_fom_solver(fom_solver_type, linear_backend);
    return make_dynamic_ROM(json_string(rom_json, "method_name", "MyStrategy"),
                            fom_solver_type,
                            json_double(rom_json, "rom_tol", 0.02),
                            json_int(rom_json, "rank1", 10),
                            json_int(rom_json, "max_rank", 20),
                            json_bool(rom_json, "online", true),
                            linear_backend);
}

std::filesystem::path run_dir_from_json(const json& output_json,
                                        const json& run_json,
                                        const std::string& input_json_path) {
    if (output_json.contains("run_dir")) return output_json.at("run_dir").get<std::string>();
    const std::filesystem::path cfg_path = std::filesystem::path(input_json_path);
    const std::filesystem::path case_dir = cfg_path.has_parent_path() ? cfg_path.parent_path() : std::filesystem::path(".");
    return case_dir / json_string(run_json, "run_name", "FOM");
}

Interp make_interp(const json& interp_json) {
    Interp interp = interp_json.is_null() ? Interp{} : interp_json.get<Interp>();
    interp.elastic_interp = [p = interp.p_E_elastic](const std::vector<double>& para, double x) {
        return RAMP(para, x, p);
    };
    interp.mass_density_interp = [p = interp.p_rho](const std::vector<double>& para, double x) {
        return RAMP(para, x, p);
    };
    return interp;
}

} // namespace

std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

json read_json_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open input JSON: " + path);
    json cfg;
    in >> cfg;
    return cfg;
}

void from_json(const json& j, Geometry& g) {
    g.Lx = json_double(j, "Lx", g.Lx);
    g.Ly = json_double(j, "Ly", g.Ly);
    g.Lz = json_double(j, "Lz", g.Lz);
    g.nelx = json_int(j, "nelx", g.nelx);
    g.nely = json_int(j, "nely", g.nely);
    g.nelz = json_int(j, "nelz", g.nelz);
}

void from_json(const json& j, Material& m) {
    m.id = json_int(j, "id", m.id);
    m.type = json_string(j, "type", m.type);
    m.E_inf = json_double(j, "E_inf", m.E_inf);
    m.E = json_double(j, "E", m.E);
    m.nu = json_double(j, "nu", m.nu);
    m.rho = json_double(j, "rho", m.rho);
}

void from_json(const json& j, Interp& interp) {
    interp.p_E_elastic = json_double(j, "p_E_elastic", interp.p_E_elastic);
    interp.p_rho = json_double(j, "p_rho", interp.p_rho);
}

void from_json(const json& j, Region& r) {
    r.id = json_int(j, "id", r.id);
    r.name = json_string(j, "name", r.name);
    r.source = json_string(j, "source", r.source);
    r.status = json_string(j, "status", r.status);
    r.range_type = json_string(j, "range_type", r.range_type);
    r.hmset_name = json_string(j, "hmset_name", r.hmset_name);
    if (j.contains("mat_ids")) j.at("mat_ids").get_to(r.mat_ids);
    if (j.contains("x_range")) j.at("x_range").get_to(r.x_range);
    if (j.contains("y_range")) j.at("y_range").get_to(r.y_range);
    if (j.contains("z_range")) j.at("z_range").get_to(r.z_range);
}

void from_json(const json& j, BoundaryCondition& bc) {
    bc.id = json_int(j, "id", bc.id);
    bc.name = json_string(j, "name", bc.name);
    bc.source = json_string(j, "source", bc.source);
    bc.kind = json_string(j, "kind", bc.kind);
    bc.hmset_name = json_string(j, "hmset_name", bc.hmset_name);
    if (j.contains("flag")) j.at("flag").get_to(bc.flag);
    if (j.contains("x_range")) j.at("x_range").get_to(bc.x_range);
    if (j.contains("y_range")) j.at("y_range").get_to(bc.y_range);
    if (j.contains("z_range")) j.at("z_range").get_to(bc.z_range);
    bc.amp = json_double(j, "amp", bc.amp);
    bc.time_type = json_string(j, "time_type", bc.time_type);
    bc.nCycle = json_double(j, "nCycle", bc.nCycle);
    bc.power = json_double(j, "power", bc.power);
    bc.phase = json_double(j, "phase", bc.phase);
}

void from_json(const json& j, Filtering& f) {
    f.rmin = json_double(j, "rmin", f.rmin);
    f.beta = json_double(j, "beta", f.beta);
    f.beta_loop = json_int(j, "beta_loop", f.beta_loop);
    f.beta_max = json_double(j, "beta_max", f.beta_max);
    f.T = json_double(j, "T", f.T);
}

void from_json(const json& j, Algorithm& a) {
    a.type = json_string(j, "type", a.type);
    a.move = json_double(j, "move", a.move);
}

void from_json(const json& j, ObjectiveTerm& t) {
    t.function_name = json_string(j, "function_name", t.function_name);
    t.weight = json_double(j, "weight", t.weight);
    t.value_initial = json_double(j, "value_initial", t.value_initial);
}

void from_json(const json& j, ObjectiveGroup& g) {
    g.analysis_id = json_int(j, "analysis_id", g.analysis_id);
    g.sens_type = json_string(j, "sens_type", g.sens_type);
    g.sens_function = json_string(j, "sens_function", g.sens_function);
    g.type = json_string(j, "type", g.type);
    if (j.contains("terms")) j.at("terms").get_to(g.terms);
}

void from_json(const json& j, Constraint& c) {
    c.analysis_id = json_int(j, "analysis_id", c.analysis_id);
    c.function_name = json_string(j, "function_name", c.function_name);
    c.limit = json_double(j, "limit", c.limit);
    c.sens_type = json_string(j, "sens_type", c.sens_type);
    c.sens_function = json_string(j, "sens_function", c.sens_function);
    c.type = json_string(j, "type", c.type);
    c.mat_id = json_int(j, "mat_id", c.mat_id);
    c.value_initial = json_double(j, "value_initial", c.value_initial);
}

void apply_solver_options_from_json(const json& solver_json) {
    PetscOptionsSetValue(nullptr, "-ksp_type", json_string(solver_json, "ksp_type", "preonly").c_str()); // get ksp_type from json, default to preonly
    PetscOptionsSetValue(nullptr, "-pc_type", json_string(solver_json, "pc_type", "cholesky").c_str()); //  get pc_type from json, default to cholesky
    const std::string factor_solver = json_string(solver_json, "pc_factor_mat_solver_type", ""); // get pc_factor_mat_solver_type from json, default to empty string
    if (!factor_solver.empty()) PetscOptionsSetValue(nullptr, "-pc_factor_mat_solver_type", factor_solver.c_str());
    if (solver_json.contains("ksp_rtol")) {
        PetscOptionsSetValue(nullptr, "-ksp_rtol", std::to_string(json_double(solver_json, "ksp_rtol", 1e-8)).c_str());
    }
    if (solver_json.contains("ksp_max_it")) {
        PetscOptionsSetValue(nullptr, "-ksp_max_it", std::to_string(json_int(solver_json, "ksp_max_it", 10000)).c_str());
    }
}

void apply_runtime_environment_from_json(const json& run_json) {
    const json threads = run_json.value("threads", json::object()); 
    const int omp_threads = std::max(1, json_int(threads, "omp_num_threads", 1)); // get omp_num_threads from json, default to 1
    const int mkl_threads = std::max(1, json_int(threads, "mkl_num_threads", 1)); // get mkl_num_threads from json, default to 1
    const int factor_threads = std::max(1, json_int(threads, "factor_num_threads", 1)); // get factor_num_threads from json, default to 1
    const int sens_threads = std::max(1, json_int(threads, "sens_element_threads", 1)); // get sens_element_threads from json, default to 1
    setenv("OMP_NUM_THREADS", std::to_string(omp_threads).c_str(), 1); // set OMP_NUM_THREADS environment variable
    setenv("MKL_NUM_THREADS", std::to_string(mkl_threads).c_str(), 1); // set MKL_NUM_THREADS environment variable
    setenv("FACTOR_NUM_THREADS", std::to_string(factor_threads).c_str(), 1); // set FACTOR_NUM_THREADS environment variable
    setenv("SENS_ELEMENT_THREADS", std::to_string(sens_threads).c_str(), 1); // set SENS_ELEMENT_THREADS environment variable
}

std::unique_ptr<FEM> InitializeFEM(const json& fem_json, const Geometry& geom) {
    const std::string mesh_source = lower_ascii(json_string(fem_json, "mesh_source", "structured"));
    const std::string fem_type = json_string(fem_json, "type", "plane_stress");
    const double nu = json_double(fem_json, "nu", 0.3);
    const double thickness = json_double(fem_json, "Thickness", 1.0);
    const bool uniform = json_bool(fem_json, "uniform_element_matrices", mesh_source == "structured");

    if (mesh_source == "hypermesh" || mesh_source == "unstructured") {
        const std::string fem_file = json_string(fem_json, "fem_file", "");
        if (fem_file.empty()) throw std::runtime_error("FEM.fem_file is required for unstructured/hypermesh mesh");
        return std::make_unique<UnstructuredMesh>(fem_file, fem_type, nu, thickness, uniform);
    }
    if (mesh_source == "structured") {
        return std::make_unique<StructuredMesh>(geom, fem_type, nu, thickness, uniform);
    }
    throw std::runtime_error("Unknown FEM.mesh_source: " + mesh_source);
}

OutputData InitializeOutput(const json& output_json,
                            const json& run_json,
                            const std::string& input_json_path) {
    OutputData output;
    const std::filesystem::path run_dir = run_dir_from_json(output_json, run_json, input_json_path);
    output.outDir_mat = output_json.value("outDir_mat", (run_dir / "result").string());
    output.outDir_mat_FOM = output.outDir_mat;
    const std::filesystem::path case_dir = std::filesystem::path(output.outDir_mat).parent_path();
    output.outDir_vtk = output_json.value("outDir_vtk", (case_dir / "vtk").string());
    output.densityDir = output_json.value("densityDir", (case_dir / "density").string());
    output.figureDir = output_json.value("figureDir", (case_dir / "figure").string());
    output.saved_Iter = json_double(output_json, "saved_Iter", output.saved_Iter);
    output.save_density_each_iter = json_bool(output_json, "save_density_each_iter", true) ? 1 : 0;
    output.save_density_interval = std::max(1, json_int(output_json, "save_density_interval", 1));
    output.save_vtk_response_interval = std::max(1, json_int(output_json, "save_vtk_response_interval", 1));
    output.save_vtk_opt_interval = std::max(1, json_int(output_json, "save_vtk_opt_interval", 1));
    output.enable_live_density = json_bool(output_json, "enable_live_density", true);
    return output;
}

std::vector<Region> InitializeRegions(const json& regions_json, const FEM& fem) {
    std::vector<Region> regions = regions_json.get<std::vector<Region>>();
    if (regions.empty()) throw std::runtime_error("JSON Regions is empty");

    bool all_range = true;
    bool all_hmset = true;
    for (const auto& region : regions) {
        const std::string source = lower_ascii(region.source.empty() ? region.range_type : region.source);
        all_range = all_range && source == "range";
        all_hmset = all_hmset && source == "hmset";
    }
    if (all_range) {
        RangeRegion::assign_elements_to_regions(regions, fem);
        return regions;
    }
    if (all_hmset) {
        HMSetRegion::assign_elements_to_regions(regions, fem);
        return regions;
    }

    regions.clear();
    for (const auto& item : regions_json) {
        Region base = item.get<Region>();
        const std::string source = lower_ascii(base.source.empty() ? base.range_type : base.source);
        if (source == "hmset") regions.push_back(HMSetRegion(fem, base.hmset_name, base));
        else if (source == "range") regions.push_back(RangeRegion(fem, base));
        else throw std::runtime_error("Unknown Region.source: " + source);
    }
    return regions;
}

std::vector<BoundaryCondition> InitializeBoundaryConditions(const json& bcs_json, const FEM& fem) {
    std::vector<BoundaryCondition> bcs;
    for (const auto& item : bcs_json) {
        BoundaryCondition base = item.get<BoundaryCondition>();
        const std::string source = lower_ascii(base.source);
        if (source == "hmset") bcs.push_back(HMSetBoundaryCondition(fem, base.hmset_name, base));
        else if (source == "range") bcs.push_back(RangeBoundaryCondition(fem, base));
        else throw std::runtime_error("Unknown BoundaryCondition.source: " + source);
    }
    return bcs;
}

std::unique_ptr<DynamicLinearElastic> InitializeAnalysis(const json& analysis_json,
                                                        const json& run_json,
                                                        const json& interp_json,
                                                        const std::vector<Material>& materials,
                                                        const std::vector<Region>& regions,
                                                        const std::vector<BoundaryCondition>& dirich,
                                                        const std::vector<BoundaryCondition>& neum) {
    const std::string linear_backend = lower_ascii(json_string(run_json, "linear_backend", "petsc"));
    std::unique_ptr<DynamicLinearElastic> analysis_owner;
    if (linear_backend == "eigen") {
        analysis_owner = std::make_unique<EigenDynamicLinearElastic>();
    } else {
        analysis_owner = std::make_unique<PetscDynamicLinearElastic>();
    }

    DynamicLinearElastic& analysis = *analysis_owner;
    analysis.id = json_int(analysis_json, "id", analysis.id);
    analysis.mat_type = json_string(analysis_json, "mat_type", analysis.mat_type);
    if (analysis_json.contains("Dirich_id")) analysis_json.at("Dirich_id").get_to(analysis.Dirich_id);
    if (analysis_json.contains("Neum_id")) analysis_json.at("Neum_id").get_to(analysis.Neum_id);
    analysis.type = json_string(analysis_json, "type", analysis.type);
    analysis.method = json_string(analysis_json, "method", analysis.method);
    analysis.Tmax = json_double(analysis_json, "Tmax", analysis.Tmax);
    analysis.Nstep = json_int(analysis_json, "Nstep", analysis.Nstep);
    analysis.dt = analysis.Tmax / static_cast<double>(analysis.Nstep);
    analysis.solver_name = json_string(analysis_json, "solver_name", analysis.solver_name);
    analysis.ROM = InitializeROM(analysis_json.value("ROM", json::object()), run_json);
    analysis.regions = regions;

    const Interp interp = make_interp(interp_json);

    init_mat(materials, analysis);
    init_interp(interp, analysis);
    attach_bcs_to_analysis_by_id(analysis, dirich, neum);
    return analysis_owner;
}

Optimization InitializeOptimization(const json& optimization_json,
                                    const json& analysis_all_json,
                                    const json& output_json,
                                    const json& run_json,
                                    const std::string& input_json_path,
                                    const FEM& fem,
                                    const DynamicLinearElastic& analysis,
                                    const std::vector<Material>& materials) {
    Optimization opt;
    opt.x = init_design_variables(fem, analysis.regions, materials, json_double(optimization_json, "x0", 0.5), opt.x_label);
    opt.MaxIter = json_int(optimization_json, "MaxIter", opt.MaxIter);
    opt.Tol = json_double(optimization_json, "Tol", opt.Tol);
    opt.plot_during_optimization = json_bool(optimization_json, "plot_during_optimization", opt.plot_during_optimization);
    opt.debug_fd_epsilon = json_double(optimization_json, "debug_fd_epsilon", opt.debug_fd_epsilon);
    opt.debug_fd_max_checks = json_int(optimization_json, "debug_fd_max_checks", opt.debug_fd_max_checks);
    opt.save_density_each_iter = json_bool(optimization_json, "save_density_each_iter", opt.save_density_each_iter);
    opt.save_density_interval = std::max(1, json_int(optimization_json, "save_density_interval", opt.save_density_interval));
    if (optimization_json.contains("Filtering")) optimization_json.at("Filtering").get_to(opt.filtering);
    if (optimization_json.contains("Algorithm")) optimization_json.at("Algorithm").get_to(opt.algorithm);
    if (optimization_json.contains("ObjectiveGroups")) optimization_json.at("ObjectiveGroups").get_to(opt.objective_groups);
    if (optimization_json.contains("Constraints")) optimization_json.at("Constraints").get_to(opt.constraints);
    opt.Output = InitializeOutput(output_json, run_json, input_json_path);

    for (auto& group : opt.objective_groups) {
        json target_analysis = analysis_all_json.empty() ? json::object() : analysis_by_id(analysis_all_json, group.analysis_id);
        group.ROM = InitializeROM(target_analysis.value("ROM", json::object()), run_json);
    }
    return opt;
}
