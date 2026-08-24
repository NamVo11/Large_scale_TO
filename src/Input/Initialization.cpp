#include "Input/Initialization.hpp"

#include "Optimization/InitializeGroupFunctional.hpp"
#include "Optimization/Optimization.hpp"
#include "ROM/ROM.hpp"
#include "myjson.hpp"

#include <petscksp.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace {

json analysis_by_id(const json& analysis_all_json, int analysis_id) {
    for (const auto& analysis_json : analysis_all_json) {
        if (myjson::int_value(analysis_json, "id", 0) == analysis_id) return analysis_json;
    }
    throw std::runtime_error("Analysis id not found in JSON: " + std::to_string(analysis_id));
}

std::shared_ptr<ROMBase> InitializeROM(const json& rom_json, const json& run_json) {
    const std::string linear_backend = myjson::string_value(run_json, "linear_backend", "petsc");
    const std::string solve_mode = lower_ascii(myjson::string_value(rom_json, "solve_mode", "FOM"));
    const std::string fom_solver_type = myjson::string_value(rom_json, "fom_solver_type", "direct");
    const std::string pod_backend = myjson::string_value(rom_json, "pod_backend", "snapshot_mkl");
    const std::string online_backend = myjson::string_value(rom_json, "online_backend", "incremental_mkl");
    if (!pod_backend.empty()) setenv("STAGE2_POD_BACKEND", pod_backend.c_str(), 1);
    if (!online_backend.empty()) setenv("STAGE2_ONLINE_BACKEND", online_backend.c_str(), 1);

    if (solve_mode != "rom") return make_fom_solver(fom_solver_type, linear_backend);
    return make_dynamic_ROM(myjson::string_value(rom_json, "method_name", "MyStrategy"),
                            fom_solver_type,
                            myjson::double_value(rom_json, "rom_tol", 0.02),
                            myjson::int_value(rom_json, "rank1", 10),
                            myjson::int_value(rom_json, "max_rank", 20),
                            myjson::bool_value(rom_json, "online", true),
                            linear_backend);
}

std::filesystem::path run_dir_from_json(const json& output_json,
                                        const json& run_json,
                                        const std::string& input_json_path) {
    if (output_json.contains("run_dir")) return output_json.at("run_dir").get<std::string>();
    const std::filesystem::path cfg_path = std::filesystem::path(input_json_path);
    const std::filesystem::path case_dir = cfg_path.has_parent_path() ? cfg_path.parent_path() : std::filesystem::path(".");
    return case_dir / myjson::string_value(run_json, "run_name", "FOM");
}

bool in_range(double x, const std::array<double, 2>& range) {
    return x >= range[0] && x <= range[1];
}

const HMSet& find_hmset_by_name_local(const FEM& fem, const std::string& name) {
    auto it = fem.hm_sets_by_name.find(name);
    if (it != fem.hm_sets_by_name.end()) return it->second;
    it = fem.hm_sets_by_name.find(lower_ascii(name));
    if (it != fem.hm_sets_by_name.end()) return it->second;
    throw std::runtime_error("HMSET not found: " + name);
}

bool element_center_in_region_box(const FEM& fem, const Region& region, int elem_id) {
    if (region.x_range.empty() || region.y_range.empty()) {
        throw std::runtime_error("Region box needs x/y ranges: " + region.name);
    }
    if (region.x_range.size() != region.y_range.size()) {
        throw std::runtime_error("Region x/y range size mismatch: " + region.name);
    }
    if (fem.dim == 3 && region.z_range.size() != region.x_range.size()) {
        throw std::runtime_error("Region z range size mismatch: " + region.name);
    }

    const auto& center = fem.elem_center[static_cast<std::size_t>(elem_id)];
    for (std::size_t r = 0; r < region.x_range.size(); ++r) {
        bool inside = in_range(center[0], region.x_range[r]) && in_range(center[1], region.y_range[r]);
        if (fem.dim == 3) inside = inside && in_range(center[2], region.z_range[r]);
        if (inside) return true;
    }
    return false;
}

void assign_range_region(Region& region, const FEM& fem, const std::vector<int>& assigned) {
    region.elem_ids.clear();
    const std::string range_type = lower_ascii(region.range_type.empty() ? "all" : region.range_type);
    if (range_type == "all") {
        region.elem_ids.resize(static_cast<std::size_t>(fem.NElem));
        std::iota(region.elem_ids.begin(), region.elem_ids.end(), 0);
    } else if (range_type == "box") {
        for (int e = 0; e < fem.NElem; ++e) {
            if (element_center_in_region_box(fem, region, e)) region.elem_ids.push_back(e);
        }
    } else if (range_type == "other") {
        for (int e = 0; e < fem.NElem; ++e) {
            if (assigned[static_cast<std::size_t>(e)] == 0) region.elem_ids.push_back(e);
        }
    } else {
        throw std::runtime_error("Unknown Region.range_type: " + range_type);
    }
}

void assign_hmset_region(Region& region, const FEM& fem) {
    const std::string set_name = !region.hmset_name.empty() ? region.hmset_name : region.status;
    const HMSet& set = find_hmset_by_name_local(fem, set_name);
    if (set.kind != 2) throw std::runtime_error("HMSET is not an element set: " + set_name);
    region.elem_ids = set.local_ids;
}

void finalize_region_assignment(Region& region, const FEM& fem, std::vector<int>& assigned) {
    std::sort(region.elem_ids.begin(), region.elem_ids.end());
    region.elem_ids.erase(std::unique(region.elem_ids.begin(), region.elem_ids.end()), region.elem_ids.end());
    if (region.elem_ids.empty()) throw std::runtime_error("Region selects no elements: " + region.name);

    for (int e : region.elem_ids) {
        if (e < 0 || e >= fem.NElem) throw std::runtime_error("Region has invalid element local id: " + region.name);
        if (assigned[static_cast<std::size_t>(e)] != 0) {
            throw std::runtime_error("Overlapping regions around element " + std::to_string(e) + ": " + region.name);
        }
        assigned[static_cast<std::size_t>(e)] = region.id == 0 ? 1 : region.id;
    }
    region.nElem = static_cast<int>(region.elem_ids.size());
}

Interp make_interp(const json& interp_json) {
    Interp interp = interp_json.is_null() ? Interp{} : Interp(interp_json);
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
    return myjson::lower_ascii(s);
}

json read_json_file(const std::string& path) {
    return myjson::read_file(path);
}

void apply_solver_options_from_json(const json& solver_json) {
    PetscOptionsSetValue(nullptr, "-ksp_type", myjson::string_value(solver_json, "ksp_type", "preonly").c_str()); // get ksp_type from json, default to preonly
    PetscOptionsSetValue(nullptr, "-pc_type", myjson::string_value(solver_json, "pc_type", "cholesky").c_str()); //  get pc_type from json, default to cholesky
    const std::string factor_solver = myjson::string_value(solver_json, "pc_factor_mat_solver_type", ""); // get pc_factor_mat_solver_type from json, default to empty string
    if (!factor_solver.empty()) PetscOptionsSetValue(nullptr, "-pc_factor_mat_solver_type", factor_solver.c_str());
    if (solver_json.contains("ksp_rtol")) {
        PetscOptionsSetValue(nullptr, "-ksp_rtol", std::to_string(myjson::double_value(solver_json, "ksp_rtol", 1e-8)).c_str());
    }
    if (solver_json.contains("ksp_max_it")) {
        PetscOptionsSetValue(nullptr, "-ksp_max_it", std::to_string(myjson::int_value(solver_json, "ksp_max_it", 10000)).c_str());
    }
}

void apply_runtime_environment_from_json(const json& run_json) {
    const json threads = run_json.value("threads", json::object()); 
    const int omp_threads = std::max(1, myjson::int_value(threads, "omp_num_threads", 1)); // get omp_num_threads from json, default to 1
    const int mkl_threads = std::max(1, myjson::int_value(threads, "mkl_num_threads", 1)); // get mkl_num_threads from json, default to 1
    const int factor_threads = std::max(1, myjson::int_value(threads, "factor_num_threads", 1)); // get factor_num_threads from json, default to 1
    const int sens_threads = std::max(1, myjson::int_value(threads, "sens_element_threads", 1)); // get sens_element_threads from json, default to 1
    setenv("OMP_NUM_THREADS", std::to_string(omp_threads).c_str(), 1); // set OMP_NUM_THREADS environment variable
    setenv("MKL_NUM_THREADS", std::to_string(mkl_threads).c_str(), 1); // set MKL_NUM_THREADS environment variable
    setenv("FACTOR_NUM_THREADS", std::to_string(factor_threads).c_str(), 1); // set FACTOR_NUM_THREADS environment variable
    setenv("SENS_ELEMENT_THREADS", std::to_string(sens_threads).c_str(), 1); // set SENS_ELEMENT_THREADS environment variable
}

std::unique_ptr<FEM> InitializeFEM(const json& fem_json, const Geometry& geom) {
    const std::string mesh_source = lower_ascii(myjson::string_value(fem_json, "mesh_source", "structured"));
    const std::string fem_type = myjson::string_value(fem_json, "type", "plane_stress");
    const double nu = myjson::double_value(fem_json, "nu", 0.3);
    const double thickness = myjson::double_value(fem_json, "Thickness", 1.0);
    const bool uniform = myjson::bool_value(fem_json, "uniform_element_matrices", mesh_source == "structured");

    if (mesh_source == "hypermesh" || mesh_source == "unstructured") {
        const std::string fem_file = myjson::string_value(fem_json, "fem_file", "");
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
    output.saved_Iter = myjson::double_value(output_json, "saved_Iter", output.saved_Iter);
    output.save_density_each_iter = myjson::bool_value(output_json, "save_density_each_iter", true) ? 1 : 0;
    output.save_density_interval = std::max(1, myjson::int_value(output_json, "save_density_interval", 1));
    output.save_vtk_response_interval = std::max(1, myjson::int_value(output_json, "save_vtk_response_interval", 1));
    output.save_vtk_opt_interval = std::max(1, myjson::int_value(output_json, "save_vtk_opt_interval", 1));
    output.enable_live_density = myjson::bool_value(output_json, "enable_live_density", true);
    return output;
}

std::vector<Region> InitializeRegions(const json& regions_json, const FEM& fem) {
    if (!regions_json.is_array() || regions_json.empty()) throw std::runtime_error("JSON Regions is empty");

    std::vector<Region> regions;
    std::vector<int> assigned(static_cast<std::size_t>(fem.NElem), 0);

    for (const auto& item : regions_json) {
        Region region(item);
        const std::string source = lower_ascii(region.source.empty() ? "range" : region.source);
        if (source == "hmset") {
            assign_hmset_region(region, fem);
        } else if (source == "range") {
            assign_range_region(region, fem, assigned);
        } else {
            throw std::runtime_error("Unknown Region.source: " + source);
        }
        finalize_region_assignment(region, fem, assigned);
        regions.push_back(std::move(region));
    }

    if (std::any_of(assigned.begin(), assigned.end(), [](int v) { return v == 0; })) {
        throw std::runtime_error("Some elements are not assigned to any region");
    }
    return regions;
}

std::vector<std::shared_ptr<BoundaryCondition>> InitializeBoundaryConditions(const json& bcs_json, const FEM& fem) {
    std::vector<std::shared_ptr<BoundaryCondition>> bcs;
    for (const auto& item : bcs_json) {
        const std::string source = lower_ascii(myjson::string_value(item, "source", "range"));
        if (source == "hmset") bcs.push_back(std::make_shared<HMSetBoundaryCondition>(fem, item));
        else if (source == "range") bcs.push_back(std::make_shared<RangeBoundaryCondition>(fem, item));
        else throw std::runtime_error("Unknown BoundaryCondition.source: " + source);
    }
    return bcs;
}

std::unique_ptr<Analysis> InitializeAnalysis(const json& analysis_json,
                                             const json& run_json,
                                             const json& interp_json,
                                             const std::vector<Material>& materials,
                                             const std::vector<Region>& regions,
                                             const std::vector<std::shared_ptr<BoundaryCondition>>& dirich,
                                             const std::vector<std::shared_ptr<BoundaryCondition>>& neum) {
    const std::string linear_backend = lower_ascii(myjson::string_value(run_json, "linear_backend", "petsc"));
    const std::string owner = "Analysis[" + std::to_string(myjson::int_value(analysis_json, "id", 0)) + "]";
    const std::string type = lower_ascii(myjson::required_string(analysis_json, "type", owner));
    const std::string method = lower_ascii(myjson::required_string(analysis_json, "method", owner));
    const std::string solver_name = lower_ascii(myjson::required_string(analysis_json, "solver_name", owner));

    std::unique_ptr<Analysis> analysis;
    if (type == "dynamic" && method == "linear" && solver_name == "dynamic_elastic_linear") {
        if (linear_backend == "eigen") analysis = std::make_unique<EigenDynamicLinearElastic>(analysis_json);
        else analysis = std::make_unique<PetscDynamicLinearElastic>(analysis_json);
    } else if (type == "static" && method == "linear" && solver_name == "static_elastic_linear") {
        if (linear_backend == "eigen") analysis = std::make_unique<EigenStaticLinearElastic>(analysis_json);
        else analysis = std::make_unique<PetscStaticLinearElastic>(analysis_json);
    } else if (type == "dynamic" && method == "linear" && solver_name == "dynamic_viscoelastic_linear") {
        throw std::runtime_error("dynamic_viscoelastic_linear is recognized but the Analysis class is not implemented yet");
    } else {
        throw std::runtime_error("Unsupported Analysis type/method/solver_name: " +
                                 type + "/" + method + "/" + solver_name);
    }

    analysis->ROM = InitializeROM(analysis_json.value("ROM", json::object()), run_json);
    analysis->regions = regions;

    const Interp interp = make_interp(interp_json);

    analysis->init_mat(materials);
    analysis->init_interp(interp);
    analysis->attach_bcs_by_id(dirich, neum);
    return analysis;
}

Optimization InitializeOptimization(const json& optimization_json,
                                    const json& analysis_all_json,
                                    const json& output_json,
                                    const json& run_json,
                                    const std::string& input_json_path,
                                    const FEM& fem,
                                    const Analysis& analysis,
                                    const std::vector<Material>& materials) {
    Optimization opt;
    opt.x = init_design_variables(fem, analysis.regions, materials, myjson::double_value(optimization_json, "x0", 0.5), opt.x_label);
    opt.MaxIter = myjson::int_value(optimization_json, "MaxIter", opt.MaxIter);
    opt.Tol = myjson::double_value(optimization_json, "Tol", opt.Tol);
    opt.plot_during_optimization = myjson::bool_value(optimization_json, "plot_during_optimization", opt.plot_during_optimization);
    opt.debug_fd_epsilon = myjson::double_value(optimization_json, "debug_fd_epsilon", opt.debug_fd_epsilon);
    opt.debug_fd_max_checks = myjson::int_value(optimization_json, "debug_fd_max_checks", opt.debug_fd_max_checks);
    if (optimization_json.contains("Filtering")) opt.filtering = Filtering(optimization_json.at("Filtering"));
    if (optimization_json.contains("Algorithm")) opt.algorithm = Algorithm(optimization_json.at("Algorithm"));

    const std::string linear_backend = myjson::string_value(run_json, "linear_backend", "petsc");

    for (const auto& group_json : optimization_json.value("ObjectiveGroups", json::array())) {
        opt.objective.add_group(InitializeGroupFunctional(group_json, linear_backend));
    }

    for (const auto& constraint_json : optimization_json.value("Constraints", json::array())) {
        opt.constraint.add_group(InitializeGroupFunctional(constraint_json, linear_backend), constraint_limit_from_json(constraint_json));

    }
    opt.Output = InitializeOutput(output_json, run_json, input_json_path);

    for (auto& group : opt.objective.groups) {
        json target_analysis = analysis_all_json.empty() ? json::object() : analysis_by_id(analysis_all_json, group->analysis_id);
        group->ROM = InitializeROM(target_analysis.value("ROM", json::object()), run_json);
    }

    for (auto& group : opt.constraint.groups) {
        json target_analysis = analysis_all_json.empty() ? json::object() : analysis_by_id(analysis_all_json, group->analysis_id);
        group->ROM = InitializeROM(target_analysis.value("ROM", json::object()), run_json);
    }
    return opt;
}
