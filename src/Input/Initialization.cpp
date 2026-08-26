#include "Input/Initialization.hpp"

#include <petscksp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

std::string lower_ascii(std::string s) {
    return myjson::lower_ascii(s);
}

json read_json_file(const std::string& path) {
    return myjson::read_file(path);
}

void InitializationData::apply_solver_options() {
    const json solver_json = cfg.value("Solver", json::object());
    PetscOptionsSetValue(nullptr, "-ksp_type", myjson::string_value(solver_json, "ksp_type", "preonly").c_str());
    PetscOptionsSetValue(nullptr, "-pc_type", myjson::string_value(solver_json, "pc_type", "cholesky").c_str());
    const std::string factor_solver = myjson::string_value(solver_json, "pc_factor_mat_solver_type", "");
    if (!factor_solver.empty()) PetscOptionsSetValue(nullptr, "-pc_factor_mat_solver_type", factor_solver.c_str());
    if (solver_json.contains("ksp_rtol")) {
        PetscOptionsSetValue(nullptr, "-ksp_rtol", std::to_string(myjson::double_value(solver_json, "ksp_rtol", 1e-8)).c_str());
    }
    if (solver_json.contains("ksp_max_it")) {
        PetscOptionsSetValue(nullptr, "-ksp_max_it", std::to_string(myjson::int_value(solver_json, "ksp_max_it", 10000)).c_str());
    }
}

void InitializationData::apply_runtime_environment() {
    const json threads = run_json.value("threads", json::object());
    const int omp_threads = std::max(1, myjson::int_value(threads, "omp_num_threads", 1));
    const int mkl_threads = std::max(1, myjson::int_value(threads, "mkl_num_threads", 1));
    const int factor_threads = std::max(1, myjson::int_value(threads, "factor_num_threads", 1));
    const int sens_threads = std::max(1, myjson::int_value(threads, "sens_element_threads", 1));
    setenv("OMP_NUM_THREADS", std::to_string(omp_threads).c_str(), 1);
    setenv("MKL_NUM_THREADS", std::to_string(mkl_threads).c_str(), 1);
    setenv("FACTOR_NUM_THREADS", std::to_string(factor_threads).c_str(), 1);
    setenv("SENS_ELEMENT_THREADS", std::to_string(sens_threads).c_str(), 1);
}

InitializationData::InitializationData(const std::string& input_json_path) {
    cfg = read_json_file(input_json_path);
    run_json = cfg.at("Run");

    apply_solver_options();
    apply_runtime_environment();

    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size != 1 && lower_ascii(run_json.value("linear_backend", std::string("petsc"))) == "eigen") {
        if (rank == 0) {
            std::cerr << "[warning] LINEAR_BACKEND=eigen requires mpi_size=1; "
                      << "falling back to LINEAR_BACKEND=petsc for this run.\n";
        }
        run_json["linear_backend"] = "petsc";
    }

    fem = InitializeFEM(cfg.at("FEM"));
    interps = InitializeInterpolations(cfg.at("Interp"));
    fem->Regions = InitializeRegions(cfg.at("Regions"), *fem);
    fem->InitializeMaterials(cfg.at("Materials"));

    const json bc_json = cfg.at("BoundaryConditions");
    dirich = InitializeBoundaryConditions(bc_json.at("Dirich"), *fem);
    neum = InitializeBoundaryConditions(bc_json.at("Neum"), *fem);

    analysis_all_json = cfg.at("Analysis");
    std::vector<std::unique_ptr<Analysis>> analysis_all;
    analysis_all.reserve(analysis_all_json.size());
    
    for (const auto& analysis_json : analysis_all_json) {

        analysis_all.push_back(InitializeAnalysis(analysis_json,run_json, interps,*fem, dirich, neum));

    }
    if (analysis_all.empty()) throw std::runtime_error("JSON Analysis is empty");

    optimization = InitializeOptimization(cfg.at("Optimization"),
                                          analysis_all_json,
                                          cfg.value("Output", json::object()),
                                          run_json,
                                          input_json_path,
                                          *fem,
                                          std::move(analysis_all));
}
