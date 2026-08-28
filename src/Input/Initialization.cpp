#include "Input/Initialization.hpp"

#include <petscksp.h>
#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

void InitializationData::apply_solver_options() {
    const json solver_json = cfg.value("Solver", json::object());
    PetscOptionsSetValue(nullptr, "-ksp_type", myjson::string_value(solver_json, "ksp_type", "preonly").c_str());
    PetscOptionsSetValue(nullptr, "-pc_type", myjson::string_value(solver_json, "pc_type", "cholesky").c_str());
    const std::string factor_solver = myjson::string_value(solver_json, "factor_solver", "");
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
    //
    setenv("OMP_NUM_THREADS", std::to_string(omp_threads).c_str(), 1);  // general for OpenMP
    setenv("MKL_NUM_THREADS", std::to_string(mkl_threads).c_str(), 1);  // for MKL, BLAS/LAPACK, example: dgemm(...)
    setenv("FACTOR_NUM_THREADS", std::to_string(factor_threads).c_str(), 1); // factorizattion
    setenv("SENS_ELEMENT_THREADS", std::to_string(sens_threads).c_str(), 1); // explicit derivative, for element loops
    omp_set_num_threads(omp_threads);
}

InitializationData::InitializationData(const std::string& input_json_path) {
    cfg = myjson::read_file(input_json_path);
    run_json = cfg.at("Run");

    apply_solver_options();
    apply_runtime_environment();

    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // check number of nodes for MPI for Petsc, when using Eigen, the number of nodes N must = 1
    if (size != 1 && myjson::lower_ascii(run_json.value("linear_backend", std::string("petsc"))) == "eigen") {
        if (rank == 0) {
            std::cerr << "[warning] LINEAR_BACKEND=eigen requires mpi_size=1; "
                      << "falling back to LINEAR_BACKEND=petsc for this run.\n";
        } 
        run_json["linear_backend"] = "petsc"; // change to Petsc when number of nodes >1
    }

    fem = InitializeFEM(cfg.at("FEM"));
    fem->InitializeRegions(cfg.at("Regions"));
    fem->InitializeMaterials(cfg.at("Materials"));
    interps = InitializeInterpolations(cfg.at("Interp"));

    const json bc_json = cfg.at("BoundaryConditions");
    dirich = InitializeBoundaryConditions(bc_json.at("Dirich"), *fem);
    neum = InitializeBoundaryConditions(bc_json.at("Neum"), *fem);

    analysis_all_json = cfg.at("Analysis");
    std::vector<std::shared_ptr<Analysis>> analysis_all;
    analysis_all.reserve(analysis_all_json.size());
    
    for (const auto& analysis_json : analysis_all_json) {

        analysis_all.push_back(InitializeAnalysis(analysis_json, run_json, interps, dirich, neum));

    }
    if (analysis_all.empty()) throw std::runtime_error("JSON Analysis is empty");

    optimization = InitializeOptimization(cfg.at("Optimization"),
                                          cfg.value("Output", json::object()),
                                          run_json, input_json_path, *fem,
                                          analysis_all);
}
