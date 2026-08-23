#include "Input/Initialization.hpp"
#include "MatlabLikeData.hpp"
#include "Optimization/Optimization.hpp"
#include "ROM/ROM.hpp"

#include <petscksp.h>

#include <filesystem>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

void write_matrix_columns(const std::string& path, const Eigen::MatrixXd& cols) {
    if (cols.size() == 0) return;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path);
    out.precision(17);
    for (Eigen::Index i = 0; i < cols.rows(); ++i) {
        for (Eigen::Index j = 0; j < cols.cols(); ++j) {
            if (j) out << ",";
            out << cols(i, j);
        }
        out << "\n";
    }
}

void write_info(const std::string& path,
                const FEM& fem,
                const Analysis& A,
                int mpi_size) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path);
    out.precision(17);
    out << "mode=stage2_matlab_like_forward\n";
    out << "mesh_type=" << fem.mesh_type << "\n";
    out << "dim=" << fem.dim << "\n";
    out << "nnodes=" << fem.NNode << "\n";
    out << "nelem=" << fem.NElem << "\n";
    out << "ndof=" << fem.ndof << "\n";
    out << "nstep=" << A.Nstep << "\n";
    out << "dt=" << A.dt << "\n";
    out << "tmax=" << A.Tmax << "\n";
    out << "rom_name=" << (A.ROM ? A.ROM->name : "none") << "\n";
    out << "mpi_size=" << mpi_size << "\n";
    out << "assembly_time=" << A.results.assembly_time << "\n";
    out << "solve_time=" << A.results.solve_time << "\n";
    out << "ksp_iterations=";
    for (int v : A.results.ksp_iterations) out << "," << v;
    out << "\n";
    out << "ksp_residuals=";
    for (double v : A.results.ksp_residuals) out << "," << v;
    out << "\n";
}

} // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    int exit_code = 0;
    try {
        char input_json_buf[PETSC_MAX_PATH_LEN] = {0};
        PetscBool input_json_set = PETSC_FALSE;
        PetscOptionsGetString(nullptr, nullptr, "-input_json", input_json_buf, sizeof(input_json_buf), &input_json_set);
        if (!input_json_set) throw std::runtime_error("-input_json is required");

        const std::string input_json_path = input_json_buf;
        json cfg = read_json_file(input_json_path);
        json run_json = cfg.at("Run");
        apply_solver_options_from_json(cfg.value("Solver", json::object()));
        apply_runtime_environment_from_json(run_json);

        if (size != 1 && lower_ascii(run_json.value("linear_backend", std::string("petsc"))) == "eigen") {
            if (rank == 0) {
                std::cerr << "[warning] LINEAR_BACKEND=eigen requires mpi_size=1; "
                          << "falling back to LINEAR_BACKEND=petsc for this run.\n";
            }
            run_json["linear_backend"] = "petsc";
        }

        Geometry geom = cfg.at("Geom").get<Geometry>();
        std::unique_ptr<FEM> fem_owner = InitializeFEM(cfg.at("FEM"), geom);
        FEM& fem = *fem_owner;
        std::vector<Material> Mat = cfg.at("Materials").get<std::vector<Material>>();
        std::vector<Region> regions = InitializeRegions(cfg.at("Regions"), fem);
        const json bc_json = cfg.at("BoundaryConditions");
        std::vector<BoundaryCondition> Dirich = InitializeBoundaryConditions(bc_json.at("Dirich"), fem);
        std::vector<BoundaryCondition> Neum = InitializeBoundaryConditions(bc_json.at("Neum"), fem);
        const json analysis_all_json = cfg.at("Analysis");


        std::unique_ptr<DynamicLinearElastic> A1_owner =
            InitializeAnalysis(analysis_all_json.at(0), run_json, cfg.value("Interp", json::object()), Mat, regions, Dirich, Neum);
            
        DynamicLinearElastic& A1 = *A1_owner;


        Optimization Opt = InitializeOptimization(cfg.at("Optimization"),
                                                  analysis_all_json,
                                                  cfg.value("Output", json::object()),
                                                  run_json,
                                                  input_json_path,
                                                  fem,
                                                  A1,
                                                  Mat);

        if (rank == 0) {
            const char* factor_env = std::getenv("FACTOR_NUM_THREADS");
            const char* sens_env = std::getenv("SENS_ELEMENT_THREADS");
            std::cout << "stage2_dynamic_forward: mesh=" << fem.mesh_type
                      << " ndof=" << fem.ndof
                      << " nelem=" << fem.NElem
                      << " nstep=" << A1.Nstep
                      << " fem_uniform=" << (fem.uniform_element_matrices ? 1 : 0)
                      << " factor_threads=" << (factor_env ? factor_env : "<unset>")
                      << " sens_element_threads=" << (sens_env ? sens_env : "<unset>")
                      << " linear_backend=" << run_json.value("linear_backend", std::string("petsc"))
                      << " mpi_size=" << size << "\n";
        }

        std::unique_ptr<OptimizationBase> OptSolver =
            make_optimization_solver(run_json.value("linear_backend", std::string("petsc")), std::move(A1_owner), std::move(Opt));
        if (run_json.value("run_optimization", true)) {
            OptSolver->run(fem);
        } else {
            OptSolver->primary_analysis().solver(fem, OptSolver->x, 0);
        }

        /*
        DynamicLinearElastic& Aout = OptSolver->primary_analysis();

        if (rank == 0) {
            const std::string out_dir = OptSolver->Output.outDir_mat;
            std::filesystem::create_directories(out_dir);
            write_matrix_columns(out_dir + "/U_history.csv", Aout.results.U);
            write_info(out_dir + "/solver_info.txt", fem, Aout, size);
            std::cout << "stage2 output: " << out_dir << "\n";
        }
        */
    
    } catch (const std::exception& ex) {
        std::cerr << "[rank " << rank << "] stage2_dynamic_forward error: " << ex.what() << "\n";
        exit_code = 1;
    }

    PetscFinalize();
    return exit_code;
}
