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
        InputConfig input = InitializeInputConfig();
        if (size != 1 && lower_ascii(input.linear_backend) == "eigen") {
            if (rank == 0) {
                std::cerr << "[warning] LINEAR_BACKEND=eigen requires mpi_size=1; "
                          << "falling back to LINEAR_BACKEND=petsc for this run.\n";
            }
            input.linear_backend = "petsc";
        }

        std::unique_ptr<FEM> fem_owner = InitializeFEM(input);
        FEM& fem = *fem_owner;
        OutputData Output = InitializeOutput(input);
        std::vector<Material> Mat = InitializeMaterial(input);
        std::vector<Region> regions = InitializeRegions(fem, input);
        std::vector<BoundaryCondition> Dirich = InitializeDirichlet(fem, input);
        std::vector<BoundaryCondition> Neum = InitializeNeumann(fem, input);
        std::unique_ptr<DynamicLinearElastic> A1_owner = InitializeAnalysis(input, Mat, regions, Dirich, Neum);
        DynamicLinearElastic& A1 = *A1_owner;
        Optimization Opt = InitializeOptimization(fem, A1, Mat, input);

        if (rank == 0) {
            std::cout << "stage2_dynamic_forward: mesh=" << fem.mesh_type
                      << " ndof=" << fem.ndof
                      << " nelem=" << fem.NElem
                      << " nstep=" << A1.Nstep
                      << " fem_uniform=" << (fem.uniform_element_matrices ? 1 : 0)
                      << " factor_threads=" << input.factor_num_threads
                      << " sens_element_threads=" << input.sens_element_threads
                      << " linear_backend=" << input.linear_backend
                      << " mpi_size=" << size << "\n";
        }

        std::unique_ptr<OptimizationBase> OptSolver =
            make_optimization_solver(input.linear_backend, std::move(A1_owner), std::move(Opt));
        if (input.run_optimization != 0) {
            OptSolver->run(fem);
        } else {
            OptSolver->primary_analysis().solver(fem, OptSolver->x, 0);
        }
        DynamicLinearElastic& Aout = OptSolver->primary_analysis();

        if (rank == 0) {
            std::filesystem::create_directories(input.out_dir);
            write_matrix_columns(input.out_dir + "/U_history.csv", Aout.results.U);
            write_info(input.out_dir + "/solver_info.txt", fem, Aout, size);
            std::cout << "stage2 output: " << input.out_dir << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "[rank " << rank << "] stage2_dynamic_forward error: " << ex.what() << "\n";
        exit_code = 1;
    }

    PetscFinalize();
    return exit_code;
}
