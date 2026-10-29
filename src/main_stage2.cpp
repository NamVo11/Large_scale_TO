#include "Input/Initialization.hpp"
#include "Optimization/Optimization.hpp"

#include <petscksp.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

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

        InitializationData data(input_json_buf);
        FEM& fem = *data.fem;
        if (data.optimization.analysis_all.empty()) throw std::runtime_error("Optimization has no Analysis object");
        Analysis& A1 = *data.optimization.analysis_all.front();

        if (rank == 0) {
            const char* factor_env = std::getenv("FACTOR_NUM_THREADS");
            const char* sens_env = std::getenv("SENS_ELEMENT_THREADS");
            std::cout << "stage2_dynamic_forward: fem_type=" << fem.fem_type
                      << " ndof=" << fem.ndof
                      << " nelem=" << fem.NElem
                      << " nstep=" << A1.Nstep
                      << " fem_uniform=" << (fem.uniform_element_matrices ? 1 : 0)
                      << " factor_threads=" << (factor_env ? factor_env : "<unset>")
                      << " sens_element_threads=" << (sens_env ? sens_env : "<unset>")
                      << " linear_backend=" << data.run_json.value("linear_backend", std::string("petsc"))
                      << " mpi_size=" << size << "\n";
        }

        if (data.run_json.value("run_optimization", true)) {
            data.optimization.run(fem);
        } else {
            data.optimization.analysis_all.front()->solver(fem, data.optimization.x, 0);
            const auto& response = *data.optimization.analysis_all.front();
            std::cout << "analysis max_abs_u=" << response.U.cwiseAbs().maxCoeff() << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "[rank " << rank << "] stage2_dynamic_forward error: " << ex.what() << "\n";
        exit_code = 1;
    }

    PetscFinalize();
    return exit_code;
}
