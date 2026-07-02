#include "DynamicElasticSolver.hpp"
#include "IO.hpp"

#include <petscksp.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

constexpr const char* kDebugManifestPath = "debug_manifest.txt";

std::string trim(const std::string& s) {
    auto begin = std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
    auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    if (begin >= end) return "";
    return std::string(begin, end);
}

std::unordered_map<std::string, std::string> readManifest(const std::string& path) {
    std::ifstream in(path);
    std::unordered_map<std::string, std::string> manifest;
    if (!in) return manifest;

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        if (!key.empty()) manifest[key] = value;
    }
    return manifest;
}

bool truthy(const std::string& value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

void applyPetscOptionsFromManifest(const std::unordered_map<std::string, std::string>& manifest) {
    const char* petscKeys[] = {"ksp_type", "pc_type", "ksp_rtol", "ksp_atol", "ksp_max_it"};
    for (const char* key : petscKeys) {
        auto it = manifest.find(key);
        if (it == manifest.end()) continue;
        std::string option = "-" + std::string(key);
        PetscOptionsSetValue(nullptr, option.c_str(), it->second.c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    int exitCode = 0;
    try {
        const auto manifest = readManifest(kDebugManifestPath);
        applyPetscOptionsFromManifest(manifest);

        char caseBuffer[PETSC_MAX_PATH_LEN] = "cases/simple_3d_cantilever";
        char outBuffer[PETSC_MAX_PATH_LEN] = "results/simple_3d_cantilever";
        if (auto it = manifest.find("case"); it != manifest.end()) {
            PetscStrncpy(caseBuffer, it->second.c_str(), sizeof(caseBuffer));
        }
        if (auto it = manifest.find("out"); it != manifest.end()) {
            PetscStrncpy(outBuffer, it->second.c_str(), sizeof(outBuffer));
        }
        PetscBool noHistory = PETSC_FALSE;
        if (auto it = manifest.find("no_history"); it != manifest.end() && truthy(it->second)) {
            noHistory = PETSC_TRUE;
        }
        if (auto it = manifest.find("write_history"); it != manifest.end()) {
            noHistory = truthy(it->second) ? PETSC_FALSE : PETSC_TRUE;
        }

        const std::string caseDir = caseBuffer;
        const std::string outDir = outBuffer;
        SolverOptions options;
        options.writeHistory = !static_cast<bool>(noHistory);

        CaseData data = readCaseData(caseDir);
        ensureDirectory(outDir);
        if (rank == 0) {
            std::cout << "stage1_petsc_backend: ndof=" << data.ndof
                      << " nelem=" << data.elements.size()
                      << " mpi_size=" << size << "\n";
        }

        SolveResult result = solveDynamicElastic(data, options);
        if (rank == 0) {
            writeVectorCsv(outDir + "/u_final.csv", result.uFinal);
            if (options.writeHistory) {
                writeMatrixCsvColumns(outDir + "/u_history.csv", result.uHistory);
            }
        }
        writeSolverInfo(outDir + "/solver_info.txt", size, rank, data.ndof,
                        static_cast<PetscInt>(data.elements.size()),
                        result.assemblyTime, result.solveTime,
                        result.kspIterations, result.kspResiduals);
    } catch (const std::exception& ex) {
        std::cerr << "stage1_petsc_backend error: " << ex.what() << "\n";
        exitCode = 1;
    }

    PetscFinalize();
    return exitCode;
}
