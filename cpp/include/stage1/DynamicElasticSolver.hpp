#pragma once

#include "stage1/CaseData.hpp"

#include <petscksp.h>
#include <string>
#include <vector>

struct SolveResult {
    std::vector<double> uFinal;
    std::vector<std::vector<double>> uHistory;
    std::vector<PetscInt> kspIterations;
    std::vector<PetscReal> kspResiduals;
    double assemblyTime = 0.0;
    double solveTime = 0.0;
};

SolveResult solveDynamicElastic(const CaseData& data, const SolverOptions& options);
