#pragma once

#include "CaseData.hpp"

#include <petscksp.h>
#include <string>
#include <vector>

CaseData readCaseData(const std::string& caseDir);
void ensureDirectory(const std::string& path);
void writeVectorCsv(const std::string& path, const std::vector<double>& values);
void writeMatrixCsvColumns(const std::string& path, const std::vector<std::vector<double>>& columns);
void writeSolverInfo(const std::string& path, PetscInt size, PetscInt rank,
                     PetscInt ndof, PetscInt nelem, double assemblyTime,
                     double solveTime, const std::vector<PetscInt>& iterations,
                     const std::vector<PetscReal>& residuals);
