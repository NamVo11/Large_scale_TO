#pragma once

#include <array>
#include <string>
#include <vector>

struct MaterialData {
    double E0 = 1.0;
    double Emin = 1e-6;
    double nu = 0.3;
    double rho0 = 1.0;
    double rhomin = 1e-12;
    double penalE = 3.0;
    double penalRho = 1.0;
};

struct TimeData {
    double dt = 1.0;
    int nstep = 1;
};

struct LoadEntry {
    int dof = 0;           // zero-based global dof
    double amplitude = 0;  // scalar amplitude
};

struct CaseData {
    std::vector<std::array<double, 3>> nodes;
    std::vector<std::array<int, 8>> elements; // zero-based node ids
    std::vector<int> fixedDofs;               // zero-based global dofs
    std::vector<LoadEntry> loads;
    std::vector<double> design;
    MaterialData material;
    TimeData time;
    int ndof = 0;
    bool uniformElements = true;
};

struct SolverOptions {
    bool writeHistory = true;
};
