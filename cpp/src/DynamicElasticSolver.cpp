#include "DynamicElasticSolver.hpp"

#include "FEMH8.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

std::vector<double> gatherVecToRoot(Vec v) {
    MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(v));
    PetscMPIInt rank = 0;
    MPI_Comm_rank(comm, &rank);

    Vec vzero = nullptr;
    VecScatter scatter = nullptr;
    VecScatterCreateToZero(v, &scatter, &vzero);
    VecScatterBegin(scatter, v, vzero, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatter, v, vzero, INSERT_VALUES, SCATTER_FORWARD);

    std::vector<double> values;
    if (rank == 0) {
        PetscInt n = 0;
        VecGetSize(vzero, &n);
        values.resize(static_cast<std::size_t>(n));
        const PetscScalar* array = nullptr;
        VecGetArrayRead(vzero, &array);
        for (PetscInt i = 0; i < n; ++i) values[static_cast<std::size_t>(i)] = PetscRealPart(array[i]);
        VecRestoreArrayRead(vzero, &array);
    }

    VecScatterDestroy(&scatter);
    VecDestroy(&vzero);
    return values;
}

double loadScale(int step, int nstep) {
    if (nstep <= 0) return 0.0;
    constexpr double pi = 3.141592653589793238462643383279502884;
    return std::sin(pi * static_cast<double>(step) / static_cast<double>(nstep));
}

void setFixedValuesZero(Vec rhs, const std::vector<int>& fixedDofs) {
    std::vector<PetscInt> idx(fixedDofs.begin(), fixedDofs.end());
    std::vector<PetscScalar> zeros(idx.size(), 0.0);
    if (!idx.empty()) {
        VecSetValues(rhs, static_cast<PetscInt>(idx.size()), idx.data(), zeros.data(), INSERT_VALUES);
        VecAssemblyBegin(rhs);
        VecAssemblyEnd(rhs);
    }
}

} // namespace

SolveResult solveDynamicElastic(const CaseData& data, const SolverOptions& options) {
    SolveResult result;
    if (data.ndof <= 0) throw std::runtime_error("Invalid ndof");

    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const PetscInt ndof = data.ndof;
    const PetscInt nelem = static_cast<PetscInt>(data.elements.size());
    const PetscInt elemStart = (nelem * rank) / size;
    const PetscInt elemEnd = (nelem * (rank + 1)) / size;

    Mat K = nullptr;
    Mat M = nullptr;
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ndof, ndof,
                 96, nullptr, 96, nullptr, &K);
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ndof, ndof,
                 96, nullptr, 96, nullptr, &M);

    PetscLogDouble t0 = 0.0;
    PetscLogDouble t1 = 0.0;
    PetscTime(&t0);
    ElementMatrices cachedElementMatrices;
    if (data.uniformElements && nelem > 0) {
        cachedElementMatrices = computeH8ElementMatrices(data, 0);
    }
    for (PetscInt e = elemStart; e < elemEnd; ++e) {
        const ElementMatrices& em = data.uniformElements
            ? cachedElementMatrices
            : computeH8ElementMatrices(data, static_cast<int>(e));
        const double x = data.design[static_cast<std::size_t>(e)];
        const double eScale = effectiveE(data.material, x);
        const double rhoScale = effectiveRho(data.material, x);

        PetscInt rows[24];
        const auto& elem = data.elements[static_cast<std::size_t>(e)];
        for (int a = 0; a < 8; ++a) {
            rows[3 * a + 0] = 3 * elem[a] + 0;
            rows[3 * a + 1] = 3 * elem[a] + 1;
            rows[3 * a + 2] = 3 * elem[a] + 2;
        }

        PetscScalar ke[24 * 24];
        PetscScalar me[24 * 24];
        for (int i = 0; i < 24 * 24; ++i) {
            ke[i] = eScale * em.ke[static_cast<std::size_t>(i)];
            me[i] = rhoScale * em.me[static_cast<std::size_t>(i)];
        }
        MatSetValues(K, 24, rows, 24, rows, ke, ADD_VALUES);
        MatSetValues(M, 24, rows, 24, rows, me, ADD_VALUES);
    }
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    PetscTime(&t1);
    result.assemblyTime = t1 - t0;

    const double beta = 0.25;
    const double gamma = 0.5;
    const double dt = data.time.dt;
    const double a0c = 1.0 / (beta * dt * dt);

    Mat Keff = nullptr;
    MatDuplicate(K, MAT_COPY_VALUES, &Keff);
    MatAXPY(Keff, a0c, M, SAME_NONZERO_PATTERN);

    std::vector<PetscInt> fixed(data.fixedDofs.begin(), data.fixedDofs.end());
    IS fixedIs = nullptr;
    if (!fixed.empty()) {
        ISCreateGeneral(PETSC_COMM_WORLD, static_cast<PetscInt>(fixed.size()), fixed.data(), PETSC_COPY_VALUES, &fixedIs);
        MatZeroRowsColumnsIS(Keff, fixedIs, 1.0, nullptr, nullptr);
    }

    Vec u = nullptr;
    Vec v = nullptr;
    Vec acc = nullptr;
    Vec uPred = nullptr;
    Vec vPred = nullptr;
    Vec rhs = nullptr;
    Vec tmp = nullptr;
    Vec fext = nullptr;
    MatCreateVecs(Keff, &u, &rhs);
    VecDuplicate(u, &v);
    VecDuplicate(u, &acc);
    VecDuplicate(u, &uPred);
    VecDuplicate(u, &vPred);
    VecDuplicate(u, &tmp);
    VecDuplicate(u, &fext);
    VecSet(u, 0.0);
    VecSet(v, 0.0);
    VecSet(acc, 0.0);

    KSP ksp = nullptr;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, Keff, Keff);
    KSPSetType(ksp, KSPCG);
    PC pc = nullptr;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCGAMG);
    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    PetscTime(&t0);
    if (options.writeHistory) {
        result.uHistory.push_back(gatherVecToRoot(u));
    }

    for (int step = 1; step <= data.time.nstep; ++step) {
        if (rank == 0) {
            PetscPrintf(PETSC_COMM_SELF, "step %d/%d\n", step, data.time.nstep);
        }

        VecCopy(u, uPred);
        VecAXPY(uPred, dt, v);
        VecAXPY(uPred, dt * dt * (0.5 - beta), acc);

        VecCopy(v, vPred);
        VecAXPY(vPred, dt * (1.0 - gamma), acc);

        VecSet(fext, 0.0);
        const double scale = loadScale(step, data.time.nstep);
        if (rank == 0) {
            for (const auto& load : data.loads) {
                PetscInt dof = load.dof;
                PetscScalar value = load.amplitude * scale;
                VecSetValues(fext, 1, &dof, &value, ADD_VALUES);
            }
        }
        VecAssemblyBegin(fext);
        VecAssemblyEnd(fext);

        MatMult(M, uPred, tmp);
        VecCopy(fext, rhs);
        VecAXPY(rhs, a0c, tmp);
        setFixedValuesZero(rhs, data.fixedDofs);

        KSPSolve(ksp, rhs, u);

        PetscInt its = 0;
        PetscReal rnorm = 0.0;
        KSPGetIterationNumber(ksp, &its);
        KSPGetResidualNorm(ksp, &rnorm);
        result.kspIterations.push_back(its);
        result.kspResiduals.push_back(rnorm);

        VecCopy(u, tmp);
        VecAXPY(tmp, -1.0, uPred);
        VecCopy(tmp, acc);
        VecScale(acc, a0c);

        VecCopy(vPred, v);
        VecAXPY(v, gamma * dt, acc);

        if (options.writeHistory) {
            result.uHistory.push_back(gatherVecToRoot(u));
        }
    }
    PetscTime(&t1);
    result.solveTime = t1 - t0;
    result.uFinal = gatherVecToRoot(u);

    KSPDestroy(&ksp);
    VecDestroy(&u);
    VecDestroy(&v);
    VecDestroy(&acc);
    VecDestroy(&uPred);
    VecDestroy(&vPred);
    VecDestroy(&rhs);
    VecDestroy(&tmp);
    VecDestroy(&fext);
    if (fixedIs) ISDestroy(&fixedIs);
    MatDestroy(&Keff);
    MatDestroy(&K);
    MatDestroy(&M);

    return result;
}
