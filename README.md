Thí Branch is a vesion that devide in to 2 backends:
    1. Eigen for: matrix/vector manipulation, ROM
      PETSc for FOM solver, MPI
    
    2. Eigen for all, NO PETSc

These two backends are arranged in tow separate folder: Eigen_bacnkend and Petsc_backend.
To decide one of them, set parameter "LINEAR_BACKEND" = "PETSc" or "Eigen" in task.json file