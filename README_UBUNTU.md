# Stage 1 PETSc Backend on UbuntuD

Build:

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j 4
```

Run single rank:

```bash
./build/stage1_petsc_backend
```

Run MPI on one machine:

```bash
mpirun -n 2 ./build/stage1_petsc_backend
mpirun -n 4 ./build/stage1_petsc_backend
```

The executable reads `debug_manifest.txt` from this directory.
