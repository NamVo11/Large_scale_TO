# Stage 1 PETSc Backend on UbuntuD

Build:

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
PKG_CONFIG_PATH=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx/lib/pkgconfig \
cmake -S . -B build-mpich-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpicxx \
  -DPETSC_PKG=PETSc
cmake --build build-mpich-release -j
```

Run single rank:

```bash
bash scripts/make_manifest.sh simple_3d_cantilever local_smoke cg gamg 1e-10 1 20000
bash scripts/mpiexec_mpich_ch3_cxx.sh -n 1 build-mpich-release/stage1_petsc_backend
```

Run MPI across master + worker:

```bash
bash scripts/run_mpich_multinode_backend.sh localhost,192.168.11.21 8
```

The executable reads `debug_manifest.txt` from this directory.
See `README_VSCODE_WSL.md` for VS Code tasks and WSL user setup.


