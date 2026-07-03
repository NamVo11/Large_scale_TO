# VS Code / WSL setup

Use the normal WSL user:

```bash
whoami
# expected: votrongnam
```

Open this folder from WSL, not from the Windows `D:` mirror:

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
code .
```

The project is configured to use the synchronized MPICH/PETSc stack:

```text
MPICH: /home/votrongnam/opt/mpich-4.0-ch3-cxx
PETSc: /home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx
Build: /home/votrongnam/PROJECT/Large_scale_TO/build-mpich-release
```

Recommended VS Code tasks:

```text
CMake: build MPICH Release
Run: local 1 rank
Run: multinode smoke 2 ranks
Run: multinode large 8 ranks
```

For command-line runs:

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
bash scripts/make_manifest.sh simple_3d_cantilever vscode_simple_smoke cg gamg 1e-10 1 20000
bash scripts/mpiexec_mpich_ch3_cxx.sh -n 1 build-mpich-release/stage1_petsc_backend
bash scripts/run_mpich_multinode_backend.sh localhost,192.168.11.21 8
```

Notes:

- Do not use system `mpirun`, `/usr/bin/mpicxx`, or apt PETSc for this project.
- The launcher uses port range `45010:46000`.
- `HWLOC_COMPONENTS=-gl` is set in the scripts to avoid WSL/X11 initialization hangs.
- If VS Code opens as `root`, run `wsl --shutdown` from Windows PowerShell and reopen WSL.
