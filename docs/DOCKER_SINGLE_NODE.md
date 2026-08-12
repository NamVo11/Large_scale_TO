# Docker Single-Node Workflow

Purpose: run `Large_scale_TO` on a new Windows machine with Docker Desktop,
without installing MPICH, PETSc, MKL, MUMPS, GCC, CMake, or Python tooling on
the host.

The Docker image contains the fixed build/runtime environment. The Git checkout
stays on the host and is mounted into the container when VS Code tasks run.

## 1. Layout

```text
Docker image
  Ubuntu 22.04
  GCC 11 and GCC 13
  Python 3 + pip/venv/numpy/matplotlib
  MPICH 4.0 for gcc11 and gcc13
  Intel oneAPI MKL
  PETSc 3.19.6 + MKL/PARDISO + MUMPS for gcc11 and gcc13

Host Git repo
  /path/to/Large_scale_TO
  source code, input.json files, build-docker-* outputs

Running container
  image filesystem
  + host repo mounted at /workspace/Large_scale_TO
```

## 2. VS Code Tasks

The Docker workflow is declared directly in `.vscode/tasks.json`, matching the
style of `Run batch: 11111`.

Available Docker tasks:

```text
Docker: build image env
Docker: smoke test env
Docker: configure stage2 Release
Docker: build stage2 Release
Docker Run batch: 11111
```

The active configuration is GCC13. GCC11 lines are kept beside the GCC13 lines
as comments, so switching toolchain is the same manual edit style as the
existing native tasks.

The run task uses:

```text
INPUT_JSONS=case_study/Engine_bracket/mesh_size_1.5_new/input.json
```

To run multiple models sequentially, put them in the same `INPUT_JSONS` value:

```text
INPUT_JSONS=case_study/model1/input.json case_study/model2/input.json case_study/model3/input.json
```

Runtime model settings should be edited in `input.json`, especially:

```text
Run.mpi.np
Run.mpi.hosts
Run.threads
Run.linear_backend
Solver
Analysis[].ROM
Output
```

## 3. First Run On A Machine With Docker Desktop

From VS Code:

```text
Terminal -> Run Task -> Docker: build image env
Terminal -> Run Task -> Docker: smoke test env
Terminal -> Run Task -> Docker: build stage2 Release
Terminal -> Run Task -> Docker Run batch: 11111
```

The image build is heavy because it builds both GCC11 and GCC13 MPICH/PETSc
stacks. Ordinary source changes only need the build/run tasks, not an image
rebuild.

## 4. Build Outputs

GCC13 output:

```text
build-docker-release-gcc13/stage2_dynamic_forward
```

GCC11 output, if the task comments are switched:

```text
build-docker-release-gcc11/stage2_dynamic_forward
```

These executables are intended to run inside the Docker environment because
they link to PETSc/MPICH/MKL inside the image.

## 5. Move The Image To A New Windows Machine

Option A: save/load a tar file.

```bash
docker save large-scale-to-env:gcc11-gcc13-petsc-mkl-mumps \
  -o large-scale-to-env.tar
```

On the new Windows machine:

```bash
docker load -i large-scale-to-env.tar
```

Option B: use a registry.

```bash
docker tag large-scale-to-env:gcc11-gcc13-petsc-mkl-mumps <registry>/large-scale-to-env:gcc11-gcc13-petsc-mkl-mumps
docker push <registry>/large-scale-to-env:gcc11-gcc13-petsc-mkl-mumps
```

Then on the new machine:

```bash
docker pull <registry>/large-scale-to-env:gcc11-gcc13-petsc-mkl-mumps
```

## 6. Daily Workflow

```text
git pull
edit input.json if needed
VS Code task: Docker: build stage2 Release
VS Code task: Docker Run batch: 11111
```

No Docker image rebuild is needed after ordinary C++ or JSON changes.
