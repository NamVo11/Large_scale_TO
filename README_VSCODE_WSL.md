# VS Code / WSL Workflow

This file is the short version. For the full workflow, read:

- [docs/CLEAN_ENV_AND_RUN.md](//wsl.localhost/UbuntuD/home/votrongnam/PROJECT/Large_scale_TO/docs/CLEAN_ENV_AND_RUN.md)

## Open the right workspace

- Start VS Code with **Remote - WSL**
- Open:
  `/home/votrongnam/PROJECT/Large_scale_TO`
- Do not open:
  `/mnt/d/...`

## Before any run or debug

Open the VS Code terminal and run:

```bash
bash scripts/kill_stage2_processes.sh
```

If the next run is multi-node, do the same on the worker.

## Build tasks

Use:

- `Terminal -> Run Task -> CMake: build stage2 Debug`
- `Terminal -> Run Task -> CMake: build stage2 Release`

## Run tasks

Use:

- `Terminal -> Run Task -> Run: single node single rank`
- `Terminal -> Run Task -> Run: single node multi_rank`
- `Terminal -> Run Task -> Run: multi_node multi_rank`

## Debug

Use:

- `Run and Debug -> Debug: single node single rank`

Recommended order:

1. `bash scripts/kill_stage2_processes.sh`
2. `CMake: build stage2 Debug`
3. `Debug: single node single rank`

## Environment source of truth

Shared runtime environment:

- [scripts/env_petsc_mpi.sh](//wsl.localhost/UbuntuD/home/votrongnam/PROJECT/Large_scale_TO/scripts/env_petsc_mpi.sh)

This is the file that the project scripts use for:

- PETSc path
- MPICH path
- MKL path
- MPI port range
- library path

That is why task runs should stay consistent across master and worker.
