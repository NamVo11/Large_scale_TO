# VS Code Task And Debug Workflow

This project should be run from VS Code in WSL. Treat terminal commands as
support tools only, mainly for cleanup and recovery.

## 1. Open the correct folder

- Use **Remote - WSL**
- Open:
  `/home/votrongnam/PROJECT/Large_scale_TO`
- Do not open:
  `/mnt/d/...`

Quick terminal check inside VS Code:

```bash
pwd
ls CMakeLists.txt
```

Expected folder:

```text
/home/votrongnam/PROJECT/Large_scale_TO
```

## 2. Clean old MPI and solver processes before every run

In the VS Code terminal, run:

```bash
bash scripts/kill_stage2_processes.sh
```

If you are preparing a multi-node run, do the same on the worker machine too.

This script only kills the project runtime processes:

- `stage2_dynamic_forward`
- `mpiexec`
- `hydra_pmi_proxy`
- `hydra_nameserver`

## 3. Build from VS Code tasks

Open:

- `Terminal -> Run Task`

Use these tasks:

- `CMake: build stage2 Debug`
- `CMake: build stage2 Release`

Notes:

- The build tasks already call the matching configure task first.
- Use `Debug` for breakpoints and `Release` for timing or larger runs.

## 4. Run from VS Code tasks

Open:

- `Terminal -> Run Task`

Use these tasks:

- `Run: single node single rank`
- `Run: single node multi_rank`
- `Run: multi_node multi_rank`

What they mean:

- `Run: single node single rank`
  - 1 MPI rank
  - direct solver setup in the task
  - best for simple validation

- `Run: single node multi_rank`
  - multiple MPI ranks on one machine
  - iterative solver setup in the task

- `Run: multi_node multi_rank`
  - ranks split across master and worker
  - current task uses:
    - `HOSTS=localhost,192.168.11.21`
    - `NP=2`

## 5. Debug from VS Code launch

Open:

- `Run and Debug`

Use:

- `Debug: single node single rank`

This launch is for:

- one machine
- one rank
- direct debugging with `gdb`

Important:

- It does **not** launch through `mpiexec`
- It uses the debug executable:
  - `${workspaceFolder}/build-mpich-debug/stage2_dynamic_forward`

So before using it:

1. open the correct WSL folder
2. run `bash scripts/kill_stage2_processes.sh`
3. run `CMake: build stage2 Debug`
4. start `Debug: single node single rank`

## 6. Clean environment policy

The project runtime environment is centralized in:

- [scripts/env_petsc_mpi.sh](//wsl.localhost/UbuntuD/home/votrongnam/PROJECT/Large_scale_TO/scripts/env_petsc_mpi.sh)

The MPI scripts already source this file:

- [scripts/mpiexec_mpich_ch3_cxx.sh](//wsl.localhost/UbuntuD/home/votrongnam/PROJECT/Large_scale_TO/scripts/mpiexec_mpich_ch3_cxx.sh)
- [scripts/run_stage2_case.sh](//wsl.localhost/UbuntuD/home/votrongnam/PROJECT/Large_scale_TO/scripts/run_stage2_case.sh)
- [scripts/run_stage2_multinode.sh](//wsl.localhost/UbuntuD/home/votrongnam/PROJECT/Large_scale_TO/scripts/run_stage2_multinode.sh)

Because of that, normal task runs should not depend on old shell variables,
`.bashrc`, or previously exported PETSc/MPI paths.

## 7. When a task or debug session behaves strangely

Do this in order:

1. Confirm the folder is `/home/votrongnam/PROJECT/Large_scale_TO`
2. Run:
   ```bash
   bash scripts/kill_stage2_processes.sh
   ```
3. Re-run the relevant build task
4. Start the run/debug task again

## 8. Recovery checks

Use these only if VS Code behavior still does not make sense.

Check the binary exists:

```bash
ls build-mpich-debug/stage2_dynamic_forward
ls build-mpich-release/stage2_dynamic_forward
```

Check local MPI bootstrap:

```bash
/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpiexec -iface eth1 -n 1 /bin/hostname
```

Check worker network identity:

```bash
scripts/ssh_worker_2222.sh 192.168.11.21 'hostname; ip -4 addr show dev eth1'
```
