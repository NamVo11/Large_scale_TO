# Multi-node Setup Scripts

These scripts support the current `stage2_dynamic_forward` workflow on one or
more Ubuntu machines with the shared MPICH/PETSc/MKL/MUMPS stack.

Run everything from the master node as user `votrongnam`.

## Main files

- `env_petsc_mpi.sh`: shared runtime environment.
- `check_master.sh`: verify local toolchain and build.
- `check_remote_hosts.sh`: verify SSH and runtime tools on workers.
- `sync_to_hosts.sh`: sync this project to workers at the same path.
- `run_stage2_case.sh`: main case launcher used by VS Code tasks.
- `run_stage2_multinode.sh`: multi-node launcher used by `run_stage2_case.sh`.
- `ssh_worker_2222.sh`: shared SSH wrapper for worker access on port `2222`.
- `hosts.example`: editable host list for helper scripts.

## Typical flow

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
bash scripts/check_master.sh
bash scripts/check_remote_hosts.sh
bash scripts/sync_to_hosts.sh
bash scripts/run_stage2_case.sh
```

For the current two-node setup, use `localhost,192.168.11.21` in `HOSTS` and
keep `MPIEXEC_PORT_RANGE=40000:41000`, `MPICH_PORT_RANGE=40000:41000`, and
`HWLOC_COMPONENTS=-gl`.
