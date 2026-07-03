# Multi-node Setup Scripts

This folder prepares `Large_scale_TO` to run the PETSc backend across multiple
Ubuntu machines with the synchronized MPICH/PETSc stack.

Run everything from the master node as user `votrongnam`.

## Files

- `hosts.example`: template hostfile.
- `check_master.sh`: check local MPI/PETSc/build/SSH prerequisites.
- `make_manifest.sh`: write `debug_manifest.txt` for a case/output/options.
- `run_mpich_multinode_smoke.sh`: run a hostname smoke test with MPICH.
- `run_mpich_multinode_backend.sh`: launch the backend with MPICH.
- `check_remote_hosts.sh`: verify passwordless SSH and remote environment.
- `sync_to_hosts.sh`: copy this project to the same path on remote nodes.
- `run_multinode.sh`: launch the backend with a hostfile.

## Typical Flow

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
bash scripts/run_mpich_multinode_smoke.sh localhost,192.168.11.21 2
bash scripts/make_manifest.sh simple_3d_cantilever multinode_smoke cg gamg 1e-10 1 20000
bash scripts/run_mpich_multinode_backend.sh localhost,192.168.11.21 2
```

Use `localhost,192.168.11.21` for the current two-node setup. The scripts set
`MPIEXEC_PORT_RANGE=45010:46000`, `MPICH_PORT_RANGE=45010:46000`, and
`HWLOC_COMPONENTS=-gl`.
