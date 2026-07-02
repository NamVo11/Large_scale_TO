# Multi-node Setup Scripts

This folder prepares `Large_scale_TO` to run the PETSc backend across multiple
Ubuntu machines with OpenMPI.

Run everything from the master node as user `votrongnam`.

## Files

- `hosts.example`: template hostfile.
- `check_master.sh`: check local MPI/PETSc/build/SSH prerequisites.
- `make_manifest.sh`: write `debug_manifest.txt` for a case/output/options.
- `run_local_smoke.sh`: run a small local MPI smoke test.
- `check_remote_hosts.sh`: verify passwordless SSH and remote environment.
- `sync_to_hosts.sh`: copy this project to the same path on remote nodes.
- `run_multinode.sh`: launch the backend with a hostfile.

## Typical Flow

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
bash scripts/check_master.sh
bash scripts/run_local_smoke.sh

cp scripts/hosts.example scripts/hosts.txt
# edit scripts/hosts.txt after the other machine's hostname/IP is known

bash scripts/check_remote_hosts.sh scripts/hosts.txt
bash scripts/sync_to_hosts.sh scripts/hosts.txt
bash scripts/run_multinode.sh scripts/hosts.txt 4 simple_3d_cantilever multinode_smoke_n4
```

