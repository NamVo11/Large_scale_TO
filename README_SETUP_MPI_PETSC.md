# MPI/PETSc Setup Guide

This guide starts from a clean Ubuntu machine and ends with a runnable
`Large_scale_TO` PETSc backend.

Use a normal Linux user, not `root`. The examples use:

```text
user: votrongnam
project: /home/votrongnam/PROJECT/Large_scale_TO
MPICH: /home/votrongnam/opt/mpich-4.0-ch3-cxx
PETSc: /home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx
```

Do not use system `/usr/bin/mpirun`, system `/usr/bin/mpicxx`, or apt PETSc for
this project. Use the project scripts.

## 1. Single Machine Setup

### 1.1 Install Ubuntu packages

Purpose: install compilers and build tools.

```bash
sudo apt update
sudo apt install -y \
  build-essential gcc g++ gfortran make cmake pkg-config git rsync \
  wget curl python3 gdb netcat-openbsd
```

Potential errors:

- `pkg-config: command not found`: install `pkg-config`.
- `cmake: command not found`: install `cmake`.
- Permission errors inside the project: do not build as `root`; fix ownership:

```bash
sudo chown -R "$USER:$USER" /home/votrongnam/PROJECT/Large_scale_TO
```

### 1.2 Build MPICH

Purpose: create one known MPI stack that can be copied unchanged to all nodes.

```bash
mkdir -p /home/votrongnam/src /home/votrongnam/opt
cd /home/votrongnam/src
wget -nc https://www.mpich.org/static/downloads/4.0/mpich-4.0.tar.gz
tar -xf mpich-4.0.tar.gz
mkdir -p mpich-4.0-build-ch3-cxx
cd mpich-4.0-build-ch3-cxx

../mpich-4.0/configure \
  --prefix=/home/votrongnam/opt/mpich-4.0-ch3-cxx \
  --disable-fortran \
  --disable-romio \
  --with-device=ch3:nemesis

make -j"$(nproc)"
make install
```

Check:

```bash
/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpiexec --version | head
/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpicxx -show
```

Potential errors:

- `mpicxx` missing: MPICH was configured with C++ disabled. Rebuild without
  `--disable-cxx`.
- Multi-node Hydra errors such as proxy ID mismatch: MPI versions differ across
  machines. Build MPICH once, then `rsync` the installed prefix to every worker.
- OpenMPI hangs under WSL mirrored networking: avoid system OpenMPI for this
  project; use this MPICH build.

### 1.3 Build PETSc

Purpose: build PETSc against the custom MPICH stack.

```bash
cd /home/votrongnam/src
wget -nc https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.19.6.tar.gz
tar -xf petsc-3.19.6.tar.gz
cd petsc-3.19.6

unset PETSC_DIR PETSC_ARCH
export PETSC_DIR=/home/votrongnam/src/petsc-3.19.6
export PETSC_ARCH=arch-mpich-ch3-cxx-opt

./configure \
  --prefix=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx \
  --with-mpi-dir=/home/votrongnam/opt/mpich-4.0-ch3-cxx \
  --with-debugging=0 \
  --with-shared-libraries=1 \
  --download-f2cblaslapack=1 \
  --with-x=0 \
  --with-batch=1 \
  COPTFLAGS="-O3 -march=native" \
  CXXOPTFLAGS="-O3 -march=native"

make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" install
```

Check:

```bash
PKG_CONFIG_PATH=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx/lib/pkgconfig \
pkg-config --modversion PETSc
```

Expected:

```text
3.19.6
```

Potential errors:

- PETSc configure hangs at `configureMPIEXEC`: this can happen in WSL/networked
  environments. Use `--with-batch=1` as shown, then validate runtime separately.
- PETSc finds system MPI instead of custom MPICH: make sure
  `--with-mpi-dir=/home/votrongnam/opt/mpich-4.0-ch3-cxx` is used.
- Missing BLAS/LAPACK: keep `--download-f2cblaslapack=1`, or install system
  BLAS/LAPACK and configure explicitly.

### 1.4 Build the backend

Purpose: compile `stage1_petsc_backend` with the custom MPICH/PETSc stack.

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO

PKG_CONFIG_PATH=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx/lib/pkgconfig \
cmake -S . -B build-mpich-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER=/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpicxx \
  -DPETSC_PKG=PETSc

cmake --build build-mpich-release -j
```

Potential errors:

- CMake finds apt PETSc/OpenMPI: check `PKG_CONFIG_PATH` and
  `CMAKE_CXX_COMPILER`.
- VS Code or CMake reports `EACCES`: the build folder is probably root-owned.

```bash
sudo chown -R votrongnam:votrongnam \
  /home/votrongnam/PROJECT/Large_scale_TO/build-mpich-release \
  /home/votrongnam/PROJECT/Large_scale_TO/.vscode
```

### 1.5 Run local smoke tests

Purpose: verify PETSc runtime and the backend on one machine.

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO

bash scripts/make_manifest.sh simple_3d_cantilever local_smoke cg gamg 1e-10 1 20000
bash scripts/mpiexec_mpich_ch3_cxx.sh -n 1 build-mpich-release/stage1_petsc_backend
```

Run multiple local ranks:

```bash
bash scripts/mpiexec_mpich_ch3_cxx.sh -n 4 build-mpich-release/stage1_petsc_backend
```

Potential errors:

- Program hangs in `MPI_Init` with stack containing `XOpenDisplay` or
  `hwloc_gl`: set `HWLOC_COMPONENTS=-gl`. The project scripts already do this.
- Wrong `mpirun`: use `scripts/mpiexec_mpich_ch3_cxx.sh`, not system `mpirun`.

## 2. Multi-node Setup

This section assumes:

```text
master IP: 192.168.11.20
worker IP: 192.168.11.21
user: votrongnam
```

Adjust the IPs for your network.

### 2.1 Set default WSL user

Purpose: prevent VS Code/build files from becoming root-owned.

On each WSL Ubuntu machine:

```bash
sudo sh -c 'cat > /etc/wsl.conf <<EOF
[boot]
systemd=true

[user]
default=votrongnam
EOF'
```

Then from Windows PowerShell:

```powershell
wsl --shutdown
```

Reopen Ubuntu and check:

```bash
whoami
```

Expected:

```text
votrongnam
```

Potential errors:

- VS Code opens as `root`: run `wsl --shutdown`, reopen WSL, and open the
  project from `/home/votrongnam/PROJECT/Large_scale_TO`.
- Existing root-owned files:

```bash
sudo chown -R votrongnam:votrongnam /home/votrongnam/PROJECT/Large_scale_TO
```

### 2.2 Configure WSL mirrored networking

Purpose: make WSL nodes reachable on the LAN.

On Windows, edit:

```text
C:\Users\<your-user>\.wslconfig
```

Recommended:

```ini
[wsl2]
networkingMode=mirrored
firewall=false
dnsTunneling=true
autoProxy=true
```

Apply:

```powershell
wsl --shutdown
```

Potential errors:

- Worker is not reachable from master: confirm both WSLs are using mirrored
  networking and check `ip addr`.
- SSH works but MPI rank-to-rank traffic hangs: firewall/port range is likely
  still blocked; see step 2.5.

### 2.3 Configure SSH from master to worker

Purpose: allow MPICH Hydra to launch remote ranks.

On master:

```bash
ssh-keygen -t rsa -b 4096
ssh-copy-id votrongnam@192.168.11.21
```

If worker SSH is exposed through Windows portproxy on port `2222`, create:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat >> ~/.ssh/config <<EOF
Host worker1
  HostName 192.168.11.21
  Port 2222
  User votrongnam
  IdentityFile ~/.ssh/id_rsa
  StrictHostKeyChecking accept-new
  ServerAliveInterval 30
  ServerAliveCountMax 3
EOF
chmod 600 ~/.ssh/config
```

Check:

```bash
ssh worker1 hostname
```

Potential errors:

- `Connection refused`: worker SSH service or Windows portproxy is not ready.
- Password prompt during MPI launch: SSH key auth is not configured correctly.
- `ssh` works manually but MPICH cannot use port 2222: use
  `scripts/ssh_worker_2222.sh`, which is already used by the project launchers.

### 2.4 Sync toolchain and project to worker

Purpose: keep MPICH/PETSc/project paths identical on every node.

From master:

```bash
rsync -az --delete /home/votrongnam/opt/mpich-4.0-ch3-cxx/ \
  worker1:/home/votrongnam/opt/mpich-4.0-ch3-cxx/

rsync -az --delete /home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx/ \
  worker1:/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx/

cd /home/votrongnam/PROJECT/Large_scale_TO
rsync -az --delete \
  --exclude .git \
  --exclude build \
  --exclude build-release \
  --exclude build-mpich-release \
  --exclude build-mpich-debug \
  --exclude results \
  ./ worker1:/home/votrongnam/PROJECT/Large_scale_TO/
```

Build on worker:

```bash
ssh worker1 '
cd /home/votrongnam/PROJECT/Large_scale_TO &&
PKG_CONFIG_PATH=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx/lib/pkgconfig \
cmake -S . -B build-mpich-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER=/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpicxx \
  -DPETSC_PKG=PETSc &&
cmake --build build-mpich-release -j
'
```

Potential errors:

- MPICH version mismatch: do not install separately on each worker; sync the same
  `/home/votrongnam/opt/mpich-4.0-ch3-cxx` directory.
- PETSc version mismatch: sync the PETSc install prefix too.
- Worker reads an old `debug_manifest.txt`: use
  `scripts/run_mpich_multinode_backend.sh`; it syncs the manifest before launch.

### 2.5 Open firewall ports

Purpose: allow Hydra control traffic and MPI rank-to-rank TCP traffic.

On each Windows host, run PowerShell as Administrator:

```powershell
New-NetFirewallRule `
  -DisplayName "WSL MPI Range 45000-46000" `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort 45000-46000
```

If Windows/WSL networking still blocks traffic, also add a Hyper-V firewall rule
as needed for your Windows version.

Check from master while worker listens:

```bash
# on worker WSL
nc -lk 45000

# on master WSL
nc -vz 192.168.11.21 45000
```

Potential errors:

- Master shows `SYN-SENT` to worker port `45000`: worker firewall is blocking
  inbound MPI traffic.
- `45000` is already occupied by a test listener: stop it:

```bash
sudo pkill -f "nc -k -l -p 45000"
```

- To avoid test listeners on `45000`, the project scripts use:

```text
MPIEXEC_PORT_RANGE=45010:46000
MPICH_PORT_RANGE=45010:46000
```

### 2.6 Run multi-node hostname smoke

Purpose: verify remote rank launch and host distribution.

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
bash scripts/run_mpich_multinode_smoke.sh localhost,192.168.11.21 2
```

Expected:

```text
DESKTOP-KADLI42
DESKTOP-9LN8UUJ
```

For 16 ranks on two 8-core machines:

```bash
bash scripts/run_mpich_multinode_smoke.sh localhost,192.168.11.21 16 | sort | uniq -c
```

Expected distribution:

```text
8 <master-hostname>
8 <worker-hostname>
```

Potential errors:

- SSH command hangs: check `scripts/ssh_worker_2222.sh` and `~/.ssh/config`.
- Hydra proxy errors: MPICH installation differs between nodes.
- Hostname smoke works but backend hangs: check PETSc, manifest sync, and
  firewall rank-to-rank ports.

### 2.7 Run multi-node backend smoke

Purpose: verify the PETSc backend across nodes.

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
bash scripts/make_manifest.sh simple_3d_cantilever multinode_smoke cg gamg 1e-10 1 20000
bash scripts/run_mpich_multinode_backend.sh localhost,192.168.11.21 2
cat results/multinode_smoke/solver_info.txt
```

Expected:

```text
mpi_size=2
ndof=60
nelem=4
```

Run a larger case:

```bash
bash scripts/make_manifest.sh large_h8_240x60x24 run_np8 cg gamg 1e-8 1 20000
bash scripts/run_mpich_multinode_backend.sh localhost,192.168.11.21 8
cat results/run_np8/solver_info.txt
```

Potential errors:

- Backend uses only one node: check host list and `NP`.
- `Missing manifest`: run `scripts/make_manifest.sh` before launching.
- `Missing executable`: build `build-mpich-release` on master and worker.
- `Unknown PC type` or strange PETSc options: check `debug_manifest.txt`; wrong
  shell quoting can shift arguments.

## 3. VS Code Notes

Open from WSL:

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
code .
```

Useful tasks:

```text
Ctrl+Shift+P -> Tasks: Run Task -> CMake: build MPICH Debug
Ctrl+Shift+P -> Tasks: Run Task -> CMake: build MPICH Release
Ctrl+Shift+P -> Tasks: Run Task -> Run: multinode smoke 2 ranks
Ctrl+Shift+P -> Tasks: Run Task -> Run: multinode large 8 ranks
```

Debug one machine, one rank:

```text
Run and Debug -> Debug: stage1_petsc_backend local
```

Potential errors:

- Debug runs through without stopping: use `build-mpich-debug`, set breakpoints,
  and ensure `stopAtEntry=true`.
- VS Code CMake writes `EACCES`: fix ownership of `.vscode` and build folders.

## 4. Quick Validation Checklist

```bash
whoami
/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpiexec --version | head
PKG_CONFIG_PATH=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx/lib/pkgconfig pkg-config --modversion PETSc
```

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
bash scripts/make_manifest.sh simple_3d_cantilever local_check cg gamg 1e-10 1 20000
bash scripts/mpiexec_mpich_ch3_cxx.sh -n 1 build-mpich-release/stage1_petsc_backend
bash scripts/run_mpich_multinode_smoke.sh localhost,192.168.11.21 2
bash scripts/run_mpich_multinode_backend.sh localhost,192.168.11.21 2
```

git add .gitignore
git commit -m"--"
git push origin main

git fetch origin
git reset --hard origin/main