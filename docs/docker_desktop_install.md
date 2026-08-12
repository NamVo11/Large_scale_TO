# Docker Desktop Install Notes For Large_scale_TO

Purpose: prepare a new Windows machine so `Large_scale_TO` can build and run
through Docker Desktop without installing MPICH, PETSc, MKL, MUMPS, GCC, CMake,
or Python toolchains on the host.

## 1. Install Docker Desktop

1. Download and install Docker Desktop for Windows.
2. Use the WSL 2 backend.
3. Start Docker Desktop and wait until the engine is running.

## 2. Enable WSL Integration

Open Docker Desktop:

```text
Settings -> Resources -> WSL integration
```

Enable:

```text
Enable integration with my default WSL distro
```

Then enable the distro that contains the repo, for example:

```text
UbuntuD
```

Click `Apply & Restart`.

## 3. Restart WSL

From Windows PowerShell:

```powershell
wsl --shutdown
```

Open the target distro again, then check:

```bash
id
docker --version
docker ps
```

Expected:

```text
id includes group docker
docker --version prints a Docker version
docker ps lists containers without permission errors
```

## 4. Credential Helper Fix In WSL

If Docker in WSL fails with:

```text
docker-credential-desktop.exe: exec format error
```

edit:

```text
~/.docker/config.json
```

and use:

```json
{}
```

This avoids the Windows credential helper from being called by the Linux Docker
CLI. Public image pulls do not need a credential helper.

## 5. Verify Pull And Run

In WSL:

```bash
docker run --rm hello-world
```

Expected:

```text
Hello from Docker!
```

## 6. If Integration Looks Enabled But Docker Is Missing

Symptoms:

```text
The command 'docker' could not be found in this WSL 2 distro.
/mnt/wsl/docker-desktop is missing
docker-desktop is Stopped in wsl -l -v
```

Check from PowerShell:

```powershell
wsl -l -v
```

If `docker-desktop` is stopped or Docker Desktop shows an error dialog:

1. In Docker Desktop, click `Restart the WSL integration` if offered.
2. If that does not recover, quit Docker Desktop from the system tray.
3. In PowerShell:

   ```powershell
   wsl --shutdown
   taskkill /IM "Docker Desktop.exe" /F
   taskkill /IM "com.docker.backend.exe" /F
   ```

4. Start Docker Desktop again.
5. Wait until the engine is running.
6. Reopen WSL and run:

   ```bash
   docker ps
   ```

## 7. Build Large_scale_TO Image

From the repo root in WSL:

```bash
docker build \
  -t large-scale-to-env:gcc11-gcc13-petsc-mkl-mumps \
  -f docker/Dockerfile \
  --build-arg BUILD_JOBS=8 \
  .
```

The image contains both GCC11 and GCC13 MPICH/PETSc stacks. The source is not
copied into the image; only the Docker build files are used because `.dockerignore`
limits the build context.

## 8. Run VS Code Tasks

After the image is built:

```text
Docker: smoke test env
Docker: build stage2 Release
Docker Run batch: 11111
```

The Docker run tasks mount the whole repo:

```text
${workspaceFolder}:/workspace/Large_scale_TO
```

So source, `input.json`, case files, and build outputs remain on the host.
