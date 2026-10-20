# Docker Desktop setup on this PC

The checked machine is an ASUS TUF Gaming FX505DT, AMD Ryzen 5 3550H,
running Windows 10 Home 22H2 build 19045. Windows reports hardware virtualization
as **disabled**, SLAT support present, and no active hypervisor. WSL 3.0.3.0 is
installed; both Windows Subsystem for Linux and Virtual Machine Platform features
are enabled. The Server service is running with automatic startup.

Docker Desktop 4.95.0 is installed successfully in per-user mode with the WSL 2
backend. Its official installer checksum and Docker Inc signature were verified;
the installer exited with code 0. Its Linux engine
cannot start until firmware virtualization is enabled and Windows is restarted.

## Enable firmware virtualization

After saving your work and finishing any running checks:

1. Shut down the laptop. Hold **F2** while pressing Power, then release F2 once
   the BIOS screen appears. Alternatively, Windows **Advanced startup → Restart
   now → Troubleshoot → Advanced options → UEFI Firmware Settings → Restart**
   enters firmware setup.
2. Open **Advanced Mode** if necessary (often F7), find **SVM Mode** under the
   Advanced settings, and set it to **Enabled**. Menu placement can differ by
   firmware version.
3. Save changes and exit using the firmware's save/exit command, then boot Windows.
4. In **Task Manager → Performance → CPU**, confirm **Virtualization: Enabled**.

These steps follow ASUS's [BIOS entry instructions](https://www.asus.com/support/faq/1008829/)
and [AMD virtualization instructions](https://www.asus.com/us/support/faq/1043992/).
The firmware change needs to be performed on the laptop; the assignment scripts
do not change BIOS settings or restart Windows.

## Finish Docker startup

Open Docker Desktop after the firmware change. If its subscription agreement is
displayed, review it and accept only if you agree. Select the WSL 2/Linux backend.
An additional Ubuntu distribution is not required for Docker Desktop's own engine.

Open a new PowerShell window and verify:

```powershell
docker version
docker info --format '{{.OSType}}'
```

The first command must report both client and server. The second must return
`linux`. If `docker` is not on the current terminal PATH, a new terminal normally
loads the installer-updated PATH. Per-user installations place Docker Desktop
under `%LOCALAPPDATA%/Programs/DockerDesktop`; the CLI is in its `resources/bin` folder.

Then run the assignment's actual container verification from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/docker_smoke.py
```

This builds and runs the packaged API and saves evidence. Until that command
succeeds, container execution remains pending even if Docker Desktop is installed.
See [API and Docker instructions](api_docker.md).

Official sources: [Docker Windows installation and requirements](https://docs.docker.com/desktop/setup/install/windows-install/),
[WSL backend](https://docs.docker.com/desktop/features/wsl/),
and [release notes/checksums](https://docs.docker.com/desktop/release-notes/).
