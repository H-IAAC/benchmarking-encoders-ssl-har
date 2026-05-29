# Docker Dev Container Setup

This guide explains how to set up the development environment using Docker and VSCode Dev Containers for the **Benchmarking Encoders and SSL for HAR** project.

> **Note:** `shared_data/` is not mounted by default. Data can be added later — see [Downloading Data](#downloading-data).

---

## Prerequisites

1. Install [Docker](https://docs.docker.com/get-docker/) on the **remote machine**.
2. Install [Visual Studio Code](https://code.visualstudio.com/) on your **local machine**.
3. Install the [Remote Development Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) on your **local machine**.

We use the following terms throughout this guide:
- **local machine** — your personal computer (Windows/macOS/Linux).
- **remote machine** — the server where Docker runs, accessed via SSH (e.g., `ssh <username>@node01`).
- **remote container** — the Docker container running on the remote machine.

---

## Step 1 — SSH Key for GitHub (remote machine, once only)

Since the project uses private repositories, you need a GitHub SSH key on the remote machine.

**1.1** Generate an SSH key pair on the **remote machine**:

```bash
ssh-keygen -t ed25519 -C "github key" -f ~/.ssh/id_ed25519_github
```

**1.2** Print the public key and copy its output:

```bash
cat ~/.ssh/id_ed25519_github.pub
```

Add it to your GitHub account at [github.com/settings/keys](https://github.com/settings/keys) → **New SSH key** → Key Type: *Authentication Key*.

**1.3** Register the key with the SSH agent:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github
```

**1.4** Verify the connection:

```bash
ssh -T git@github.com
# Expected: Hi <username>! You've successfully authenticated...
```

---

## Step 2 — Clone the Repository (remote machine)

```bash
git clone git@github.com:H-IAAC/benchmarking-encoders-ssl-har.git ~/validate_paper
```

The `.devcontainer/` folder inside the repository already contains the `Dockerfile`, `devcontainer.json`, and `post_start.sh` needed for the next steps. No additional configuration script is required.

---

## Step 3 — Remove Any Stale Container (remote machine)

If a container named `home-hiaac-m4-<username>` already exists from a previous session, remove it first:

```bash
docker rm -f home-hiaac-m4-$(whoami)
```

Skip this step if no container exists yet.

---

## Step 4 — Open in VSCode and Build the Container (local machine)

**4.1** Open VSCode and connect to the remote machine via the Remote Development extension (bottom-left corner → *Connect to Host*).

**4.2** Once connected, open the cloned folder:
`File → Open Folder… → ~/validate_paper`

**4.3** VSCode will detect the `.devcontainer/` folder and prompt you to reopen in a container. Click **Reopen in Container**. Alternatively, open the Command Palette (`Ctrl+Shift+P`) and run:

```
Dev Containers: Rebuild and Reopen in Container
```

The first build will take several minutes as it pulls the base image and installs dependencies.

**4.4** Once the container starts, the `post_start.sh` script runs automatically. It installs `minerva` and all required Python packages. You can monitor progress in the terminal panel.

---

## Step 5 — Verify the Setup (inside container)

Open a terminal inside the container (`Terminal → New Terminal`) and check:

```bash
# Confirm Python environment
python -c "import torch; print(torch.__version__)"

# Confirm Minerva is installed
python -c "import minerva; print('Minerva OK')"

# Confirm GPU access
nvidia-smi
```

Also verify GitHub SSH still works (useful if you need to pull/push from within the container):

```bash
ssh -T git@github.com
```

---

## Downloading Data

Data is **not** included in the repository. After the container is running, download and prepare all datasets with:

```bash
cd /workspaces/validate_paper
bash download_data.sh
```

This will populate `shared_data/` with:

```
shared_data/
├── daghar/standardized_view/          # DAGHAR benchmark (6 datasets)
├── rodrigues_2024_datasets/1-1/       # Per-user CSVs for SSL pretraining
└── xu_2023_datasets/1-1/             # NumPy arrays for TNC pretraining
```

See [`prepare_data.py`](prepare_data.py) for advanced options (custom root, skip download, partial preparation).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| *Container name already in use* | Run `docker rm -f home-hiaac-m4-$(whoami)` on the remote machine, then rebuild. |
| *post_start.sh: No such file or directory* | Confirm `.devcontainer/post_start.sh` exists and is executable (`chmod +x .devcontainer/post_start.sh`). |
| *GPU not detected inside container* | Ensure the remote machine has `nvidia-docker` / the NVIDIA Container Toolkit installed. |

---

## Notes

- The container is configured with `--gpus all`, `--ipc host`, and `--network host` for GPU and distributed training workloads.
- For a deeper explanation of the dev container workflow, see [github.com/otavioon/container-workspace](https://github.com/otavioon/container-workspace). Kudos to @otavioon for the original guide.