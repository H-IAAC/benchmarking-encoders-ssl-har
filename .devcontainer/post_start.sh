#!/bin/bash
# post_start.sh — runs automatically when the dev container starts.
# Installs Minerva from PyPI plus the orchestration dependencies for
# benchmarks/scripts.

set -e

pip install --upgrade pip setuptools

# -----------------------------------------------------------------------------
# Minerva (from PyPI)
# -----------------------------------------------------------------------------
echo "🎈 Installing Minerva"
pip install minerva==0.3.10b0

# -----------------------------------------------------------------------------
# Extra dependencies for the orchestration scripts in benchmarks/scripts
# -----------------------------------------------------------------------------
echo "🎈 Installing extra benchmark dependencies"
pip install -r /workspaces/benchmarking-encoders-ssl-har/requirements.txt
sudo pip install ray[default]==2.46.0
pip install git+https://github.com/KaiyangZhou/Dassl.pytorch
pip3 install opencv-python-headless
pip install --upgrade numba
pip uninstall numpy -y
pip install numpy==1.26.4
pip uninstall kaleido -y
pip install kaleido==0.2.1

echo "🎈 Everything installed"

# -----------------------------------------------------------------------------
# Quality-of-life tweaks
# -----------------------------------------------------------------------------
echo "🎈 Adding useful options...."

# tmux: 256-color, mouse on, ^B e/E to sync/unsync panes
echo -e "set -g default-terminal \"screen-256color\"\nset -g mouse on\nbind e setw synchronize-panes on\nbind E setw synchronize-panes off" >> ~/.tmux.conf

# bash prompt: short directory instead of full path
sed -i '/^\s*PS1.*\\w/s/\\w/\\W/g' ~/.bashrc

# Convenience symlinks for shared data/runs (if mounted)
if [ -d "/workspaces/shared/data" ]; then
    ln -sf "/workspaces/shared/data" /workspaces/benchmarking-encoders-ssl-har/shared_data
fi
if [ -d "/workspaces/shared/runs" ]; then
    ln -sf "/workspaces/shared/runs" /workspaces/benchmarking-encoders-ssl-har/shared_runs
fi

echo "Done 🎉 🎈"
exit 0
