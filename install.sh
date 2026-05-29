#!/usr/bin/env bash
# Install all dependencies for the benchmarking-encoders-ssl-har project.
#
# Minerva pins an older pandas, so a plain `pip install -r requirements.txt`
# leaves an incompatible version behind. This script installs everything and
# then force-reinstalls the pandas version the orchestration scripts need.
#
# Usage (run inside your activated virtual environment):
#   ./install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PANDAS_VERSION="2.3.3"

echo ">> Installing requirements..."
python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ">> Pinning pandas==${PANDAS_VERSION} (overrides Minerva's dependency)..."
python -m pip install --force-reinstall --no-deps "pandas==${PANDAS_VERSION}"

echo ">> Verifying installation..."
python -c "import minerva; print('minerva:', minerva.__version__)"
python -c "import ray, pandasql, networkx; print('extras OK')"
python -c "import pandas; print('pandas:', pandas.__version__)"

echo ">> Done."
