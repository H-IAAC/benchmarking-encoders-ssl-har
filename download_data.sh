#!/usr/bin/env bash
# download_data.sh — Download DAGHAR from Zenodo and prepare all data formats
#
# Usage:
#   bash download_data.sh                    # default: ./shared_data
#   bash download_data.sh /path/to/data      # custom root
#
# After running, the following directories will be ready:
#   shared_data/daghar/standardized_view/         (DAGHAR benchmark, 6 datasets)
#   shared_data/rodrigues_2024_datasets/1-1/      (per-user CSVs for SSL pretraining)
#   shared_data/xu_2023_datasets/1-1/             (NumPy arrays for TNC pretraining)
#
# Requires: wget, unzip, python (with pandas, numpy, minerva-ml)

set -euo pipefail

DATA_ROOT="${1:-./shared_data}"
DAGHAR_DIR="$DATA_ROOT/daghar/standardized_view"
ZENODO_ID="13987073"
# ZENODO_ID="13987073"

echo "============================================================"
echo "Step 1: Download DAGHAR from Zenodo (record $ZENODO_ID)"
echo "  target: $DAGHAR_DIR"
echo "============================================================"

# Check if already extracted (look for at least one expected dataset folder)
if [ -d "$DAGHAR_DIR/MotionSense" ] && [ -d "$DAGHAR_DIR/KuHar" ]; then
    echo "DAGHAR already present at $DAGHAR_DIR. Skipping download."
else
    mkdir -p "$DATA_ROOT/daghar"
    echo "Downloading standardized_view.zip ..."
    wget -q --show-progress \
        "https://zenodo.org/records/${ZENODO_ID}/files/standardized_view.zip?download=1" \
        -O "$DATA_ROOT/daghar/standardized_view.zip"
    echo "Extracting ..."
    unzip -q -o "$DATA_ROOT/daghar/standardized_view.zip" -d "$DATA_ROOT/daghar/"
    rm "$DATA_ROOT/daghar/standardized_view.zip"
    echo "Done. DAGHAR extracted to $DAGHAR_DIR"
fi

echo ""
echo "============================================================"
echo "Step 2: Prepare Rodrigues 2024 + Xu 2023 formats"
echo "  (required for SSL pretraining)"
echo "============================================================"
python prepare_data.py --data_root "$DATA_ROOT" --skip_download

echo ""
echo "============================================================"
echo "All data ready under $DATA_ROOT"
echo "You can now run the replication experiments."
echo "============================================================"
