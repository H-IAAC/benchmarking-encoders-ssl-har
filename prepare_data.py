"""
prepare_data.py
===============
Download the DAGHAR benchmark datasets from Zenodo and create the two
preprocessed formats used during SSL pretraining:

  1. DAGHAR standardized view   →  shared_data/daghar/standardized_view/
  2. Rodrigues 2024 format      →  shared_data/rodrigues_2024_datasets/1-1/
  3. Xu 2023 format (TNC only)  →  shared_data/xu_2023_datasets/1-1/

Usage
-----
    python prepare_data.py                          # defaults
    python prepare_data.py --data_root /my/data     # custom root
    python prepare_data.py --skip_download          # skip Zenodo download
    python prepare_data.py --only_daghar            # stop after download/extract
    python prepare_data.py --only_rodrigues         # also create Rodrigues
    python prepare_data.py --only_xu                # only create Xu (needs Rodrigues)

After running this script the folder structure expected by all YAML configs
under base_configs/ will be in place.
"""

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset name mapping
# ---------------------------------------------------------------------------
# Keys are the DAGHAR standardized-view folder names.
# Values are the subfolder names used in Rodrigues / Xu formats.
DAGHAR_TO_INTERNAL = {
    "KuHar":           "kuhar",
    "MotionSense":     "motion",
    "UCI":         "uci",
    "WISDM":           "wisdm",
    "RealWorld_waist": "rw-waist",
    "RealWorld_thigh": "rw-tight",
}

# KuHar: drop subjects with fewer than this many timesteps before stacking
KUHAR_DROP_THRESHOLD = 361

SENSOR_PREFIXES = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
MAINTAIN_COLS   = ["user", "standard activity code"]


# ---------------------------------------------------------------------------
# Step 1 – Download DAGHAR from Zenodo
# ---------------------------------------------------------------------------

def _zenodo_file_urls(zenodo_id: str) -> list[tuple[str, str]]:
    """Query the Zenodo REST API and return [(filename, download_url), …]."""
    import json
    import urllib.request

    api_url = f"https://zenodo.org/api/records/{zenodo_id}"
    with urllib.request.urlopen(api_url) as resp:
        record = json.loads(resp.read())
    return [(f["key"], f["links"]["self"]) for f in record["files"]]


def download_daghar(zenodo_id: str, daghar_root: Path) -> None:
    """Download DAGHAR standardized view from Zenodo using wget and extract it."""
    print(f"\n{'='*60}")
    print(f"Step 1: Download DAGHAR from Zenodo record {zenodo_id}")
    print(f"  → target: {daghar_root}")
    print("="*60)

    # Check if already downloaded
    expected = [daghar_root / d for d in DAGHAR_TO_INTERNAL]
    if all(p.exists() for p in expected):
        print("  All DAGHAR datasets already present. Skipping download.")
        return

    download_tmp = daghar_root.parent / "_zenodo_tmp"
    download_tmp.mkdir(parents=True, exist_ok=True)

    print("  Fetching file list from Zenodo API ...")
    files = _zenodo_file_urls(zenodo_id)
    print(f"  Found {len(files)} file(s):")
    for name, url in files:
        print(f"    {name}")

    for name, url in files:
        dest = download_tmp / name
        if dest.exists():
            print(f"  [SKIP] {name} already downloaded.")
            continue
        print(f"  Downloading {name} ...")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest), url],
            check=True,
        )

    # Extract archives into daghar_root
    daghar_root.mkdir(parents=True, exist_ok=True)
    for archive in list(download_tmp.glob("*.zip")) + list(download_tmp.glob("*.tar.gz")):
        print(f"  Extracting {archive.name} ...")
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(daghar_root)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(daghar_root)

    shutil.rmtree(download_tmp, ignore_errors=True)

    # Verify
    missing = [d for d in DAGHAR_TO_INTERNAL if not (daghar_root / d).exists()]
    if missing:
        print("\n  WARNING: These dataset folders were not found after extraction:")
        for m in missing:
            print(f"    {daghar_root / m}")
        print("  Check the Zenodo archive structure and adjust DAGHAR_TO_INTERNAL if needed.")
    else:
        print("  All 6 DAGHAR datasets found.")


# ---------------------------------------------------------------------------
# Step 2 – Rodrigues 2024 format
# ---------------------------------------------------------------------------

def _numbered_cols(df: pd.DataFrame, prefix: str) -> list:
    return [c for c in df.columns if re.match(rf"^{re.escape(prefix)}-\d+$", c)]


def _linearize(df: pd.DataFrame) -> pd.DataFrame:
    """Unpack windowed columns (prefix-0 … prefix-N) into a flat continuous series.

    The DAGHAR standardized view stores each 60-sample window as one row with
    columns like accel-x-0, accel-x-1, …, accel-x-59.  This function flattens
    those back into a single column per sensor, replicating the metadata columns
    (user, standard activity code) for each time step.
    """
    linearized = {p: [] for p in SENSOR_PREFIXES}
    for m in MAINTAIN_COLS:
        linearized[m] = []

    first_prefix_cols = _numbered_cols(df, SENSOR_PREFIXES[0])
    n_per_row = len(first_prefix_cols)

    for _, row in df.iterrows():
        for prefix in SENSOR_PREFIXES:
            cols = _numbered_cols(df, prefix)
            linearized[prefix].extend(row[cols].tolist())
        for m in MAINTAIN_COLS:
            linearized[m].extend([row[m]] * n_per_row)

    out = pd.DataFrame(linearized)
    out["activity code"] = out["standard activity code"]
    return out


def create_rodrigues(daghar_root: Path, rodrigues_root: Path) -> None:
    """Convert DAGHAR standardized view → Rodrigues 2024 per-user CSV format."""
    print(f"\n{'='*60}")
    print("Step 2: Create Rodrigues 2024 format")
    print(f"  source : {daghar_root}")
    print(f"  target : {rodrigues_root}")
    print("="*60)

    for daghar_name, internal_name in DAGHAR_TO_INTERNAL.items():
        src = daghar_root / daghar_name
        out_root = rodrigues_root / internal_name
        print(f"\n  {daghar_name} → {out_root}")

        if not src.exists():
            print(f"    [SKIP] source not found: {src}")
            continue

        # If already done, skip
        if all((out_root / s).exists() for s in ("train", "val", "test")):
            first_train = list((out_root / "train").glob("*.csv"))
            if first_train:
                print("    Already exists, skipping.")
                continue

        for csv_file, split_name in [
            ("train.csv",      "train"),
            ("validation.csv", "val"),
            ("test.csv",       "test"),
        ]:
            fpath = src / csv_file
            if not fpath.exists():
                print(f"    [WARN] {csv_file} not found, skipping split '{split_name}'")
                continue

            df   = pd.read_csv(fpath)
            flat = _linearize(df)

            out_split = out_root / split_name
            out_split.mkdir(parents=True, exist_ok=True)

            for user_id, user_df in flat.groupby("user"):
                user_df.to_csv(out_split / f"{user_id}.csv", index=False)

            n_users = flat["user"].nunique()
            print(f"    {split_name:5s}: {len(df):6d} windows → "
                  f"{flat.shape[0]:8d} timesteps, {n_users} users")

    print("\n  Rodrigues 2024 format done.")


# ---------------------------------------------------------------------------
# Step 3 – Xu 2023 format (required by TNC)
# ---------------------------------------------------------------------------
#
# TNC (Temporal Neighborhood Coding) samples temporal neighborhoods on-the-fly
# from a CONTINUOUS recording.  The DAGHAR standardized view stores pre-cut
# 60-sample windows — once segmented the temporal position information is lost.
# The Rodrigues per-user CSVs restore continuity, but TNC's HarDataModule
# expects NumPy arrays of shape (n_subjects, n_timesteps, 6).
#
# This step reads the per-user CSVs produced in Step 2, aligns them to the
# minimum length, and stacks them into train_data.npy / val_data.npy /
# test_data.npy.  For KuHar, subjects shorter than KUHAR_DROP_THRESHOLD are
# dropped first because some users in that dataset have very few recordings.


def _read_split(split_dir: Path) -> tuple[dict, int]:
    """Read all per-user CSVs in split_dir. Returns (dfs, min_length)."""
    dsets = {}
    for f in split_dir.glob("*.csv"):
        dsets[f.stem] = pd.read_csv(f)
    if not dsets:
        raise FileNotFoundError(f"No CSVs found in {split_dir}")
    return dsets, min(len(d) for d in dsets.values())


def _stack(dsets: dict, length: int) -> np.ndarray:
    arrays = [
        d.iloc[:length][SENSOR_PREFIXES].values
        for d in dsets.values()
    ]
    return np.stack(arrays, axis=0).astype(np.float32)


def create_xu(rodrigues_root: Path, xu_root: Path) -> None:
    """Convert Rodrigues 2024 per-user CSVs → Xu 2023 NumPy arrays."""
    print(f"\n{'='*60}")
    print("Step 3: Create Xu 2023 format (TNC pretraining)")
    print(f"  source : {rodrigues_root}")
    print(f"  target : {xu_root}")
    print("="*60)

    for internal_name in DAGHAR_TO_INTERNAL.values():
        src = rodrigues_root / internal_name
        out = xu_root / internal_name / "processed"
        print(f"\n  {internal_name} → {out}")

        if not src.exists():
            print(f"    [SKIP] Rodrigues source not found: {src}")
            continue

        # Skip if already done
        if all((out / f).exists()
               for f in ("train_data.npy", "val_data.npy", "test_data.npy")):
            print("    Already exists, skipping.")
            continue

        train_d, min_train = _read_split(src / "train")
        val_d,   min_val   = _read_split(src / "val")
        test_d,  min_test  = _read_split(src / "test")

        min_len  = min(min_train, min_val)
        drop_thr = KUHAR_DROP_THRESHOLD if internal_name == "kuhar" else 0

        if drop_thr > 0:
            # KuHar: enforce minimum length and drop very short subjects
            if min_len < drop_thr:
                min_len = drop_thr
            if min_test < drop_thr:
                min_test = drop_thr
            train_d = {k: v for k, v in train_d.items() if len(v) >= min_len}
            val_d   = {k: v for k, v in val_d.items()   if len(v) >= min_len}
            test_d  = {k: v for k, v in test_d.items()  if len(v) >= min_test}

        train_arr = _stack(train_d, min_len)
        val_arr   = _stack(val_d,   min_len)
        test_arr  = _stack(test_d,  min_test)

        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "train_data.npy", train_arr)
        np.save(out / "val_data.npy",   val_arr)
        np.save(out / "test_data.npy",  test_arr)

        print(f"    train {train_arr.shape}  val {val_arr.shape}  "
              f"test {test_arr.shape}")

    print("\n  Xu 2023 format done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        default="./shared_data",
        help="Root directory for all data (default: ./shared_data)",
    )
    p.add_argument(
        "--daghar_zenodo_id",
        default="13987073",
        help="Zenodo record ID for the DAGHAR benchmark (default: 13987073)",
    )
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip the Zenodo download step (DAGHAR must already be in place)",
    )
    p.add_argument(
        "--only_daghar",
        action="store_true",
        help="Only download and extract DAGHAR; do not create Rodrigues or Xu formats",
    )
    p.add_argument(
        "--only_rodrigues",
        action="store_true",
        help="Download DAGHAR and create Rodrigues format; skip Xu",
    )
    p.add_argument(
        "--only_xu",
        action="store_true",
        help="Only create Xu format (Rodrigues must already exist)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    data_root      = Path(args.data_root).resolve()
    daghar_root    = data_root / "daghar" / "standardized_view"
    rodrigues_root = data_root / "rodrigues_2024_datasets" / "1-1"
    xu_root        = data_root / "xu_2023_datasets" / "1-1"

    print(f"Data root : {data_root}")
    print(f"DAGHAR    : {daghar_root}")
    print(f"Rodrigues : {rodrigues_root}")
    print(f"Xu 2023   : {xu_root}")

    if args.only_xu:
        create_xu(rodrigues_root, xu_root)
        return

    if not args.skip_download:
        download_daghar(args.daghar_zenodo_id, daghar_root)

    if args.only_daghar:
        return

    create_rodrigues(daghar_root, rodrigues_root)

    if not args.only_rodrigues:
        create_xu(rodrigues_root, xu_root)

    print("\nAll data preparation complete.")
    print("You can now run the replication experiments.")


if __name__ == "__main__":
    main()
