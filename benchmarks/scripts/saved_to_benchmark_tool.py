"""Convert a `saved_metrics_*.csv` file (produced by `summarizer.py`) into the
CSV format expected by `benchmark_tool.html`, and optionally compare it against
the reference results in `performance_data.json`.

Output schema: technique,dataset,backbone,refinement,spc,mean,std

SPC (samples per class) is read from the *finetune* row's `data/override_id`,
found by following the evaluate row's `backbone/load_from_uid` back to its
parent finetune execution. SPC convention matches the benchmark tool:
    1, 5, 10, 25, 50, 100, 200, 1000   (1000 = 100% of the dataset)

Usage:
    python saved_to_benchmark_tool.py saved_metrics_tfc_resnet_cnn_transformers.csv
    python saved_to_benchmark_tool.py <input.csv> --output <out.csv>
    python saved_to_benchmark_tool.py <input.csv> --benchmark performance_data.json \\
            --compare-csv compare.csv --compare-fig compare.png
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SPC_VALUES = [1, 5, 10, 25, 50, 100, 200, 1000]


BACKBONE_MAP = {
    "cnnpff": "CNN-PFF",
    "rnn": "RNN",
    "resnet": "ResNet-1D",
    "transformer": "IMU Transformer",
    "ts2vec": "TS2Vec Encoder",
    "harcnn": "TS-TCC Encoder",
    "default": "TS-TCC Encoder",
    "resnetse5": "ResNet-SE-5",
}

TECHNIQUE_MAP = {
    "tfc": "TFC",
    "tnc": "TNC",
    "diet": "Diet",
    "lfr": "LFR",
}

DATASET_MAP = {
    "kuhar": "KH",
    "motionsense": "MS",
    "rw_thigh": "RW-Thigh",
    "rw_waist": "RW-Waist",
    "uci": "UCI",
    "wisdm": "WISDM",
    "hapt": "HAPT",
    "recodgait": "RecodGait",
}

OVERRIDE_TO_SPC = {
    "multimodal_samples_001": 1,
    "multimodal_samples_005": 5,
    "multimodal_samples_010": 10,
    "multimodal_samples_025": 25,
    "multimodal_samples_050": 50,
    "multimodal_samples_100": 100,
    "multimodal_samples_200": 200,
    "multimodal_perc_100": 1000,
}


def _lookup_spc(override_id) -> int | None:
    """Map a `data/override_id` to SPC. Tries an exact match first, then
    strips a trailing `_<single-digit>` run-replica suffix (e.g. `_2`, `_3`)."""
    if pd.isna(override_id):
        return None
    s = str(override_id)
    if s in OVERRIDE_TO_SPC:
        return OVERRIDE_TO_SPC[s]
    stripped = re.sub(r"_\d$", "", s)
    return OVERRIDE_TO_SPC.get(stripped)


def extract_backbone(model_name: str) -> str | None:
    """`tfc_resnetse5` -> `ResNet-SE-5`."""
    if pd.isna(model_name):
        return None
    # The encoder name is the suffix after the technique prefix.
    for tech in TECHNIQUE_MAP:
        if model_name.startswith(f"{tech}_"):
            enc = model_name[len(tech) + 1 :]
            return BACKBONE_MAP.get(enc, enc)
    return BACKBONE_MAP.get(model_name, model_name)


def extract_technique(model_name: str) -> str:
    if pd.isna(model_name):
        return "Supervised"
    for tech, label in TECHNIQUE_MAP.items():
        if model_name.startswith(f"{tech}_"):
            return label
    return "Supervised"


def extract_refinement(model_override_id: str) -> str:
    """`freeze` / `full_finetune` -> refinement label. If the override carries a
    learning-rate suffix (`full_finetune_lr3`, `freeze_lr4`, ...), it is kept on
    the label as `<base>_lr<N>` so each LR aggregates separately."""
    if pd.isna(model_override_id):
        return "full_finetuning"
    s = str(model_override_id)
    base = "freeze" if "freeze" in s else "full_finetuning"
    m = re.search(r"(lr\d+)", s)
    return f"{base}_{m.group(1)}" if m else base


def split_lr(refinement) -> tuple[str, str]:
    """`full_finetuning_lr3` -> (`full_finetuning`, `lr3`); plain values
    return ('', '') for the LR part. Used when merging with the benchmark,
    which has no per-LR rows."""
    if pd.isna(refinement):
        return "", ""
    m = re.match(r"(full_finetuning|freeze)_(lr\d+)$", str(refinement))
    if m:
        return m.group(1), m.group(2)
    return str(refinement), ""


def extract_spc_from_finetune(df: pd.DataFrame) -> pd.Series:
    """For each row, walk `backbone/load_from_uid` to the parent finetune row
    and read its `data/override_id`. Map that to a samples-per-class integer."""
    by_uid = df.set_index("execution/uid")

    def lookup(row):
        load_from = row.get("backbone/load_from_uid")
        if pd.isna(load_from) or load_from not in by_uid.index:
            return pd.NA
        parent = by_uid.loc[load_from]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        spc = _lookup_spc(parent["data/override_id"])
        return pd.NA if spc is None else spc

    return df.apply(lookup, axis=1)


def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (eval execution with a metric) annotated with the
    benchmark_tool columns plus the raw accuracy."""
    df = df.copy()
    df["kind"] = df["execution/id"].str.split("_").str[0]

    eval_rows = df[
        (df["kind"] == "evaluate")
        & df["metric/classification/accuracy"].notna()
        & (df["execution/status"] == "completed")
    ].copy()

    eval_rows["technique"] = eval_rows["model/name"].map(extract_technique)
    eval_rows["backbone"] = eval_rows["model/name"].map(extract_backbone)
    eval_rows["dataset"] = eval_rows["data/dataset"].map(DATASET_MAP)
    eval_rows["refinement"] = eval_rows["model/override_id"].map(extract_refinement)
    eval_rows["spc"] = extract_spc_from_finetune(df).loc[eval_rows.index]
    eval_rows["accuracy_pct"] = eval_rows["metric/classification/accuracy"] * 100.0

    return eval_rows[
        [
            "execution/id",
            "execution/uid",
            "backbone/load_from_uid",
            "technique",
            "dataset",
            "backbone",
            "refinement",
            "spc",
            "accuracy_pct",
            "model/name",
            "data/dataset",
        ]
    ]


def aggregate_for_benchmark_tool(long_df: pd.DataFrame) -> pd.DataFrame:
    """Group by (technique, dataset, backbone, refinement, spc) and compute
    mean/std of accuracy in %. Std is 0 for single-run groups."""
    grouped = (
        long_df.dropna(subset=["spc"])
        .groupby(
            ["technique", "dataset", "backbone", "refinement", "spc"], dropna=False
        )["accuracy_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["spc"] = grouped["spc"].astype(int)
    grouped["std"] = grouped["std"].fillna(0.0).round(2)
    grouped["mean"] = grouped["mean"].round(2)
    grouped = grouped.sort_values(
        ["technique", "dataset", "backbone", "refinement", "spc"]
    ).reset_index(drop=True)
    return grouped


def print_summary(long_df: pd.DataFrame, agg_df: pd.DataFrame) -> None:
    print("=" * 72)
    print(f"Evaluate rows with metrics : {len(long_df)}")
    print(f"Rows missing SPC link      : {long_df['spc'].isna().sum()}")
    print(f"Unique (tech,ds,bb,ref,spc): {len(agg_df)}")
    print()
    print("Overall accuracy (raw runs, %):")
    print(
        f"  mean = {long_df['accuracy_pct'].mean():.2f}   "
        f"std  = {long_df['accuracy_pct'].std():.2f}   "
        f"min  = {long_df['accuracy_pct'].min():.2f}   "
        f"max  = {long_df['accuracy_pct'].max():.2f}"
    )
    print()
    print("Mean accuracy per backbone:")
    print(
        long_df.groupby("backbone")["accuracy_pct"]
        .agg(["mean", "std", "count"])
        .round(2)
        .to_string()
    )
    print()
    print("Mean accuracy per dataset:")
    print(
        long_df.groupby("dataset")["accuracy_pct"]
        .agg(["mean", "std", "count"])
        .round(2)
        .to_string()
    )
    print()
    print("Mean accuracy per SPC:")
    print(
        long_df.dropna(subset=["spc"])
        .assign(spc=lambda d: d["spc"].astype(int))
        .groupby("spc")["accuracy_pct"]
        .agg(["mean", "std", "count"])
        .round(2)
        .to_string()
    )
    print("=" * 72)


def _parse_mean_std(s):
    """`"41.9 ± 2.8"` -> (41.9, 2.8); `"N/A"` or NaN -> (NaN, NaN)."""
    if pd.isna(s) or str(s).strip().upper() in {"N/A", "NA", ""}:
        return float("nan"), float("nan")
    parts = str(s).replace("±", "+/-").split("+/-")
    mean = float(parts[0].strip())
    std = float(parts[1].strip()) if len(parts) > 1 else 0.0
    return mean, std


def load_performance_data(json_path: Path) -> pd.DataFrame:
    """Read `performance_data.json` and return a long DataFrame with one row
    per (technique, dataset, backbone, refinement, spc) including mean & std."""
    records = json.loads(Path(json_path).read_text())
    out = []
    for r in records:
        for spc in SPC_VALUES:
            mean, std = _parse_mean_std(r.get(str(spc)))
            out.append(
                {
                    "technique": r["technique"],
                    "dataset": r["dataset"],
                    "backbone": r["backbone"],
                    "refinement": r["method"],
                    "spc": spc,
                    "mean": mean,
                    "std": std,
                }
            )
    return pd.DataFrame(out)


def compare_with_benchmark(
    csv_agg: pd.DataFrame, benchmark_long: pd.DataFrame
) -> pd.DataFrame:
    """Outer-join the CSV-derived aggregate with the reference benchmark.

    Merge keys are (technique, dataset, backbone, refinement_base, spc). The CSV
    side keeps `lr` (e.g. `lr3`) as an extra dimension, so a single benchmark
    cell can be paired with multiple CSV rows — one per LR variant. Returns
    columns: technique, dataset, backbone, refinement (with LR suffix if any),
    refinement_base, lr, spc, mean_csv, std_csv, mean_bench, std_bench, delta.
    """
    a = csv_agg.copy()
    a[["refinement_base", "lr"]] = a["refinement"].apply(
        lambda r: pd.Series(split_lr(r))
    )
    a = a.rename(columns={"mean": "mean_csv", "std": "std_csv"})

    b = benchmark_long.rename(
        columns={"mean": "mean_bench", "std": "std_bench", "refinement": "refinement_base"}
    )
    b["lr"] = ""  # benchmark has no LR variants

    join_keys = ["technique", "dataset", "backbone", "refinement_base", "spc"]
    merged = pd.merge(
        a[join_keys + ["refinement", "lr", "mean_csv", "std_csv"]],
        b[join_keys + ["mean_bench", "std_bench"]],
        on=join_keys,
        how="outer",
    )
    # Rows present only in the benchmark have no `refinement`/`lr` from the CSV;
    # fill them from the base so the comparison table still renders a row.
    merged["refinement"] = merged["refinement"].fillna(merged["refinement_base"])
    merged["lr"] = merged["lr"].fillna("")
    merged["delta"] = merged["mean_csv"] - merged["mean_bench"]
    merged["spc"] = merged["spc"].astype(int)
    return merged.sort_values(
        ["technique", "dataset", "backbone", "refinement_base", "lr", "spc"]
    ).reset_index(drop=True)


def format_comparison_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Wide table with one `benchmark` row plus one `csv` row per LR variant
    per (technique, dataset, encoder, refinement_base) combination.

    Columns: Technique, Dataset, Encoder, Refinement, LR, Source, Max, Min,
             Avg, 1, 5, 10, 25, 50, 100, 200, 1000
    """
    rows = []
    group_cols = ["technique", "dataset", "backbone", "refinement_base"]
    has_lr = (
        "lr" in merged.columns
        and merged["lr"].astype(str).str.len().gt(0).any()
    )

    def make_row(label_keys, source, lr_label, mean_col, std_col, sub):
        values = sub[mean_col]
        if values.dropna().empty:
            return None
        row = {
            "Technique": label_keys[0],
            "Dataset": label_keys[1],
            "Encoder": label_keys[2],
            "Refinement": label_keys[3],
            "LR": lr_label,
            "Source": source,
            "Max": round(values.max(skipna=True), 2),
            "Min": round(values.min(skipna=True), 2),
            "Avg": round(values.mean(skipna=True), 2),
        }
        for spc in SPC_VALUES:
            m = sub.loc[spc, mean_col] if spc in sub.index else float("nan")
            s = sub.loc[spc, std_col] if spc in sub.index else float("nan")
            row[str(spc)] = "N/A" if pd.isna(m) else f"{m:.1f} ± {s:.1f}"
        return row

    for keys, sub in merged.groupby(group_cols, dropna=False):
        bench_sub = sub.drop_duplicates(subset=["spc"]).set_index("spc")
        row = make_row(keys, "benchmark", "", "mean_bench", "std_bench", bench_sub)
        if row is not None:
            rows.append(row)

        for lr_val, lr_sub in sub.groupby("lr", dropna=False):
            lr_sub = lr_sub.set_index("spc")
            row = make_row(keys, "csv", str(lr_val), "mean_csv", "std_csv", lr_sub)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    if not has_lr and "LR" in df.columns:
        df = df.drop(columns=["LR"])
    return df


def plot_comparison(
    merged: pd.DataFrame, output: Path | None = None, title: str | None = None
):
    """Grid of small subplots — one per (dataset, backbone). Each subplot
    shows benchmark vs csv mean accuracy across SPC, with ±std bands."""
    import matplotlib.pyplot as plt
    import numpy as np

    have = merged.dropna(subset=["mean_csv", "mean_bench"], how="all")
    datasets = sorted(have["dataset"].dropna().unique())
    backbones = sorted(have["backbone"].dropna().unique())
    n_rows, n_cols = len(datasets), len(backbones)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("No (dataset, backbone) cells overlap to plot.")

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), sharex=True, sharey=True,
        squeeze=False,
    )
    lr_present = (
        "lr" in have.columns and have["lr"].astype(str).str.len().gt(0).any()
    )
    lr_colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    for i, ds in enumerate(datasets):
        for j, bb in enumerate(backbones):
            ax = axes[i][j]
            cell = have[(have["dataset"] == ds) & (have["backbone"] == bb)].sort_values("spc")
            if cell.empty:
                ax.set_visible(False)
                continue

            bench_cell = cell.drop_duplicates(subset=["spc"]).sort_values("spc")
            x_b = bench_cell["spc"].astype(int).to_numpy()
            m_b = bench_cell["mean_bench"].to_numpy(dtype=float)
            s_b = bench_cell["std_bench"].to_numpy(dtype=float)
            mask = ~np.isnan(m_b)
            if mask.any():
                ax.plot(x_b[mask], m_b[mask], "o-", label="benchmark",
                        color="#1f77b4", lw=1.4, ms=4)
                s_safe = np.nan_to_num(s_b[mask], nan=0.0)
                ax.fill_between(x_b[mask], m_b[mask] - s_safe, m_b[mask] + s_safe,
                                color="#1f77b4", alpha=0.15)

            lr_groups = (
                list(cell.groupby("lr")) if lr_present else [("", cell)]
            )
            for k, (lr_val, lr_cell) in enumerate(lr_groups):
                lr_cell = lr_cell.sort_values("spc")
                x = lr_cell["spc"].astype(int).to_numpy()
                m = lr_cell["mean_csv"].to_numpy(dtype=float)
                s = lr_cell["std_csv"].to_numpy(dtype=float)
                mask = ~np.isnan(m)
                if not mask.any():
                    continue
                label = f"csv ({lr_val})" if lr_val else "csv"
                color = lr_colors[k % len(lr_colors)]
                ax.plot(x[mask], m[mask], "o-", label=label,
                        color=color, lw=1.4, ms=4)
                s_safe = np.nan_to_num(s[mask], nan=0.0)
                ax.fill_between(x[mask], m[mask] - s_safe, m[mask] + s_safe,
                                color=color, alpha=0.15)
            ax.set_xscale("log")
            ax.set_xticks(SPC_VALUES)
            ax.set_xticklabels(SPC_VALUES, rotation=45, fontsize=7)
            ax.grid(True, ls=":", alpha=0.4)
            if i == 0:
                ax.set_title(bb, fontsize=9)
            if j == 0:
                ax.set_ylabel(ds, fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("SPC", fontsize=8)

    handles, labels = [], []
    seen = set()
    for ax_row in axes:
        for ax in ax_row:
            if not ax.get_visible():
                continue
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 4),
                   frameon=False, bbox_to_anchor=(0.5, 0.97))
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Wrote comparison figure  -> {output}")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_csv", type=Path, help="saved_metrics_*.csv from summarizer")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: benchmark_tool_<input_stem>.csv next to input)",
    )
    p.add_argument(
        "--long",
        type=Path,
        default=None,
        help="Optional: also write the un-aggregated long table to this path",
    )
    p.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Reference performance_data.json to compare against",
    )
    p.add_argument(
        "--compare-csv",
        type=Path,
        default=None,
        help="Write the side-by-side comparison table to this CSV (requires --benchmark)",
    )
    p.add_argument(
        "--compare-fig",
        type=Path,
        default=None,
        help="Write the comparison figure to this path, e.g. compare.png (requires --benchmark)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    long_df = build_long_table(df)
    agg_df = aggregate_for_benchmark_tool(long_df)

    print_summary(long_df, agg_df)

    output = args.output or args.input_csv.with_name(
        f"benchmark_tool_{args.input_csv.stem}.csv"
    )
    agg_df[["technique", "dataset", "backbone", "refinement", "spc", "mean", "std"]].to_csv(
        output, index=False
    )
    print(f"\nWrote benchmark_tool CSV -> {output}")

    if args.long:
        long_df.to_csv(args.long, index=False)
        print(f"Wrote long table         -> {args.long}")

    if args.benchmark:
        bench = load_performance_data(args.benchmark)
        merged = compare_with_benchmark(agg_df, bench)
        # Restrict comparison output to cells where the CSV has data.
        # Use refinement_base (LR-stripped) so benchmark-only rows still match.
        csv_keys = agg_df.copy()
        csv_keys["refinement_base"] = csv_keys["refinement"].apply(
            lambda r: split_lr(r)[0]
        )
        csv_keys = csv_keys[
            ["technique", "dataset", "backbone", "refinement_base"]
        ].drop_duplicates()

        merged_for_view = merged.merge(
            csv_keys,
            on=["technique", "dataset", "backbone", "refinement_base"],
            how="inner",
        )

        table = format_comparison_table(merged_for_view)
        print("\nSide-by-side comparison (benchmark vs csv):")
        print(table.to_string(index=False))

        if args.compare_csv:
            table.to_csv(args.compare_csv, index=False)
            print(f"\nWrote comparison table   -> {args.compare_csv}")

        if args.compare_fig:
            # Split by refinement_base so freeze and full_finetuning go to
            # separate figures. When only one refinement is present, behaves
            # like before (single output at the given path).
            refs = sorted(
                r for r in merged_for_view["refinement_base"].dropna().unique() if r
            )
            base = Path(args.compare_fig)
            if len(refs) <= 1:
                plot_comparison(
                    merged_for_view,
                    output=base,
                    title="CSV vs benchmark — mean ± std accuracy",
                )
            else:
                for r in refs:
                    sub = merged_for_view[merged_for_view["refinement_base"] == r]
                    out = base.with_name(f"{base.stem}_{r}{base.suffix}")
                    plot_comparison(
                        sub,
                        output=out,
                        title=f"CSV vs benchmark — {r} (mean ± std accuracy)",
                    )


if __name__ == "__main__":
    main()
