#!/usr/bin/env bash
# Convert summarizer `saved_metrics_*.csv` files into the format expected by
# `benchmark_tool.html` and compare them against `performance_data.json`.
#
# Usage:
#     ./run_benchmark_compare.sh                              # all saved_metrics_*.csv
#     ./run_benchmark_compare.sh saved_metrics_tnc_harcnn.csv # one file
#     ./run_benchmark_compare.sh saved_metrics_*.csv          # explicit glob
#
# Each input produces:
#     benchmark_tool_<stem>.csv  — drop into benchmark_tool.html
#     comparison_<tag>.csv       — side-by-side benchmark vs csv table
#     comparison_<tag>.png       — 6×N panel figure (dataset × backbone)
# where <stem> is the input filename without extension and <tag> is <stem> with
# the `saved_metrics_` prefix stripped.

set -euo pipefail

cd "$(dirname "$0")"

BENCHMARK_JSON="performance_data.json"
SCRIPT="saved_to_benchmark_tool.py"

if [[ ! -f "$SCRIPT" ]]; then
    echo "error: $SCRIPT not found in $(pwd)" >&2
    exit 1
fi
if [[ ! -f "$BENCHMARK_JSON" ]]; then
    echo "error: $BENCHMARK_JSON not found in $(pwd)" >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    inputs=("$@")
else
    shopt -s nullglob
    inputs=(saved_metrics_*.csv)
    shopt -u nullglob
fi

if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "error: no saved_metrics_*.csv files found" >&2
    exit 1
fi

for csv in "${inputs[@]}"; do
    if [[ ! -f "$csv" ]]; then
        echo "warning: skipping missing file $csv" >&2
        continue
    fi
    stem="$(basename "$csv" .csv)"
    tag="${stem#saved_metrics_}"

    echo "================================================================"
    echo "Processing $csv  (tag=$tag)"
    echo "================================================================"

    python3 "$SCRIPT" "$csv" \
        --benchmark "$BENCHMARK_JSON" \
        --compare-csv "comparison_${tag}.csv" \
        --compare-fig "comparison_${tag}.png"

    echo
done

echo "Done."
