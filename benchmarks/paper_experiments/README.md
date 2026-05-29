# Replication Guide — Paper Experiments

This directory contains all experiment configurations used to produce the results in:

> **Benchmarking Encoders and Self-Supervised Learning for Smartphone-Based Human Activity Recognition**
> da Luz et al., IEEE Access 2026

Each subdirectory under `to_be_validated/` is a self-contained experiment with its `experiments.csv` and overrides. The lifecycle (`to_be_validated/` → `validation_run/` → `validated/`) is documented here.

---

## Prerequisites

**1. Setup from the root directory**: follow the install instructions in the **[root README](../../README.md)** (run `./install.sh` to install Minerva and the orchestration dependencies). If you prefer a Docker dev container, follow **[../../set_docker.md](../../set_docker.md)** instead.

**2. Update config paths**

The YAML files under `base_configs/` contain absolute `data_path` values. After cloning (or moving the repo), run once to rewrite them to your local path:

```bash
cd benchmarks
python scripts/update_config_paths.py
```

This only rewrites the host-specific prefix before `shared_data/` in the affected YAML files. Dataset suffixes are never touched.

**3. Working directory**

All commands below assume:

```bash
cd benchmarks
export BASE_CONFIGS_PATH="$(pwd)/base_configs"
```

---

## Directory Structure

```
benchmarking-encoders-ssl-har/
└── benchmarks/paper_experiments/
    ├── to_be_validated/                              # static catalog 
    │   ├── tfc/                                      # TF-C (12 experiment dirs)
    │   │   ├── tfc_ts2vec_run{1,2,3}/               # TS2Vec backbone × 3 seeds
    │   │   ├── tfc_rnn_run{1,2,3}/                  # RNN backbone × 3 seeds
    │   │   ├── tfc_resnet_cnn_transformers_run{1,2,3}/  # CNN-PFF + ResNetSE-5 + Transformer × 3 seeds
    │   │   └── tfc_harcnn_run{1,2,3}/               # HARSCnnEncoder (TS-TCC) × 3 seeds
    │   ├── lfr/                                      # LFR (12 experiment dirs)
    │   ├── tnc/                                      # TNC (12 experiment dirs)
    │   ├── diet/                                     # DIET (12 experiment dirs)
    │   └── ablations/
    │       └── learning_rate/                        # LR ablation (all encoders)
    ├── validation_run/                               # currently running (work area)
    └── validated/                                    # finished + saved_metrics_*.csv
```

> **Note on `harcnn` model yaml names**: LFR and DIET experiments for `harcnn` use the generic
> `lfr_default` / `diet_default` yaml (which already targets `HARSCnnEncoder`). TFC and TNC use
> explicit `tfc_harcnn` / `tnc_harcnn` yamls. The backbone is identical in all four cases.

Each experiment directory has the same internal layout:

```
<technique>/<name>_run{N}/
├── configs/
│   ├── experiments.csv       # Defines pretrain → finetune → evaluate pipeline
│   ├── generated_executions.csv  # DAG produced by execution_planner.py (commit when stable)
│   └── overrides/
│       ├── data_modules.csv  # Data percentage and sample-count overrides
│       ├── models.csv        # Freeze vs. full-finetune strategy
│       └── pipelines.csv     # Epoch count overrides
└── logs/                     # Training logs (generated at runtime)
```

### Pipeline defined in `experiments.csv`

Each experiment runs three phases in order:

| Phase | `execution/id` | Description |
|-------|---------------|-------------|
| 1 | `pretrain_<backbone>` | SSL pretraining without overlap |
| 2 | `finetune_<backbone>` | Refinement on DAGHAR `standardized_view` (freeze or full finetune) |
| 3 | `evaluate_<backbone>` | Evaluation on DAGHAR test split |

### Model overrides (from `overrides/models.csv`)

| `override_id` | `freeze_backbone` | `learning_rate` |
|--------------|-------------------|-----------------|
| `freeze` | `True` | `1e-4` |
| `full_finetune` | `False` | `1e-4` |

### Data overrides (from `overrides/data_modules.csv`)

| `override_id` | Samples per class |
|--------------|------------------|
| `multimodal_samples_001` | 1 |
| `multimodal_samples_005` | 5 |
| `multimodal_samples_010` | 10 |
| `multimodal_samples_025` | 25 |
| `multimodal_samples_050` | 50 |
| `multimodal_samples_100` | 100 |
| `multimodal_samples_200` | 200 |
| `multimodal_perc_100` | 100% of data |

---

## Practical commands

Copy and run (from the `benchmarks/` directory):

```bash
# pick one experiment from the catalog
cp -r paper_experiments/to_be_validated/tnc/tnc_rnn_run1 \
      paper_experiments/validation_run/tnc/tnc_rnn_run1

# plan + run + summarize (loop over validation_run)
BASE_CONFIGS_PATH="$(pwd)/base_configs"

for d in paper_experiments/validation_run/*/*/; do
    python scripts/execution_planner.py \
        --executions_path   "$d/configs/experiments.csv" \
        --base_configs_path "$BASE_CONFIGS_PATH" \
        --overrides_path    "$d/configs/overrides" \
        --db_file           "$d/configs/generated_executions.csv" \
        --log_dir           "$d/logs" \
        --seed              42
done

for d in paper_experiments/validation_run/*/*/; do
    python scripts/submit_it.py "$d/configs/generated_executions.csv" \
        -w 4 --use-ray --ray-address 127.0.0.1:6379
done

for run_dir in paper_experiments/validation_run/*/*; do
    name=$(basename "$run_dir")
    python scripts/summarizer.py \
        "${run_dir}/configs/generated_executions.csv" \
        --output_csv "saved_metrics_${name}.csv"
done

# promote what has been validated
mv paper_experiments/validation_run/tnc/tnc_rnn_run1 \
   paper_experiments/validated/tnc/tnc_rnn_run1
mv saved_metrics_tnc_rnn_run1.csv paper_experiments/validated/tnc/tnc_rnn_run1/
```

---

## Comparing against the paper baseline

Once you have one or more `saved_metrics_*.csv` files produced by `summarizer.py`, you can compare them against the reference results shipped with the repo (`scripts/performance_data.json`) using `scripts/run_benchmark_compare.sh`.

### What the script does

For each input `saved_metrics_<tag>.csv`, the script calls `saved_to_benchmark_tool.py` to:

1. **Reshape the metrics** into the schema expected by [`benchmark_tool.html`](https://www.students.ic.unicamp.br/~ra271582/paper_encoders/benchmark_tool.html):
   `technique, dataset, backbone, refinement, spc, mean, std`.
2. **Aggregate** across seeds for each `(technique, dataset, backbone, refinement, spc)` group (mean ± std).
3. **Compare** against the paper baseline stored in `scripts/performance_data.json`.
4. **Plot** side-by-side curves (mean accuracy vs. samples-per-class), one PNG per `refinement` (`freeze` and `full_finetuning` go to separate figures).

Each input produces three outputs in the current directory:

| Output | Description |
|---|---|
| `benchmark_tool_<tag>.csv` | Drop-in CSV for `benchmark_tool.html` |
| `comparison_<tag>.csv` | Side-by-side table: your results vs. paper baseline (per dataset / backbone / refinement / spc) |
| `comparison_<tag>_<refinement>.png` | Grid plot per refinement (e.g. `comparison_diet_rnn_run1_freeze.png`, `comparison_diet_rnn_run1_full_finetuning.png`) |

### Usage

Run from inside `benchmarks/` (the same directory where the `saved_metrics_*.csv` files live):

```bash
# single file
./scripts/run_benchmark_compare.sh saved_metrics_diet_rnn_run1.csv

# all files at once
./scripts/run_benchmark_compare.sh

# explicit glob
./scripts/run_benchmark_compare.sh saved_metrics_diet_*.csv
```

If you only have one seed (`_run1`), `std` will be 0 across the board. To get the paper-style mean ± std over 3 seeds, concatenate the per-run metrics first:

```bash
# combine run1, run2, run3 into a single file before comparing
python -c "
import pandas as pd
dfs = [pd.read_csv(f'saved_metrics_diet_rnn_run{i}.csv') for i in (1, 2, 3)]
pd.concat(dfs).to_csv('saved_metrics_diet_rnn.csv', index=False)
"

./scripts/run_benchmark_compare.sh saved_metrics_diet_rnn.csv
```

### Visualizing in `benchmark_tool`

Open [the hosted benchmark tool](https://www.students.ic.unicamp.br/~ra271582/paper_encoders/benchmark_tool.html) (or a local copy) and load the generated `benchmark_tool_<tag>.csv`. The tool plots your results alongside the paper's baseline curves and lets you filter by technique, refinement, dataset and backbone.

---

## Conventions

- **Never edit `base_configs/` directly.** Override fields per experiment via `overrides/<type>.csv`.
- **`logs/` is disposable** — it's in `.gitignore`. Heavy logs stay only on the machine that ran the job.
- **`saved_metrics_*.csv` is the canonical output** that feeds the paper results.
- **Seeds**: the paper uses 3 seeds per experiment (`_run1`, `_run2`, `_run3`). A quick sanity check can be run with a single seed.
