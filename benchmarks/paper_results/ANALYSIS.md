# Analysis Guide — Paper Results

This directory contains all notebooks, scripts, CSVs, and figures used to generate the results in:

> **Benchmarking Encoders and Self-Supervised Learning for Smartphone-Based Human Activity Recognition**
> da Luz et al., IEEE Access 2026

---

## Prerequisites

Python 3.11.7 was used for all analysis. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Input Data

All analysis notebooks read from pre-aggregated CSV files produced by `summarizer.py` (see `paper_experiments/REPLICATION.md`).

| File | Description | Rows |
|------|-------------|------|
| `saved_metrics_paper_final.csv` | Raw aggregated metrics from all paper runs | ~11,232 |
| `clean_saved_metrics_paper_final.csv` | After preprocessing (notebook 1) — primary input for RQ notebooks | 7,776 |
| `combined_df_wilcoxon_ft_saved_metrics_paper_final.csv` | Wilcoxon test results, full-finetune strategy | — |
| `combined_df_wilcoxon_freeze_saved_metrics_paper_final.csv` | Wilcoxon test results, freeze strategy | — |
| `best_per_dataset_technique_ft_saved_metrics_paper_final.csv` | Best model per dataset × technique (full finetune) | — |
| `best_per_dataset_technique_freeze_saved_metrics_paper_final.csv` | Best model per dataset × technique (freeze) | — |
| `ablation/clean_saved_metrics_paper_daghar_supervised_*.csv` | Ablation study inputs | — |

---

## Workflow

Run notebooks in order. Notebook 1 is mandatory — it produces `clean_saved_metrics_paper_final.csv` used by all subsequent notebooks.

```
1-preproc_wilcoxontests.ipynb          ← MUST RUN FIRST
        │
        └──► clean_saved_metrics_paper_final.csv
                    │
          ┌─────────┴──────────────────────────────────────┐
          │         │         │         │         │         │
       2-rq1    3-rq2      4-rq3    5-rq4      6-rq5    7-rq6
```

For ablation analysis:
```
ablation/extract_ablations.ipynb       ← preprocesses ablation CSVs
ablation/0-ablationlr_*.ipynb          ← LR  analysis
```

---

## Notebook Descriptions

### `1-preproc_wilcoxontests.ipynb` — Preprocessing & Statistical Tests

**Input:** `saved_metrics_paper_final.csv`
**Outputs:** `clean_saved_metrics_paper_final.csv`, `combined_df_wilcoxon_ft_*.csv`, `combined_df_wilcoxon_freeze_*.csv`

- Filters and cleans the raw metrics CSV
- Separates results by refinement strategy (freeze vs. full finetune)
- Runs pairwise Wilcoxon signed-rank tests with Bonferroni correction across backbone × dataset combinations
- Result: statistically validated comparison tables used in all RQ plots

---

### `2-rq1.ipynb` — RQ1: Which encoder performs best overall?

**Input:** `clean_saved_metrics_paper_final.csv`
**Output:** `ssl_rq1.png`, `supervised_rq1.png`

Compares balanced accuracy across all SSL techniques and supervised baselines, aggregated over datasets and runs.

---

### `3-rq2.ipynb` — RQ2: Does the refinement strategy matter?

**Input:** `clean_saved_metrics_paper_final.csv`
**Output:** `ssl_sl_rq2.png`, `backbone_comparisons_boxplot.png/.pdf`

Compares freeze vs. full-finetune strategies across all encoder × technique combinations.

---

### `4-rq3.ipynb` — RQ3: Is the SSL technique or the backbone architecture more important?

**Input:** `clean_saved_metrics_paper_final.csv`
**Output:** `ssl_sl_rq3.png`, `technique_comparisons_boxplot.png/.pdf`, `technique_backbone_comparison.png`

Decomposes performance variance into technique contribution vs. architecture contribution.

---

### `5-rq4.ipynb` — RQ4: How does encoder performance vary across different target datasets?

**Input:** `clean_saved_metrics_paper_final.csv`
**Output:** `finetune_rq4.png`, `finetune_rq42.png`

Evaluates performance per dataset.

---

### `6-rq5.ipynb` — RQ5: How much labeled data is needed?

**Input:** `clean_saved_metrics_paper_final.csv` (sample-count override results)
**Output:** `backbone_dataset_performance.png`

Plots accuracy vs. labeled samples per class (1 → 200) for all encoder × technique combinations.

---

### `7-rq6_modified.ipynb` — RQ6: When does SSL outperform supervised learning?

**Input:** `clean_saved_metrics_paper_final.csv`
**Output:** `backbones_freeze_lr4_lr3_seed.png`, `backbones_vs_supervised_ft_lr4_lr3_seed.png`

Identifies the labeled-data regime where SSL models match or exceed supervised baselines.

---

### `ablation/extract_ablations.ipynb`

Preprocesses raw ablation CSVs into the cleaned format required by the ablation analysis notebooks.

---

### `ablation/0-ablationlr_*.ipynb` — Learning Rate Ablation

Three variants covering different encoder subsets:
- `0-ablationlr_remainingencoders.ipynb` — CNN-PFF, IMU Transformer, RNN, TS2Vec
- `0-ablationlr_resnetse5.ipynb` — ResNetSE-5
- `0-ablationlr_tstcc.ipynb` — TS-TCC encoder

---

## Utility Functions (`utils.py`)

Shared analysis helpers used across notebooks:

| Function | Description |
|----------|-------------|
| `prepare_experiment_df()` | Filter results by technique, backbone, dataset, strategy |
| `calculate_variant_wilcoxon_pandas()` | Pairwise Wilcoxon tests with Bonferroni correction |
| `create_precedence_graph()` | Build ranking graphs from pairwise comparisons |
| `aggregate_backbone_performance()` | Summary statistics per backbone |
