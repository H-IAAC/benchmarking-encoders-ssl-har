<h1 align="center">
Benchmarking Encoders and Self-Supervised Learning <br>
for Smartphone-Based Human Activity Recognition
</h1>

<p align="center">
    <a href="https://ieeexplore.ieee.org/document/11417778">
        <img alt="IEEE Access" src="https://img.shields.io/badge/IEEE_Access-2026-red">
    </a>
    <a href="https://zenodo.org/records/19301058">
        <img alt="Zenodo" src="https://img.shields.io/badge/Zenodo-Models-blue">
    </a>
    <a href="https://www.students.ic.unicamp.br/~ra271582/paper_encoders/benchmark_tool.html">
        <img alt="Benchmark Tool" src="https://img.shields.io/badge/Benchmark-Interactive_Tool-purple">
    </a>
</p>

<div align="center">

**Gustavo P. C. P. da Luz**<sup>1</sup> &nbsp;,&nbsp;
**Darlinne H. P. Soto**<sup>1</sup> &nbsp;,&nbsp;
**Otávio O. Napoli**<sup>1</sup> &nbsp;,&nbsp;
**Anderson Rocha**<sup>1</sup> (Fellow, IEEE) &nbsp;,&nbsp;
**Levy Boccato**<sup>1</sup> &nbsp;,&nbsp;
**Edson Borin**<sup>1</sup> (IEEE)

<sup>1</sup> Hub for Artificial Intelligence and Cognitive Architectures (H.IAAC)
University of Campinas (UNICAMP), Brazil

</div>

<br>

<p align="center">
Large-scale evaluation of self-supervised learning for smartphone-based HAR using the DAGHAR benchmark. 11,232 models were evaluated across four SSL techniques, six encoders, two refinement strategies, and six datasets. We show that SSL can outperform supervised learning with fewer labeled samples.
</p>

<p align="center">
<img src="graphicalabstract.png" width="900" alt="Graphical Abstract of Benchmark">
</p>

---

## Pretrained & Finetuned Models

Best pretrained and finetuned checkpoints for **36 SSL + encoder combinations** across **6 HAR datasets** are available on Zenodo:

**[https://zenodo.org/records/19301058](https://zenodo.org/records/19301058)**

Each combination provides:
- `*_pretrained.ckpt` : SSL pretrained backbone (no labels used during pretraining)
- `*_finetuned.ckpt` : Best finetuned model (full fine-tuning, best of 3 seeds)

**Datasets:** KuHar · MotionSense · RealWorld Thigh · RealWorld Waist · UCI-HAR · WISDM
**Encoders:** TS2Vec · CNN-PFF · ResNet-SE-5 · RNN · IMU Transformer · TS-TCC
**SSL techniques:** LFR · TF-C · DIET

---

## Quick Start

The easiest way to load, evaluate, or fine-tune any of the 36 models is through the interactive notebook:

| Notebook | Description |
|----------|-------------|
| [`ssl_har_model_zoo.ipynb`](ssl_har_model_zoo.ipynb) | Select any SSL + encoder + dataset combination from the ones available, download the checkpoint from Zenodo, and evaluate or fine-tune in a few cells |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1rIyAhPUTJhZjGPqecppAbNxfl43feei6/view?usp=sharing) | Run directly in Google Colab (no local setup required) |

The notebook handles checkpoint download, model instantiation, data loading, and metric reporting automatically.

---

## Dataset: DAGHAR

All models are trained and evaluated on the [**DAGHAR**](https://github.com/H-IAAC/DAGHAR) benchmark  a curated collection of smartphone-based HAR datasets with a standardized view (6 IMU channels, window size 60) enabling fair cross-dataset comparison.

- GitHub: [https://github.com/H-IAAC/DAGHAR](https://github.com/H-IAAC/DAGHAR)
- Zenodo: [https://zenodo.org/records/13987073](https://zenodo.org/records/13987073)


### Downloading Data

Data is **not** included in the repository. Download and prepare all datasets with:

```bash
./download_data.sh
```

This will populate a new folder `shared_data/` with:

```
shared_data/
├── daghar/standardized_view/          # DAGHAR benchmark (6 datasets)
├── rodrigues_2024_datasets/1-1/       # Per-user CSVs for SSL pretraining
└── xu_2023_datasets/1-1/             # NumPy arrays for TNC pretraining
```

See [`prepare_data.py`](prepare_data.py) for advanced options (custom root, skip download, partial preparation).

---

---

## Built with Minerva

All experiments in this paper were conducted using [**Minerva**](https://github.com/discovery-unicamp/Minerva), a PyTorch Lightning-based framework for training machine learning models. Minerva provides the model definitions, SSL pipelines, data modules, and evaluation tools used throughout this benchmark.

- GitHub: [https://github.com/discovery-unicamp/Minerva](https://github.com/discovery-unicamp/Minerva)
- PyPI: [https://pypi.org/project/minerva/](https://pypi.org/project/minerva/) (`pip install minerva==0.3.10b0`)

---

## 1. Python environment

Recommended: Python ≥ 3.10 in an isolated environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
pip uninstall pandas
pip install pandas==2.3.3
```

This installs Minerva (`minerva==0.3.10b0`) and the packages used by the orchestration scripts under `benchmarks/scripts/` (`pandas`, `pandasql`, `networkx`, `ray`, etc.).

---

## 3. Verify the installation

```bash
python -c "import minerva; print('minerva:', minerva.__version__)"
python -c "import ray, pandasql, networkx; print('extras OK')"
```

---

## Notes

- **Ray**: `submit_it.py` assumes an active Ray cluster. Start a local one with:
  ```bash
  ray start --head --port=6379
  ```
- **GPU**: to run pretraining/finetuning you'll need a GPU compatible with the PyTorch version shipped with Minerva. If you have trouble with pytorch you can uninstall and install specific wheels (e.g `pip install --index-url https://download.pytorch.org/whl/cu121`       torch torchvision )
- **Docker / VSCode dev container**: a ready-to-use alternative is documented in [`set_docker.md`](set_docker.md).



---

## Citation

```bibtex
@article{daLuz2026benchmarking,
  author={Da Luz, Gustavo P. C. P. and Soto, Darlinne H. P. and Napoli, Otávio O. and Rocha, Anderson and Boccato, Levy and Borin, Edson},
  journal={IEEE Access},
  title={Benchmarking Encoders and Self-Supervised Learning for Smartphone-Based Human Activity Recognition},
  year={2026},
  volume={14},
  pages={37451-37475},
  doi={10.1109/ACCESS.2026.3669412}
}
```

---

## Pre-Training Replication

Full step-by-step instructions to also replicate the entire pre-training pipeline with Minerva, with configs, experiment execution, and result analysis.


### Repository layout

```
benchmarking-encoders-ssl-har/
├── requirements.txt
└── benchmarks/
    ├── scripts/              ← planner, submitter, summarizer (see scripts/README.md)
    ├── base_configs/         ← base YAML configs (data_modules / models / pipelines)
    └── paper_experiments/
        ├── to_be_validated/  ← ALL paper experiments (static catalog)
        ├── validation_run/   ← currently running (work area)
        └── validated/        ← finished
    ├── paper_results/         ← analysis of the results presented in the paper
├── download_data.sh
├── prepare_data.py
├── set_docker.md
├── ssl_har_model_zoo.ipynb
```

The lifecycle is: copy from `to_be_validated/` → run in `validation_run/` → move to `validated/`. See **[`benchmarks/paper_experiments/README.md`](benchmarks/paper_experiments/README.md)** for details on how to run and analyse results.

