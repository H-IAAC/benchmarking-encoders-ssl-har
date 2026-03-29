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



---

## Built with Minerva

All experiments in this paper were conducted using [**Minerva**](https://github.com/discovery-unicamp/Minerva), a PyTorch Lightning-based framework for training machine learning models. Minerva provides the model definitions, SSL pipelines, data modules, and evaluation tools used throughout this benchmark.

- GitHub: [https://github.com/discovery-unicamp/Minerva](https://github.com/discovery-unicamp/Minerva)
- PyPI: [https://pypi.org/project/minerva/](https://pypi.org/project/minerva/) (`pip install minerva`)

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

> **📢 Coming soon (next weeks):** Full step-by-step instructions to also replicate the entire pre-training pipeline with Minerva, with configs, experiment execution, and result analysis.