# Iterative Partial Refinement (IPR)

Official code for **"Inference-Time Scaling in Diffusion Models through Iterative Partial Refinement"** (ICLR 2026 Workshop on AI with Recursive Self-Improvement). [[Paper]](https://openreview.net/forum?id=QopjICzGwr)

IPR is a simple inference-time scaling method for sequential diffusion models. Starting from an initial sample, IPR repeatedly selects a random subset of regions, re-noises them, and regenerates them conditioned on the remaining regions. This enables the model to revise earlier decisions and correct global inconsistencies — without external verifiers, reward models, or additional training.

## Project Structure

```
IPR/
├── config/              # Hydra base configs (model, dataset, sampler, etc.)
├── experiments/         # Experiment configs (ep_test, cp_test, ms_hard_test, ...)
├── src/
│   ├── main.py          # Entry point
│   ├── _main.py         # Hydra main function
│   ├── config.py        # Config dataclasses
│   ├── model/           # Sequential diffusion model (flow matching)
│   ├── dataset/         # Dataset loaders
│   ├── sampler/         # IPR sampler implementation
│   ├── evaluation/      # Task-specific evaluation metrics
│   └── visualization/   # Sampling visualization utilities
├── srmbench/            # SRM Benchmarks library (installed as editable package)
├── outputs/             # Checkpoints (not tracked by git)
├── run_exp.sh           # Experiment runner script
└── requirements.txt
```

## Setup

### 1. Environment

```bash
conda create -n ipr python=3.11 -y
conda activate ipr
pip install -r requirements.txt
```

### 2. Checkpoints & Datasets

Download from [SRM Releases (v1.0.0)](https://github.com/Chrixtar/SRM/releases/tag/v1.0.0) and extract each zip at the project root:

```bash
# Checkpoints
wget https://github.com/Chrixtar/SRM/releases/download/v1.0.0/even_pixels.zip
wget https://github.com/Chrixtar/SRM/releases/download/v1.0.0/cp_ffhq.zip
wget https://github.com/Chrixtar/SRM/releases/download/v1.0.0/mnist_sudoku.zip

# Datasets (MNIST Sudoku + Counting Polygons)
wget https://github.com/Chrixtar/SRM/releases/download/v1.0.0/datasets.zip

# Extract all at project root
unzip even_pixels.zip
unzip cp_ffhq.zip
unzip mnist_sudoku.zip
unzip datasets.zip
```

This creates the required directory structure:

```
outputs/
├── ep_4/paper/checkpoints/last.ckpt          # Even Pixels
├── cp_ffhq_8/paper/checkpoints/last.ckpt     # Counting Polygons
└── ms1000_28/paper/checkpoints/last.ckpt     # MNIST Sudoku

datasets/
├── counting_polygons/
└── mnist_sudoku/
```

Counting Polygons additionally requires [FFHQ](https://github.com/NVlabs/ffhq-dataset) — download and place under `datasets/counting_polygons/ffhq/`.

## Running Experiments

```bash
# Usage: bash run_exp.sh <config_name> <gpu_id>

# Even Pixels
bash run_exp.sh ep_test 0

# Counting Polygons
bash run_exp.sh cp_test 0

# MNIST Sudoku (hard)
bash run_exp.sh ms_hard_test 0

# MNIST Sudoku (K-corrupted)
bash run_exp.sh ms_corrupted_test 0
```

## Hyperparameters

| Hyperparameter | Description | EP | CP | MS |
|---|---|---|---|---|
| `overlap` | Init scheduling overlap ratio during SRM generation | 0.9 | 0.9 | 0.0 |
| `steps_per_patch` | Init denoising steps per region | 30 | 10 | 3 |
| `ipr_overlap` | Scheduling overlap ratio during IPR refinement | 0.9 | 0.9 | 0.8 |
| `ipr_steps_per_patch` | Denoising steps per re-sampled region during IPR | 30 | 10 | 10 |
| `stochasticity` | Randomness injected during diffusion sampling | 0.5 | 0.5 | 0.5 |
| `resampling_ratio` | Fraction of regions re-noised per IPR iteration | 0.25 | 0.25 | 0.25 |
| `max_ipr_budget` | Total number of IPR iterations | 50 | 50 | 50 |

## Citation

```bibtex
@inproceedings{kang2026ipr,
  title={Inference-Time Scaling in Diffusion Models through Iterative Partial Refinement},
  author={Kang, Taegu and Yoon, Jaesik and Ahn, Sungjin},
  booktitle={ICLR 2026 Workshop on AI with Recursive Self-Improvement},
  year={2026},
  url={https://openreview.net/forum?id=QopjICzGwr}
}
```

## Acknowledgements

This codebase builds upon [Spatial Reasoning Models (SRMs)](https://github.com/Chrixtar/SRM) by Wewer et al. (2025).
