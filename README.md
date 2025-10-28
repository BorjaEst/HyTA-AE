## Overview

This repository contains the code and paper for a study on forward-only learning in an entorhinal–hippocampal (EHC) autoencoder. We compare Backpropagation (BP), Direct Feedback Alignment (DFA), Direct Random Target Projection (DRTP), and a Hybrid Target Alignment variant (HyTA-AE) under dense spatial-map reconstruction and Full-Context Masked Training (FCMT).

The LaTeX sources for the manuscript live under `article/` and compile to a paper with external references to the supplement. Python code for training and figure generation is under `experiments/` and `figures/` respectively.

### How to cite this work
> TODO 


## Repository layout

- `experiments/` — training scripts for each algorithm:
	- `bpae_baseline.py` (BP), `dfaae_training.py` (DFA), `drtpae_training.py` (DRTP), `hytaae_training.py` (Hybrid).
- `figures/` — scripts and cached assets to generate plots used in the paper:
	- `gen_timeseries.py`, `gen_contour.py`, `gen_heatmap.py` and subfolders with generated PDFs.
- `examples/` — small, self-contained scripts to visualize datasets and masks.
- `src/` — library code (`ehc_sn`) used by experiments and examples.
- `article/` — LaTeX sources (`main.tex`, `suplemental/supplemental.tex`, `references.bib`, `public/`, `figures/`).
- `pyproject.toml` — project metadata and dependencies.
- `LICENSE` — GPLv3.

## Environment setup

The project targets Python 3.10+. Dependencies are declared in `pyproject.toml` (PyTorch, Lightning, MiniGrid, Matplotlib/Seaborn, TensorBoard, etc.). Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Notes:
- Torch/Torchvision versions are pinned in `pyproject.toml`. On systems with CUDA, you may prefer installing a CUDA-specific PyTorch wheel first, then `pip install -e .`.
- TensorBoard is used for logging; launch it against your log directory when training.


## Docker (reproducible environment)

Build a CPU-only image based on Python 3.12-slim with all project dependencies installed:

```bash
docker build -t ehc-sn:cpu .
```

Run training inside the container (mount logs to persist them on the host):

```bash
docker run --rm -it -v "$PWD/logs:/app/logs" ehc-sn:cpu \
	python experiments/dfaae_training.py
```

Regenerate figures from logs (mount figures to save outputs on the host):

```bash
docker run --rm -it \
	-v "$PWD/logs:/app/logs" \
	-v "$PWD/figures:/app/figures" \
	ehc-sn:cpu python figures/gen_timeseries.py
```

Notes:
- The provided image targets CPU for maximum portability. For GPU, build a derived image FROM an appropriate `nvidia/cuda:*` runtime and preinstall matching `torch/torchvision` wheels, then run with `--gpus all`.
- The image does not include a LaTeX toolchain. Build the paper on your host (see LaTeX setup below) or extend the image with TeXLive if needed.


## Datasets and masking

We generate obstacle maps from MiniGrid with deterministic per-index seeding and apply Full-Context Masked Training (FCMT). The model always observes the full input plus a mask channel, but the reconstruction loss is computed only on visible indices.

Quick visualizations:
- `examples/obstacle_maps.py` — sample unmasked maps into a PDF grid.
- `examples/masked_maps.py` — show (target, mask, masked input) triplets.

Run with defaults (CLI flags are provided via pydantic-settings):

```bash
python examples/obstacle_maps.py
python examples/masked_maps.py --mask_ratio 0.75 --keep_rects 2
```

## Training scripts

Each method shares the same geometry; only credit assignment and local objectives differ. Scripts log metrics and checkpoints to `./logs/` by default.

- BP baseline: `python experiments/bpae_baseline.py`
- DFA: `python experiments/dfaae_training.py`
- DRTP: `python experiments/drtpae_training.py`
- Hybrid (HyTA-AE): `python experiments/hytaae_training.py`

Common defaults (see supplement S1 for details):
- Optimizer: Adam with layer-wise learning rates (encoder/latent `2e-5`, decoder/head `1e-3`).
- Batch size: 32; budget: 200 epochs (~20k steps in figures).
- Width sweeps: separator (DG) `\ddg ∈ {64,128,250,500,1000,2000,4000,8000}`, latent (CA3) `\dca ∈ {100,200,400,800}`.

Fairness controls: in forward-only regimes, sparsity penalties applied to DG are strictly local; the BP baseline detaches gradients from the separator sparsity term to match this constraint.

## Reproducing figures

Once training logs exist under `logs/`, the plotting utilities regenerate the paper’s PDFs under `figures/`:

```bash
python figures/gen_timeseries.py
python figures/gen_contour.py
python figures/gen_heatmap.py
```

## LaTeX build (paper and supplement)

Compile the supplement first to generate the AUX file used by the main document’s cross-references, then compile the main paper:

```bash
cd article
latexmk -pdf -interaction=nonstopmode suplemental/supplemental.tex
latexmk -pdf -interaction=nonstopmode main.tex
```

## License

This project is licensed under the GNU GPLv3 (see `LICENSE`).

## Acknowledgments

If you use this code or ideas in your work, please cite the paper and/or link back to this repository. Portions of this README were assisted by GitHub Copilot.
