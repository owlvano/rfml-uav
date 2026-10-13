# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RFML-UAV: PhD-level research repo for radio-frequency machine learning applied to UAV (drone) signal
detection and identification. Two ongoing research papers live side by side here:

- **paper1-uav-identification**: RF fingerprinting + deep learning (CNN/RNN) classification of drone
  identity from spectral features (PSD/STFT/Wavelets). Published: MoMLeT-2025 (CEUR Vol. 4004).
- **paper2-cascade-detection-method**: burst-aware cascade signal-presence detector (energy → burst
  aggregation → cyclostationary analysis) for computationally cheap UAV RF detection.

Experiments are notebook-driven; the `src/rfml_uav` package holds the shared, reusable code that
every notebook imports from — no experiment should reimplement detection/model logic inline.

## Environment setup

Conda-managed. No test suite, linter, or CI configured in this repo.

```bash
conda env create -f requirements.yml
conda activate rfml-uav
pip install -e .                                    # installs src/rfml_uav as editable package
python -c "import rfml_uav; print(rfml_uav.__file__)"  # verify install
```

`.env` must define `DATA_PATH` (see `.env.sample`), pointing at a directory containing a `raw/`
subfolder with the extracted DroneRF `.csv` files (dataset: https://data.mendeley.com/datasets/f4c2b4n755/1).
Every notebook and script reads `DATA_PATH` via `python-dotenv` + `os.getenv("DATA_PATH")` — there is
no config file, this env var is the single source of truth for data location.

All experimentation happens in Jupyter notebooks under `notebooks/`; there are no CLI entry points
or scripts to "run the pipeline" beyond executing notebooks in order.

## Data flow (matters for understanding any notebook)

`data/drone-rf/` (fully gitignored, see `.gitignore`) holds pipeline stages as sibling directories,
populated by running `notebooks/data-processing/*.ipynb` in numeric order:

```
raw/ → (1_Data_Processing_Chunking) → chunked/ (.npz, per UAV BUI + segment)
                                     → (2_Data_Processing_{PSD,STFT,Wavelets}) → PSD/ STFT/ Wavelets/
chunked/ → (3_Data_Balancing) → balanced/
```

`chunked/{bui}{L|H}_{n}.npz` holds chunks under positional key `arr_0` plus `starts` (window start
sample index, ascending). The Wavelets notebook also writes `Wavelets/features.npz` (`X, bui, n,
chunk_idx, start, is_dup`): the per-chunk provenance needed for recording-level splits (paper3).

Paper2's cascade detector notebook consumes `chunked/` directly (raw-ish signal chunks), not the
feature-transformed PSD/STFT/Wavelet directories — those are inputs to paper1's classifiers instead.

`archived/` holds the original `.rar` dataset archives; `utils/extract_dataset.py` and
`utils/flatten_file_structure.py` are one-off scripts (not a package API) for unpacking DroneRF's
nested archive structure into `raw/`.

## Package architecture (`src/rfml_uav`)

- **`drone_rf/consts.py`** — DroneRF dataset constants: `BUI_dict`/`BUI` (5-digit Binary UAV
  Identifier codes mapping to {Background, Bebop, AR, Phantom} drones), `FS_HZ`/`FS_MHZ` (80 MHz
  sample rate), FFT params (`M`, `Q`). This is the canonical BUI→drone-name mapping; every notebook
  that labels data by drone type imports from here rather than hardcoding BUI strings.
- **`drone_rf/utils.py`** — label derivation from BUI: `get_binary_label` (signal-present vs.
  background, for paper2), `get_class_label` (4-class drone identity, for paper1),
  `get_segment_count` (per-BUI segment counts — irregular, e.g. `10111` has 18 not 21),
  `get_input_dim` (expected feature-vector length per representation: PSD=2048, STFT=100,
  Wavelets=256).
- **`signal_detection/methods.py`** — the three detector primitives used standalone or as cascade
  stages: `energy_detector` (mean squared amplitude), `wavelet_detector` (DWT coefficient energy),
  `cyclo_detector` (blind spectral correlation function / cyclostationary detector — no prior
  knowledge of modulation params assumed).
- **`signal_detection/cascade.py`** — `cascade_detector`: orchestrates the paper2 pipeline —
  cheap `energy_detector` screens every chunk, `aggregate_bursts` groups consecutive detections into
  duration-constrained bursts (`min_burst_len`), then `cyclo_detector` runs *only* on burst regions
  (the computational-savings mechanism the paper evaluates). Pass `return_meta=True` to get
  per-burst scores/timings for the cascade selectivity analysis in the notebook.
- **`nn/models.py`** — `get_model(name, input_dim, n_classes)` factory dispatching to `"CNN"` or
  `"RNN"`; this is the only intended entry point notebooks use to construct paper1 models (avoid
  instantiating `CNN`/`RNN` directly in notebooks — go through the factory so architecture swaps
  stay a one-line change).
- **`nn/cnn.py`** / **`nn/rnn.py`** — 1D-CNN and 2-layer-LSTM classifiers over spectral feature
  vectors, both returning 4-class **logits** (background/Bebop/AR/Phantom via `get_class_label`);
  softmax is applied only at inference / in `utils/export_onnx.py` (ONNX output `probs`). Published
  MoMLeT-2025 numbers come from tag `momlet-2025` (softmax-before-CE, chunk split).
- **`data/chunking.py`** — `flag_duplicates(starts, chunk_size)`: marks windows overlapping the last
  kept window of the same recording by >50% (L/H offset near-duplicates).
- **`eval/splits.py`** — `make_splits(y, groups, mode)`: K outer test folds + val carved from train;
  `mode="recording"` groups by `f"{bui}_{n}"` (leakage-free), `mode="chunk"` is the legacy paper-1 split.
- **`eval/metrics.py`** — `classification_report_dict`: accuracy, macro-F1, per-class P/R/F1,
  confusion matrix, max-softmax confidence split by correct/wrong.

There is no top-level package API exposed in `rfml_uav/__init__.py` (it's empty) — everything is
imported from its submodule directly, e.g. `from rfml_uav.signal_detection.cascade import cascade_detector`.

## Working in notebooks

- Reproduce results by running notebooks in their numeric/dated prefix order within each directory.
- `notebooks/paper2-cascade-detection-method/review-1/` and `review-2/` hold review-round scratch
  notebooks (stress tests, timing analysis) — treat as working notes tied to specific PR/review
  rounds, not the canonical pipeline.
- When adding detection logic or model code referenced from more than one notebook, put it in
  `src/rfml_uav` and import it — don't duplicate implementations across notebooks.
