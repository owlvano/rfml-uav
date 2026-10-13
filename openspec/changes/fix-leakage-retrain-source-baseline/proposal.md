# Proposal

## Why

Paper 3 (dissertation chapter 3, label-free adaptation) needs a trustworthy DroneRF source model, but the current pipeline cannot provide one. The inherited 97% accuracy (MoMLeT-2025) comes from a split by chunk. Chunks from one recording land in both train and test, near-duplicate L/H windows add more leakage, and the "best" checkpoint is chosen on test accuracy. The CNN also applies softmax before `nn.CrossEntropyLoss`, which weakens gradients and hides the logits that the paper's energy-score open-set gate needs. Every later Paper 3 phase (DroneDetect, impairments, TTA baselines, OG-TCST) inherits these defects, so they must be fixed first. The "chunk split vs recording split" table also serves as `classification-accuracy-validation` for the JMM paper (rfml-mobile-app).

## What Changes

- Chunking keeps each window's start sample index in the `chunked/*.npz` files. The `arr_0` array stays byte-identical, so the Paper 2 notebooks that read `chunked/` are unaffected.
- WPD feature extraction also writes one metadata-bearing file (`Wavelets/features.npz`). Each row records `bui`, `n` (recording index), `chunk_idx`, `start`, and an `is_dup` flag for windows overlapping a previously kept window by more than 50%. The legacy `Wavelets/{bui}.csv` output is still written.
- New shared evaluation code in `src/rfml_uav/eval/` builds train / val / test splits in two modes:
  - `recording` mode groups by `(bui, n)` with `StratifiedGroupKFold`;
  - `chunk` mode reproduces the legacy split, kept for comparison.
  
  Both modes carve a separate validation set out of train.
- **BREAKING (model output)**: `CNN.forward` and `RNN.forward` return logits, not probabilities. Saved weights stay loadable because softmax has no parameters. Any caller that treated the output as probabilities must apply softmax itself.
- The ONNX export wraps the model with softmax and keeps the output name `probs`, so rfml-mobile-app and `onnx_inference.py` see no contract change.
- New notebook `notebooks/paper3-label-free-adaptation/0_Source_Baseline.ipynb`:
  - retrains the WPD-CNN on DroneRF with 5 seeds × 5 outer folds in both split modes, with and without dedup;
  - selects checkpoints on validation only;
  - reports accuracy, macro-F1, per-class P/R/F1, confusion matrix and confidence histograms as mean ± std;
  - exports the recording-split source model to ONNX.
- Out of scope: PSD/STFT retraining, DroneDetect, any adaptation code, and changes to the published Paper 1/2 notebooks.

## Capabilities

### New Capabilities
- `drone-rf-chunk-metadata`: per-chunk provenance (recording, position, duplicate flag) carried from chunking through feature extraction.
- `recording-level-evaluation`: leakage-free train/val/test splitting by recording, validation-only model selection, and the standard source-baseline metric report.
- `classifier-logits-output`: classifier modules emit logits; probabilities are applied only at inference/export.

### Modified Capabilities
- (none; `openspec/specs/` is empty)

## Impact

- Code:
  - `src/rfml_uav/nn/cnn.py`, `src/rfml_uav/nn/rnn.py` (drop final softmax);
  - `src/rfml_uav/utils/export_onnx.py` (softmax wrapper);
  - new `src/rfml_uav/eval/splits.py` and `src/rfml_uav/eval/metrics.py`;
  - new `src/rfml_uav/data/chunking.py` (dedup helper).
- Notebooks:
  - `data-processing/1_Data_Processing_Chunking.ipynb` (save starts);
  - `data-processing/2_Data_Processing_Wavelets.ipynb` (write `features.npz`);
  - new `paper3-label-free-adaptation/0_Source_Baseline.ipynb`.
- Data: `chunked/` and `Wavelets/` must be regenerated (raw CSVs required). New `checkpoints/paper3/` and `results/paper3/` directories under `DATA_PATH`.
- Reproducibility: rerunning the Paper 1 notebooks after this change gives different numbers (CE now sees logits). Published numbers stay tied to the pre-change commit, which should be tagged.
- Dependencies: none new (`StratifiedGroupKFold` is in scikit-learn).
