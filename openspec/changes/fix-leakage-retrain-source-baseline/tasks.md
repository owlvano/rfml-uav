# Tasks

## 1. Preserve published state

- [x] 1.1 Create local git tag `momlet-2025` on the pre-change commit; verify `git tag --list momlet-2025` shows it (pushing is the author's call)
- [x] 1.2 Record sha256 of `chunked/{bui}L_{n}.npz` `arr_0` arrays for 3 recordings (one background, one Bebop, one Phantom) to `DATA_PATH/results/paper3/phase0/arr0_checksums.txt`; verify file lists 3 hashes

## 2. Logits output and ONNX export

- [x] 2.1 Remove final `F.softmax` from `src/rfml_uav/nn/cnn.py` and `src/rfml_uav/nn/rnn.py`; verify loading `checkpoints/v2/checkpoint_wpd_best.pt` succeeds and argmax on 100 `balanced/Wavelets.npz` rows equals argmax of the ONNX `checkpoint_wpd_best.onnx` exported before the change
- [x] 2.2 Wrap the model in `nn.Sequential(model, nn.Softmax(dim=1))` in `src/rfml_uav/utils/export_onnx.py`, and add argparse `--ckpt/--onnx` defaulting to the current v2 paths; verify `python -m rfml_uav.utils.export_onnx --onnx <scratch>.onnx` re-exports v2 to a scratch path (MoMLeT `checkpoints/v2/*` left untouched) with output identical to the original v2 ONNX, and `onnx_inference.py --feature wpd` prints confidences in [0,1] with output name `probs`
- [x] 2.3 Add a note to `notebooks/paper1-uav-identification/README.md` that published numbers correspond to tag `momlet-2025` (softmax-before-CE); verify note present

## 3. Chunk provenance and dedup

- [x] 3.1 Add `src/rfml_uav/data/__init__.py` and `src/rfml_uav/data/chunking.py` with `flag_duplicates(starts, chunk_size, max_overlap=0.5)` plus an `assert`-based `__main__` self-check (starts 1000/1100 → second flagged; 0/2000 → none flagged; contiguous background starts → none flagged); verify `python -m rfml_uav.data.chunking` passes
- [x] 3.2 Update `1_Data_Processing_Chunking.ipynb` to save `starts=` (and starts `0, chunk_size, …` for background) alongside positional `arr_0`; re-run the notebook; verify the checksums from 1.2 match and `np.load(...)["starts"]` length equals `len(arr_0)` for every file
- [x] 3.3 Update `2_Data_Processing_Wavelets.ipynb` to also write `Wavelets/features.npz` (`X, bui, n, chunk_idx, start, is_dup`) from the same loop that writes `{bui}.csv`; re-run; verify row count equals the sum of CSV rows, `start` is ascending within each `(bui, n)`, and print the `is_dup` fraction per BUI (record it in the notebook output)

## 4. Recording-level evaluation code

- [x] 4.1 Add `src/rfml_uav/eval/__init__.py` and `src/rfml_uav/eval/splits.py` with `make_splits(y, groups, mode, n_folds=5, val_frac=0.2, seed=0)` per design D5. Assert that every class appears in each test fold. The `__main__` self-check runs on synthetic groups and checks:
  - recording mode: train/val/test groups are pairwise disjoint;
  - chunk mode: sets are disjoint by index;
  - the same seed gives identical indices.

  Verify `python -m rfml_uav.eval.splits` passes.
- [x] 4.2 Add `src/rfml_uav/eval/metrics.py` with `classification_report_dict(y_true, y_pred, probs, class_names)` per design D8, plus a `__main__` self-check on a hand-made 4-class example (perfect predictions → acc 1, macro-F1 1, all max-probs in the correct bucket); verify `python -m rfml_uav.eval.metrics` passes
- [x] 4.3 Add the `eval/` and `data/` modules to the package architecture section of `CLAUDE.md`; verify the section lists both

## 5. Source baseline notebook

- [x] 5.1 Create `notebooks/paper3-label-free-adaptation/0_Source_Baseline.ipynb` with:
  - loading of `Wavelets/features.npz`;
  - an optional `is_dup` filter;
  - metadata-preserving undersampling (D4);
  - a training loop via `get_model("CNN", …)` with best-val-macro-F1 epoch tracking (D6).

  Verify a single run (recording mode, seed 0, fold 0, 2 epochs) completes and writes its `results/paper3/phase0/*.npz`.
- [x] 5.2 Run the full grid: 2 modes × 2 dedup × 5 seeds × 5 folds, with resume-by-existing-file (D9). Verify all 100 result files exist.
- [x] 5.3 Render and save the outputs to `results/paper3/phase0/`:
  - the leakage table (mean ± std accuracy, macro-F1 and per-class P/R/F1 per mode × dedup), as CSV;
  - aggregated confusion matrices;
  - max-softmax confidence histograms (correct vs wrong), as PNG.

  Verify the files exist and the table has 4 rows.
- [x] 5.4 Export the selected source model (seed 0, recording mode, dedup on, best val macro-F1 fold) to `checkpoints/paper3/source_wpd_cnn.{pt,onnx}` via `export_onnx.py --ckpt/--onnx`; verify `onnx_inference.py --onnx checkpoints/paper3/source_wpd_cnn.onnx --feature wpd` runs
- [x] 5.5 Add `notebooks/paper3-label-free-adaptation/README.md` stating the Phase 0 protocol, the headline numbers from 5.3 and the gate outcome (whether the recording split drops accuracy substantially → new source model); verify the README numbers match the CSV

## Workflow follow-up

- Hand the 5.3 leakage table to rfml-mobile-app `classification-accuracy-validation` (JMM).
- Archive this change after review; next change: Phase 1 DroneDetect loader + I/Q 2-channel WPD.
