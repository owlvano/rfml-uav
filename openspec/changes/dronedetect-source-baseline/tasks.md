# Tasks

## 1. Shared code moved into src

- [x] 1.1 Add `src/rfml_uav/data/wavelets.py` with:
  - `wpd_features(x, wavelet="dmey", level=6)`, moved verbatim from `2_Data_Processing_Wavelets.ipynb`;
  - `iq_wpd_features(window)`, which removes DC, scales to unit power and concatenates I ‖ Q.

  Add an `assert`-based `__main__` self-check:
  - output length is 256;
  - `iq_wpd_features(g·w + c)` equals `iq_wpd_features(w)` within 1e-4 for gain g = 7 and DC c = 0.05+0.09j.

  Verify `python -m rfml_uav.data.wavelets` passes.
- [x] 1.2 Swap the inline WPD function in `2_Data_Processing_Wavelets.ipynb` for an import of `wpd_features`, without re-running the full notebook. Verify that on 3 chunks of `chunked/10000L_0.npz` the imported function gives features identical to the matching rows of `Wavelets/features.npz`.
- [x] 1.3 Add `src/rfml_uav/nn/train.py` with `train_best_val(model, X, y, tr, va, epochs, lr, batch_size, seed, device)` (returns the model at the best val macro-F1 epoch, plus that F1 and the epoch) and `predict_logits(model, X)`. The logic is moved from `0_Source_Baseline.ipynb`. Verify a 2-epoch run on 2000 random DroneRF feature rows completes and returns logits of shape `(n, 4)`.

## 2. Logit tail and export options

- [x] 2.1 Add `linear_tail=False` to `CNN`: when true, `bn7`/`prelu4` are not created and `forward` returns `fc4`. Pass `**kwargs` through `get_model`. Verify:
  - the default state-dict keys are unchanged;
  - `checkpoints/v2/checkpoint_wpd_best.pt` and `checkpoints/paper3/source_wpd_cnn.pt` load;
  - a `linear_tail=True` model's output equals its `fc4` output on a random batch.
- [x] 2.2 Add `--n-classes`, `--input-dim` and `--linear-tail` to `export_onnx.py`, with defaults equal to the current behaviour. Verify re-exporting `checkpoints/paper3/source_wpd_cnn.pt` to a scratch path with default flags gives outputs identical to the existing `source_wpd_cnn.onnx`.
- [x] 2.3 Update the `nn/` and `data/` lines of the package section in `CLAUDE.md` (`train.py`, `wavelets.py`, `linear_tail`); verify the lines are present.

## 3. DroneDetect loader and window selection

- [x] 3.1 Add `src/rfml_uav/drone_detect/__init__.py` and `dataset.py` with the `DRONES`/`SUBSETS`/`MODES` constants and `inventory(root)` (folder = label; raises on a code/folder mismatch; `rec_id = f"{subset}_{drone}_{mode}_{k}"`). Verify on `DRONEDETECT_PATH`:
  - 390 rows and 7 drones;
  - `MP1` rows have file prefix `MA1`;
  - a synthetic mismatched path raises.
- [x] 3.2 Add `read(path)` (memmap complex64, actual length) and `select_windows(x, size=4000, cap=1000, floor_pct=10, margin_db=6)` per design D2. The `__main__` self-check runs on a synthetic signal (noise plus bursts in 25% of windows, plus a DC offset) and checks:
  - the active starts all fall inside bursts;
  - the noise starts all fall outside;
  - both lists respect the cap and are spread across the signal;
  - the DC offset alone doesn't create active windows.

  Verify `python -m rfml_uav.drone_detect.dataset` passes.
- [x] 3.3 Add the DroneDetect module to the package section of `CLAUDE.md`, and mention `DRONEDETECT_PATH` in its environment section; verify both are present.

## 4. Feature extraction

- [x] 4.1 Create `notebooks/paper3-label-free-adaptation/2_DroneDetect_Features.ipynb`, which runs the per-recording feature pass with `joblib` (8 jobs) into `results/paper3/phase1/features/{rec_id}.npz` and skips existing files (D3). Verify a run limited to 2 recordings writes 2 files with `X` of shape `(≤2000, 256)` and both the active and noise flags.
- [x] 4.2 Run the full pass and concatenate into `results/paper3/phase1/features.npz`. Verify:
  - 390 recordings are present;
  - `start` is ascending within each `rec_id`;
  - no NaNs.

  Print and record in the notebook the active fraction and the kept active/noise counts per drone × subset.

## 5. Source baseline and interference table

- [x] 5.1 Create `3_DroneDetect_Source_Baseline.ipynb`:
  - CLEAN-only undersampling and `make_splits(..., "recording")`, asserting that no non-CLEAN row is in train or val;
  - the training grid via `train_best_val` (2 tails × 5 seeds × 5 folds);
  - per-run files with logits for CLEAN test and target rows (D4), resumable.

  Verify a single run (default tail, seed 0, fold 0, 2 epochs) writes its file with the expected keys.
- [x] 5.2 Run the main grid; verify 50 run files exist.
- [x] 5.3 Render and save to `results/paper3/phase1/`:
  - `interference_table.csv`: per subset, mean ± std accuracy, macro-F1, per-class F1, and the drop vs CLEAN with Wilcoxon p and bootstrap CI;
  - `interference_by_mode.csv`;
  - the PHA FY unseen-condition line, excluded from both tables;
  - aggregated confusion matrices per subset (PNG).

  Verify the files exist, the main table has 4 rows per tail, and PHA FY is absent from it.
- [x] 5.4 Compute energy-score and max-softmax AUROC (correct vs wrong) per tail, on CLEAN test and per subset, into `energy_auroc.csv`. Apply the tail-choice rule from the spec and print the chosen tail. Verify the CSV has 2 tails × 4 subsets and the notebook states the choice.

## 6. Receiver-shortcut test

- [x] 6.1 Run the noise-only grid (default tail × 5 seeds × 5 folds) on CLEAN noise rows. Report 7-class accuracy vs 1/7, receiver-group accuracy vs 1/2 and per-class recall into `shortcut_test.csv`. Verify 25 run files exist and the CSV states whether 7-class accuracy exceeds 2/7.
- [x] 6.2 Only if 6.1 flags a shortcut: rebuild the inputs with in-band nodes only (120-d, D6), re-run the shortcut test and the main grid with the chosen tail, and save `*_inband.csv` alongside. Verify both results are reported side by side. If no shortcut is flagged, mark this done with a one-line note in the notebook.

## 7. Export and write-up

- [x] 7.1 Export the chosen source model (seed 0, chosen tail, best val macro-F1 fold, full-band 256-d input: the in-band mitigation of 6.2 did not remove the shortcut and lowered macro-F1 on every subset, so per the author's decision on 2026-10-15 the full-band model is kept) to `checkpoints/paper3/dronedetect_source_wpd_cnn.{pt,onnx}` via `export_onnx.py --n-classes 7 ...`. Verify `onnxruntime` runs it on 10 feature rows and the `probs` rows sum to 1.
- [x] 7.2 Add a Phase 1 section to `notebooks/paper3-label-free-adaptation/README.md` covering:
  - the protocol;
  - the interference table headline (macro-F1 per subset, drop and p);
  - the shortcut verdict;
  - the tail choice;
  - limitations: no Background class, shift A confounded with gain, recordings likely from one session.

  Update the "Next session" section of `NOTES.md`. Verify the README numbers match the CSVs.

## Workflow follow-up

- Hand the interference table to the paper (section 6.1) and the thesis (section 3.1).
- Archive after review. The next change covers shift B (receiver-impairment simulator), then adaptation baselines.
