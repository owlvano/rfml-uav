# Design

## Context

See proposal.md (Why) and `1_DroneDetect_Exploration.ipynb` for the dataset facts. Current state of the code:

- **WPD.** WPD lives inline in `data-processing/2_Data_Processing_Wavelets.ipynb`. `wavelet_packet_decomposition(signal, 'dmey', level=6)` gives 64 frequency-ordered nodes × (RMS, STD) = 128 values per real channel; DroneRF concatenates L ‖ H into 256. Cost is about 0.7 ms per window per channel.
- **Training loop.** `train_run` / `predict` (Adam, lr 0.01, batch 256, 50 epochs, best val macro-F1 epoch) also live inline, in `0_Source_Baseline.ipynb`.
- **CNN.** The tail is `fc4 → bn7 → prelu4`. `get_model(name, input_dim, n_classes)` takes no extra options.
- **Reusable as is:**
  - `make_splits(y, groups, "recording")` is generic over group labels;
  - `classification_report_dict` is generic over class names.
- **Not reused:**
  - `flag_duplicates` isn't needed, because windows here are non-overlapping by construction;
  - `energy_detector` computes `mean(x**2)`, which is wrong for complex input and doesn't remove DC. The same statistic (DC-removed mean power) is computed vectorised over all windows instead.
- **Data.** 390 recordings of ~960 MB on an exFAT USB drive, at ~290 MB/s sequential.

## Goals / Non-Goals

**Goals:**
- One shared WPD implementation for both datasets.
- A feature pass over the external drive that can be resumed.
- Saved logits per run, so AUROC and later analyses need no re-scoring (Phase 0 saved only probabilities).

**Non-Goals:**
- Tuning window length, WPD level or the CNN for DroneDetect. DroneRF settings are kept so the two datasets stay comparable.
- Per-recording or running feature normalisation. That is the adaptation method (thesis 3.2), and putting it into the baseline would hide the shift that RQ1 measures.
- Re-running Phase 0 or regenerating DroneRF features.

## Decisions

### D1. Module layout
- **`src/rfml_uav/drone_detect/dataset.py`**:
  - `inventory(root) -> DataFrame`: path, subset, drone, mode, k, n_samples, rec_id;
  - `read(path) -> np.memmap[complex64]`;
  - `select_windows(x, size=4000, cap=1000, floor_pct=10, margin_db=6) -> (active_starts, noise_starts, active_frac)`.

  Also add `drone_detect/__init__.py` and `DRONES` / `SUBSETS` / `MODES` constants. `load_data.py` and `instructions.md` stay untouched; nothing imports `load_data.py`.
- **`src/rfml_uav/data/wavelets.py`**:
  - `wpd_features(x, wavelet="dmey", level=6)`: moved verbatim from the DroneRF notebook;
  - `iq_wpd_features(window)`: DC removal, unit power, then `wpd_features(I) + wpd_features(Q)`.

  The DroneRF notebook switches to importing `wpd_features`, so the two datasets can't drift apart.
- **`src/rfml_uav/nn/train.py`**: `train_best_val(...)` and `predict_logits(...)`, moved from the Phase 0 notebook. Phase 0's notebook stays as the record of that run.
- *Alternative:* keep the code inline in the new notebooks. Rejected: CLAUDE.md requires shared logic in `src`, and WPD is now used by two datasets.

### D2. Window selection
- **Power statistic.** Per-window power is the variance (`mean|x|² − |mean x|²`), computed on non-overlapping 4000-sample windows. The recording is read in blocks of about 1e7 samples, so memory stays small.
- **Why variance.** Group A (AIR/DIS/PHA) has a DC offset (|DC| ≈ 0.05) of the same order as its RMS. Raw mean power would treat DC as signal.
- **Thresholds.** Floor = 10th percentile of window power. Active means ≥ floor × 10^(6/10); noise means ≤ floor.
- **Cap.** At most 1000 active and 1000 noise windows per recording, taken at evenly spaced positions in the sorted candidate list. That gives about 390k active rows, or 400 MB as float32.
- *Alternative:* the DroneRF `get_chunk_ids` packet detector. Rejected: it is tuned to real-valued DroneRF levels (`min_power=200`), and it produces overlapping windows.
- *Alternative:* the paper-2 cascade. Rejected: too heavy for a labelling step, and it would couple Paper 3 results to Paper 2 settings.

### D3. Feature pass
- Run `joblib.Parallel(n_jobs=8)` over recordings. Each job reads, selects and extracts one recording, then writes `results/paper3/phase1/features/{rec_id}.npz`.
- Existing files are skipped, so a run can resume.
- A final cell concatenates the per-recording files into `features.npz`, with columns `X, subset, drone, mode, k, rec_id, start, active`.
- Eight jobs saturate the USB drive; more would only add seeks.

### D4. Source protocol
- **Training data.** CLEAN active rows only. Every class is undersampled to the smallest with the run seed (`RandomUnderSampler`, `"not minority"`), before splitting, as in Phase 0.
- **Splits.** `make_splits(y, rec_id, "recording", n_folds=5, seed)`. DIS and PHA have 10 CLEAN recordings each, so each test fold gets 2 per class.
- **Target scoring.**
  - All BLUE, WIFI and BOTH active rows, not undersampled; macro-F1 absorbs the imbalance.
  - The CLEAN test fold, scored as is.
  - PHA FY rows are excluded from the interference table and reported on their own.
- **Grid.**
  - main: 2 tails × 5 seeds × 5 folds = 50 runs;
  - shortcut test: default tail × 5 × 5 = 25 runs.
- **Per-run file.** `results/paper3/phase1/runs/{data}_{tail}_s{seed}_f{fold}.npz` holds:
  - `logits` and `y` for the CLEAN test rows and for the target rows;
  - row indices into `features.npz`;
  - val F1 and best epoch.

  Runs whose file exists are skipped.

### D5. Logit tail
- `CNN(input_dim, n_classes, linear_tail=False)`. With `True`, `bn7` and `prelu4` are not created, and `forward` returns the output of `fc4`.
- `get_model(name, input_dim, n_classes, **kwargs)` passes the option through.
- Not creating the modules (rather than replacing them with `Identity`) keeps the default state dict byte-compatible.
- **Tail choice** follows the rule in the spec: highest pooled-interference energy AUROC, unless CLEAN macro-F1 is more than 1 pp lower.

### D6. Shortcut test and mitigation
- **Test.** Train on CLEAN noise-only rows with the same protocol. Group accuracy is derived by mapping each predicted drone to its receiver group; no second model is trained.
- **Mitigation, if 7-class accuracy > 2/7.** Keep only in-band WPD nodes:
  - each real channel's 64 nodes span 0–30 MHz (~0.47 MHz each);
  - the 28 MHz receiver band is |f| ≤ 14 MHz, i.e. nodes 0–29;
  - the input becomes 2 channels × 30 nodes × 2 stats = 120.
- **Re-run after mitigation.** Repeat the shortcut test, plus the main grid with the chosen tail only (2 × 25 runs). The CNN already sizes itself from `input_dim`.
- **Outcome (2026-10-15).** In-band cropping lowered noise-only accuracy only from 0.43 to 0.36 (group accuracy 0.98 → 0.96), so the shortcut is still flagged. It also cost macro-F1 on every subset (CLEAN −2.6 pp, interference −5 pp). Author's decision: keep the full-band 256-d source model, and report the shortcut as a DroneDetect property for the adaptation method to address.

### D7. Statistics and model export
- **Statistics.** For each subset, compare CLEAN-test vs subset macro-F1 from the same model, paired over the 25 (seed, fold) runs: Wilcoxon plus a 95% bootstrap CI, as in Phase 0 D10.
- **Model export.**
  - Choose seed 0, the chosen tail, and the fold with the best val macro-F1.
  - Export to `checkpoints/paper3/dronedetect_source_wpd_cnn.{pt,onnx}` via `export_onnx.py --ckpt/--onnx`.
  - This export is the E9 energy model, and the DroneRF exports are untouched.
  - `export_onnx.py` must pass the tail flag and the class count. Today it builds a 4-class CNN.

## Risks / Trade-offs

- [The 10th-percentile floor holds signal for drones that transmit almost continuously (AIR, PHA in CLEAN)] → Active selection becomes stricter, and "noise" windows may contain weak signal, which inflates the shortcut test for those drones. Report the active fraction and noise-only per-class recall per drone so this can be seen.
- [PHA's CLEAN files are dominated by a wideband plateau, and the gain steps inside files may be selected as "active"] → The shortcut test and per-class F1 expose it. If needed, state it as a dataset limitation; don't hand-tune PHA.
- [In-band cropping won't remove in-band floor-shape differences] → If the shortcut survives cropping, report it openly as a property of DroneDetect. Per-recording normalisation is deliberately left to the adaptation method (thesis 3.2), and the paper can show that adaptive normalisation removes it.
- [The 5 recordings per condition are likely consecutive 2-s captures of one session] → The recording split is optimistic, as with DroneRF. State this as a limitation; no session split is possible.
- [Shift A also includes a gain/session change (exploration)] → Report shift A as interference plus session change, not pure interference.
- [The external drive is unplugged or slow] → The per-recording feature cache allows a resume. After the feature pass, nothing reads the raw data again.
- [Moving `train_run` into `src` changes nothing numerically but leaves the Phase 0 notebook with its own inline copy] → Accepted. Phase 0 is archived and its results are frozen.

## Migration Plan

All files are new, except:
- `cnn.py` / `models.py`: additive option, default unchanged;
- `export_onnx.py`: new options, defaults unchanged;
- the DroneRF WPD notebook: import swap only. Verify features are identical on 3 chunks; don't regenerate.

Rollback: revert the commit. Phase 0 artifacts are untouched.

## Open Questions

- Per-recording cap: 1000 windows is a starting point. If training is fast and the cache is small, raising it doesn't change the specs.
