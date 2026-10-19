# Design

## Context

See proposal.md (Why). Existing state this design builds on:

- **DroneDetect.** `results/paper3/phase1/features.npz` holds provenance per row (`subset, drone, mode, k, rec_id, start, active`). Raw windows are re-read with `drone_detect.dataset.read(path)[start:start+4000]`, using the path from `inventory(DRONEDETECT_PATH)` joined on `rec_id`. Phase 1 run files (`runs/active_linear_s{seed}_f{fold}.npz`) store `test_idx`, which indexes `features.npz`. Checkpoints are in `checkpoints/paper3/phase1/active_linear_s*_f*.pt`.
- **DroneRF.** `Wavelets/features.npz` holds `bui, n, chunk_idx, start, is_dup`. Raw windows are `chunked/{bui}{L|H}_{n}.npz['arr_0'][chunk_idx]`: real float64 values, the ADC's small integers. Phase 0 run files store `y_true, probs, test_groups` but **not** row indices. Checkpoints are in `checkpoints/paper3/phase0/recording_dedup_s*_f*.pt` (default tail, 4 classes).
- **Feature functions.** `iq_wpd_features` removes DC and scales to unit power. `wpd_features` has no normalisation.
- **Speed.** WPD costs ≈ 2.4 ms per window (measured), so the 12 cells per dataset take minutes, not hours.

## Goals / Non-Goals

**Goals:**
- One small, pure, deterministic impairment function that both datasets share, with a self-check that tests the physical parameters.
- Impaired sets that later E3 notebooks can load as plain feature matrices, row-aligned with the clean ones.

**Non-Goals:**
- Recording-level time models (slow drift across windows), CFO and IQ imbalance. Per-window `iq_wpd_features` treats a CFO as a spectral shift, and per-window unit power hides slow drift. Both need a stream simulator, which is the E3/OG-TCST stream work.
- Severity calibration against a real second receiver.

## Decisions

### D1. One function: `impair(x, kind, severity, rng) -> x'`, with params in a module-level dict
`SEVERITY = {"awgn": [None, 20, 10, 5, 0], ...}`. Index 0 means identity: the function returns `x` unchanged. This keeps the spec table in one place and lets the notebook loop `itertools.product(SEVERITY, range(1, 4))`.
*Alternative:* one class per impairment. Rejected: 4 callables with no shared state don't need one.

### D2. Impairment definitions (per window, power measured after DC removal)
Let `P = var(x)` (complex variance for I/Q), `N = len(x)`.
- **awgn:** `x + sqrt(P / 10^(snr/10)) · n`. `n` is unit-variance real noise, or circular complex noise (`(a + jb)/√2`) for complex windows.
- **gain:** `x · 10^(±g/20)` for the whole window, with the sign drawn. Then a step of a further ±g dB at a position uniform in `[N/4, 3N/4)`: `x[t0:] *= 10^(±g/20)`, with an independent sign.
  - On DroneDetect only the step survives unit-power normalisation. On DroneRF both parts move the features.
  - The gain-invariance check (spec) uses the offset alone, via a `step=False` argument used only by that check.
- **tilt:** multiply the FFT of the window by a linear-in-dB ramp across the band. Real windows: `rfft` bins 0 … fs/2. Complex windows: `fftshift` bins −fs/2 … +fs/2. The total edge-to-edge span is ±T dB, with a random sign. The ramp is centred on 0 dB, so tilt doesn't add a gain change.
  - Circular filtering through the FFT (instead of a FIR) is exact for the spec test and has no transient. Window edges wrap, which is acceptable for a 4000-sample window.
- **adc:** full scale `A = k · RMS` per real component (I and Q separately). Clip to `[−A, A]`, then quantise to `2^bits` uniform levels: mid-rise, step `2A / 2^bits`, values `(floor(x/Δ) + 0.5)·Δ` clipped to the outermost levels.
  - Full scale follows the window RMS, so this models a receiver whose AGC sets the ADC range per window. That keeps gain and ADC independent.
  - The more realistic fixed full scale would mix the two impairments, and the per-impairment table would no longer separate them.

### D3. Randomness: `np.random.default_rng([seed, KIND_ID[kind], severity, row_id])`
`row_id` is the row's index in the clean feature file. One generator per (row, cell) makes results independent of order and of joblib worker count, which the spec requires. The base seed is fixed at 0. Impairment noise is not varied across the 5 model seeds: the 25-run spread then measures model variance on one fixed shifted test set, the same way Phase 1 scores a fixed BLUE/WIFI/BOTH set.

### D4. Rows impaired
- **DroneDetect:** all CLEAN active rows. Every Phase 1 test row is among them, and E3 will need the remaining rows as unlabeled target data.
- **DroneRF:** all non-duplicate rows (`~is_dup`, ≈ 164k).
- Each dataset is impaired once per cell and shared by all 25 models. Each model then indexes its own test rows.

### D5. Output layout
`DATA_PATH/results/paper3/phase2/{dronedetect,dronerf}/{kind}_{severity}.npz` holds `X` (float32, rows × 256) and `row` (the clean-file indices). Results and tables go in `phase2/*.csv`.
- Severity 0 is not stored: it equals the clean `X[row]`.
- Each file is written with tmp + `os.replace`, as in notebook 2, and the notebook skips files that already exist, so interrupted runs resume.
- Disk: ≈ 100 MB (DroneDetect) and ≈ 170 MB (DroneRF) per cell, so ≈ 3.2 GB for 24 cells. Acceptable on `DATA_PATH`.

### D6. Recovering DroneRF Phase 0 test rows
Phase 0 runs don't store indices. The notebook recomputes them with Phase 0's `build_dataset(dedup=True, seed)` followed by `make_splits(..., "recording", seed=seed)`; both are deterministic. It asserts that the recomputed `y_true` and `test_groups` equal the stored ones before scoring. `build_dataset` is copied into notebook 4 as 4 lines. It isn't moved to `src`: notebook 0 is frozen, and no third user is planned.
*Alternative:* re-run Phase 0 to save indices. Rejected: it costs an hour of GPU time for information that can be recovered deterministically.

### D7. Scoring
Load each checkpoint through `get_model("CNN", 256, n_classes, linear_tail=…)`: True for Phase 1, False for Phase 0. Use `predict_logits` on the impaired rows. Per run, store `logits` for every cell in one `phase2/runs/{dataset}_s{seed}_f{fold}.npz`, keyed by cell. The report is then rebuilt from logits without touching the GPU, the same split as Phase 1.
- energy score: `−logsumexp(z)`;
- max-softmax: `max softmax(z)`;
- AUROC: `sklearn.metrics.roc_auc_score`, the same as Phase 1's `auroc` helper.

### D8. Shared paired statistics → `rfml_uav.eval.stats.paired_test(a, b, n_boot=10_000, seed=0)`
It returns the mean difference, the 95% percentile-bootstrap CI and the exact two-sided Wilcoxon p. Notebooks 0 and 3 each have an inline copy. This notebook is the third user, and E3 ("Wilcoxon vs best baseline") the fourth, so the CLAUDE.md rule applies. Notebooks 0 and 3 are **not** edited, because their results are archived. Only new notebooks import the helper. A `__main__` self-check covers it: a known shift gives p < 0.05 and a CI that excludes 0, and identical inputs give a difference of 0.

### D9. Escalation is done by hand, in the notebook
After the severity 1–3 table, a cell lists the impairments whose DroneDetect severity-3 drop is below 3 pp. For those it runs severity 4 for both datasets through the same extract + score cells. This needs no extra machinery, and the rule stays visible in the notebook.

## Risks / Trade-offs

- **[Severities too mild for DroneDetect]:** active windows are already ≥ 6 dB above the floor, and the features normalise power, so AWGN at 20 dB may do almost nothing. → Severity 3 sits at 5 dB, and the escalation rule adds 0 dB if needed. A small drop at severity 1 is a finding in itself: that is the robustness margin.
- **[DroneRF ADC on already-coarse integers]:** DroneRF samples are small integers, so 10-bit re-quantisation at 4×RMS can be finer than the original grid. Severity 1 may then be a near no-op. → This is reported as is. The identity and parameter checks still hold, because full scale is relative.
- **[DroneRF gain is trivially destructive]:** with unnormalised WPD, any gain offset scales every feature, so DroneRF drops may be dominated by the offset. → The gain-invariance check separates offset from step, and the paper presents it as a feature-design finding (normalisation is a cheap first defence), not as a weakness of adaptation.
- **[External drive I/O]:** DroneDetect reads go to `/media/.../KINGSTON`. → Read once per recording and impair all 12 cells from the windows in memory (per-recording task), so the drive is read once in total, not 12 times.
- **[Circular tilt wrap-around]** → acceptable, see D2. The windows are independent and are never concatenated.
- **[One fixed impairment draw]:** the CI covers model variance, not impairment-draw variance. → Stated in the report. If a reviewer asks, a second draw seed on the best model is cheap.

## Migration Plan

Additive only: a new module, a new notebook and a new results directory. Rollback = delete `phase2/` and the new files. Nothing existing is modified.
