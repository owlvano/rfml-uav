# Design

## Context

See proposal.md (Why) and the explore findings it cites. Relevant state:

- **Clean features.** `results/paper3/phase1/features.npz` has 780k rows. Every recording has exactly 1,000 noise-only rows and ≤ 1,000 active rows, spread over the 2 s and sorted by `start` within each recording. The cache is therefore a time-ordered, subsampled stream per recording: noise and active windows interleave.
- **Phase 1 training.** Notebook 3 trains with `source_rows(data, seed)` (RandomUnderSampler on CLEAN rows) → `make_splits(..., "recording", seed)` → `train_best_val`. Run files store `test_idx`/`tgt_idx` and logits; checkpoints are `checkpoints/paper3/phase1/active_linear_s*_f*.pt`.
- **Copies of the undersampling step.** The Phase 1 undersampling step is copied verbatim in notebooks 0, 3 and 5; this change would be its fourth user.
- **Phase 2 builder.** Notebook 4's `build(dataset, groups, part_fn, cells)` extracts impaired features per recording with per-row seeding (`apply_cell`). Extending it to new row sets is one call.
- **Phase 3.** Notebook 5 loads models via `load_model(seed, fold)` and features via `features(target, rows)`; both are single functions to switch.

## Goals / Non-Goals

**Goals:**
- One small normalisation module that training, evaluation and the future OG-TCST stream share.
- Every comparison with the Phase 1 source paired by (seed, fold), on identical rows.

**Non-Goals:**
- Online noise/active labelling. Windows keep the offline `select_windows` labels, which use a recording-level percentile. A deployable detector would be the paper-2 cascade, and that belongs to the OG-TCST / E9 work.
- Tuning β or K per target. They are fixed (D2), with one sensitivity row.

## Decisions

### D1. `rfml_uav/data/normalise.py`: four functions, no classes, centring only
- `noise_mean(X)`: per-feature mean of noise windows.
- `centre(X, mu)`: `X − mu`, in float32.
- `session_normalise(X, rec_id, is_noise) -> Xn`: per recording, the mean of its own noise rows, subtracted from all of its rows.
- `stream_normalise(X, is_noise, beta, k) -> (Xn, valid)`, for one time-ordered stream:
  - warm-up: the mean of the first k noise windows;
  - after that, each noise window updates `mu ← (1−β)·mu + β·f`;
  - each window is centred with the state *before* its own update;
  - `valid` is False until warm-up completes.

**Why no division by σ (decision 2026-10-19, during apply).** The first implementation standardised by the noise std. On 25 runs it removed the shortcut (noise-only accuracy 0.43 → 0.20) and helped BLUE/BOTH (+4.4 / +3.0 pp), but:
- gain collapsed (−27 to −30 pp at every severity);
- the stream estimator lost up to 8 pp against Phase 1.

The noise σ doesn't react to impairments the way active windows do: a 3 dB gain step shrinks it to 0.56× of clean, so dividing by it inflates active features about 1.8×. A one-seed spike of centring alone beat Phase 1 by 9–19 pp on CLEAN, shift A, AWGN and ADC. The std variant is kept as an ablation (D7).

**Alternative:** a `RunningNorm` nn.Module inside the CNN. Rejected: the model and checkpoint format would change.

### D2. Stream parameters: β = 0.01, k = 50
- With 1,000 noise rows per recording, 1/β = 100 noise windows is about 10% of a recording. That is fast enough to follow a session change and slow enough to average out bursts.
- k = 50 excludes roughly the first 5% of each recording from stream scoring. The excluded fraction is reported.
- Sensitivity row: β ∈ {0.003, 0.03} on E1 BOTH and E2 awgn 3 only. (With std normalisation, β = 0.003 was best; the rows are re-run for centring.)
- **Simplification:** each recording is its own stream, so warm-up starts cold at every recording. Deployment would carry state across sessions; per-recording streams are the pessimistic case.

### D3. Shift B normalises impaired with impaired
- An impaired active row is normalised with the impaired noise rows of the same recording and cell. Both come from notebook 4's builder with the same `apply_cell` seeding, so the receiver change hits both alike. This is exactly the situation 3.2 is meant for.
- Noise rows go to `phase2/dronedetect_noise/{cell}.npz` (`X`, `row`), as a separate set. The existing active-row files and their row-alignment checks stay untouched (modified spec).
- One extra `build("dronedetect_noise", noise_groups, dd_part, CELLS + [("tilt", 4)])` cell in notebook 4. Recordings are read once more from the external drive.

### D4. Undersampling moves to `rfml_uav.eval.splits.undersample(idx, y, seed)`
- Four lines, verbatim from Phase 1 (`RandomUnderSampler(sampling_strategy="not minority", random_state=seed)`, sorted indices).
- Notebook 6 imports it. Notebooks 0, 3 and 5 keep their frozen inline copies, because their results are archived.
- Self-check: on the Phase 1 data, `undersample(...)` for seed 0 returns the rows of `test_idx ∪ train ∪ val` of Phase 1 seed 0. This proves that the folds are identical.

### D5. Notebook 6 layout and outputs (`results/paper3/phase4/`)
1. Session-normalise all 780k rows once, keeping the result in memory (≈ 800 MB float32).
2. Train 25 linear-tail models on the session-normalised CLEAN active rows. Checkpoints go to `checkpoints/paper3/noisenorm/active_linear_s*_f*.pt`. Run files go to `phase4/runs/` and mirror the Phase 1 format: `test_idx`, `test_logits`, `tgt_idx`, `tgt_logits` for session, plus `stream_logits` and `stream_valid` for all rows.
3. Shortcut re-test: same CNN and protocol on session-normalised CLEAN noise rows.
4. E1 table, both sources and both estimators, per subset and per mode, with paired differences.
5. E2 table: the active and noise impaired sets normalised by session and by stream, scored by both sources. The Phase 1 source is scored on the *unnormalised* impaired features, as in Phase 2.
6. Adoption decision (after notebook 5 has run with `SOURCE="noisenorm"`): pooled E3 source-only comparison per the spec. The result goes to `phase4/adoption.json`.

### D6. Notebook 5 gets `SOURCE = os.getenv("SOURCE", "phase1")`, the only new switch
- `load_model` picks the checkpoint directory.
- `features` returns session-normalised rows: impaired targets with their impaired noise rows, shift A and clean with the cached noise rows.
- The output directory becomes `phase3/` for the default source and `phase3/noisenorm/` for the new one.
- Splits come from `phase3/splits/` for both. Hyperparameters come from `phase3/selected_hparams.json`; there is no re-tuning (spec).
- With the default value, a re-run reproduces the archived Phase 3 outputs, because nothing else changes.

### D7. Std-normalised variant kept as an ablation
- Before retraining, `results/paper3/phase4/` is moved to `results/paper3/phase4_std/`, and `checkpoints/paper3/noisenorm/` to `checkpoints/paper3/noisenorm_std/`.
- Notebook 6's E1/E2 tables read the std rows from `phase4_std/` and add them as a third source column. There is no re-training and no E3 for the ablation.

### D8. Realistic AGC: rare steps, composed from existing files
- Per CLEAN row (active and noise), a fixed draw `default_rng([0, 99, row]).random() < 0.05` decides whether the row takes its `gain_s` impaired features or its clean features. The same mask is used for both sources and all severities 1–3.
- No new extraction: both the `gain_s` and the clean features already exist and are row-aligned.
- A constant gain offset needs no separate condition. `iq_wpd_features` is invariant to it (Phase 2 check), so slow AGC drift across windows is a no-op for DroneDetect features.

## Risks / Trade-offs

- **[Centring fails on gain steps in every window and on 20 dB tilt]:** the one-seed spike scored 0.10–0.15 on gain and 0.38 on tilt 4. → Reported as known failure modes. The D8 rare-step condition tests whether the gain failure comes from the adversarial "every window" construction. Tilt has no such excuse; it is a limitation of a mean-only correction under a strong shape change.
- **[Noise windows carry weak signal for near-continuous transmitters]:** AIR's noise recall was 0.71 in Phase 1. The statistics then hold a little class information, so the shortcut test may not reach chance. → Reported as is. The shortcut test result says how much remains.
- **[CLEAN accuracy drops]:** about 3.5 pp in the spike. → Accepted by the adoption rule: the paper argues robustness. The CLEAN difference is reported with the decision.
- **[Per-recording streams start cold]** → D2 simplification. The warm-up exclusion fraction is reported.
- **[Stream state is polluted by interference bursts in noise windows]:** windows below the floor percentile can still hold weak BT/Wi-Fi. → This is the intended behaviour: the floor *includes* the air. A slow β limits burst influence.
- **[Spike was one seed]:** the +3 pp shift-A gains may shrink over 25 runs. → The adoption rule decides on 25 runs, not on the spike. If it fails, OG-TCST builds on the Phase 1 source, and 3.2 is reported as a negative result with this analysis.

## Migration Plan

Additive. Notebook 5 and notebook 4 get default-preserving additions. Rollback = delete `checkpoints/paper3/noisenorm/`, `phase4/`, `phase3/noisenorm/` and `phase2/dronedetect_noise/`, and revert the notebook edits.

## Open Questions

- RQ numbering (RQ2 assumed). This doesn't change the work.
