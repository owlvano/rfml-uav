# Design

## Context

See proposal.md (Why, including both spikes). Building blocks that exist today:

- **Centred source.** `checkpoints/paper3/noisenorm/active_linear_s*_f*.pt` (25 models, linear tail). Their run files (`results/paper3/phase4/runs/active_linear_s*_f*.npz`) store `test_idx`. Train/val rows are rebuilt with `undersample` + `make_splits` (Phase 4 checks that this reproduces them).
- **Centring.** `rfml_uav.data.normalise.stream_normalise` does EMA centring over a time-ordered stream with noise flags.
- **Statistics pass.** `rfml_uav.adapt.baselines.adabn` re-estimates BN statistics with a shuffled cumulative pass (fixed in Phase 3); `tent` and `pseudo_label` are offline variants.
- **Splits.** `results/paper3/phase3/splits/{target}_s*_f*.npz` exist for every target and dev cell except `agc_3`. `agc_3` uses the shift-B rule (the model's test recordings), so its split is identical to any other shift-B target's for the same (seed, fold).
- **Impaired features.** Active and noise windows for every shift-B cell are in `phase2/dronedetect{,_noise}/`. `agc_3` is composed from `gain_3` and the clean features with the fixed 5% mask (notebooks 5/6).
- **Cached windows.** ≤ 1,000 active and 1,000 noise windows per recording, spread over 2 s and sorted by start. A stream is therefore subsampled in time, about one cached window per 2 ms.
- **Per-window compute** (counted): CNN 1.87 M MACs; WPD `dmey` (filter length 62), level 6, 4,000 samples per I/Q channel.

## Goals / Non-Goals

**Goals:**
- One stream runner in `src/rfml_uav/adapt/stream.py` that runs every method through the same chunked, prequential loop, so differences come only from the method.
- Every mechanism label-free and cost-reported. The refresh is safe on single-drone buffers by construction (the guard).

**Non-Goals:** see the proposal's "Out of scope" list. In particular: no head-only *method* (one ablation row only), no EMA teacher or class balancing, no cross-condition streams.

## Decisions

### D1. Chunked prequential runner: `run_stream(model, X, active, method, hp, calib, seed) -> dict`
- `X` is the stream's centred features (all windows, time-ordered), and `active` marks the active windows. Only active windows are classified; noise windows exist only for centring (computed once per stream, outside the runner, shared by all methods).
- Active windows are processed in chunks of M = 250. For each chunk:
  1. predict (logits, energy) with the current state;
  2. gate;
  3. update the vote;
  4. append gated windows to the buffer;
  5. check the trigger;
  6. maybe adapt.
- So every prediction precedes any update it feeds.
- Returns per-active-window `pred`, `vote`, `energy` and `known`, plus `events`, `full_refreshes` and `restricted_refreshes`, `fwd`/`bwd` totals and the per-event parameter/statistic counts.

**Alternative:** window-by-window updates. Rejected: 250× slower, and real nodes batch anyway.

### D2. Vote: sliding majority over the last W per-window predictions
Kept in a ring buffer. Ties go to the most recent of the tied classes.

### D3. Gate: anchored quantile threshold
- The threshold is the (1 − q) quantile of the calibration energies. After each refresh event, it is re-anchored on the energies of **all** active windows (accepted or not) of the next R chunks, at the same time as the trigger reference (D4). Until then the previous threshold stays in force.
- Methods that update every chunk (continuous, TENT, naive PL) have no refresh events and keep the calibration threshold.
- **Calibration:** the source model's energies on its own CLEAN validation rows (centred, session estimator). Labels aren't used, only the energies of known-good data.

**Rejected (2026-10-26): a sliding quantile over the last N *accepted* windows** (the original D3). Accepted energies lie below the threshold, so it ratchets down: on a stationary toy stream (q = 0.1, N = 2,000), rejection went 0.18 → 0.70 → 0.84 over 200 chunks. **Alternative for later:** a sliding quantile over all windows. It is stable on stationary streams, but it absorbs any unknown block longer than N, which is how E5-lite presents the held-out class.

### D4. Trigger: chunk-mean energy against a reference
- Before the first refresh, the reference is (μ, σ) of chunk means over the calibration set, chunked the same way.
- After a refresh, it is re-anchored on the mean and std of the next R = 4 chunks' means.
- The trigger fires when |chunk mean − μ| > θσ. Cooldown: C = 8 chunks.
- **Alternative:** vote-disagreement trigger. Kept only as a dev-tuning candidate signal. Energy is the primary signal because it needs no history of labels.

### D5. Guarded refresh
- The buffer holds the last B gated windows.
- **Full refresh** (if the buffer's vote decisions cover ≥ D classes): `adabn(model, buffer)` replaces every BN layer's statistics.
- **Restricted refresh** (otherwise): compute `adabn` on the buffer, then for bn1–bn3 (the conv layers) set `(1 − λ)·source + λ·buffer`. bn4–bn6 keep their current statistics.
- Spike evidence (seed 0): a single-recording full refresh collapses to 0.14–0.16; conv + blend 0.5 is safe (0.55 / 0.61 / 0.37 on BOTH / awgn 3 / tilt 4).

### D6. Comparison methods, all in the same runner
- **Continuous AdaBN:** BN in train mode under `no_grad` on every chunk, with momentum m. Forward-only, always on.
- **Periodic:** a guarded refresh every P chunks, without the trigger. It isolates what the trigger saves.
- **Online TENT:** one entropy step per chunk on BN affine parameters, Adam.
- **Online naive PL:** one CE step per chunk on windows with max-softmax ≥ τ, all parameters.
- **Vote-PL (3b):** on a trigger, one epoch over the buffer, Adam, all parameters, on block-majority labels with agreement ≥ ρ, then a guarded refresh.
- **Head-only ablation:** the same, but fc4 only.

### D7. Streams
- **Receiver condition:** the prefix is the model's CLEAN test recordings (clean features). The target segment is the target's E3 test-half recordings. Recordings are shuffled within each part by `default_rng([seed, fold, TARGET_ID])`.
- **Single-drone check:** for each shift-B target test-half recording, the clean version of that recording followed by its impaired version. A change point within one drone.
- **Centring:** `stream_normalise` runs over the whole chained stream (state carried across recordings, cold start at the stream's start). For impaired segments, the noise windows are the impaired noise windows.

### D8. Dev tuning, staged to keep the grid small (seed-0 models, dev streams)
Each stage fixes its winner before the next:

| stage | parameters | selection rule |
|---|---|---|
| 1 | W ∈ {10, 25, 50} | vote F1 of source-only |
| 2 | q ∈ {0.1, 0.2} | known-window rejection ≤ q + 0.05 at equal vote F1 |
| 3 | θ ∈ {2, 3, 4}, B ∈ {1000, 3000}, D ∈ {3, 5}, λ ∈ {0.3, 0.5} | triggered refresh: dev vote F1, tie-break fewer events |
| 4 | momentum m ∈ {0.01, 0.1}; P ∈ {20, 60}; TENT lr ∈ {1e-4, 1e-3}; PL lr ∈ {1e-5, 1e-4}; ρ ∈ {0.6, 0.8} | per method |

- Tie-break everywhere: within 0.1 pp → cheaper.
- Written to `phase5/stream_hparams.json` before any target stream runs.

### D9. E5-lite: leave-one-class-out (LOCO)
- For each class c and fold f (seed 0): `undersample` the centred CLEAN active rows of the other 6 classes, `make_splits(..., "recording", seed=0)[f]`, then `train_best_val`. That gives 35 models in `checkpoints/paper3/loco/`.
- **Test streams:**
  - CLEAN: the model's known test recordings plus all CLEAN recordings of c, chained;
  - BOTH: the BOTH recordings of the known classes' test-fold drones plus all BOTH recordings of c (seen modes only).
- Run with source-only and the triggered refresh, gate on and off.
- **Metrics:**
  - energy AUROC, known vs unknown;
  - unknown false-accept rate and known rejection rate at the gate;
  - known-class vote macro-F1, over windows the gate accepted (gate on) or all windows (gate off).

### D10. Cost ledger
- **WPD MACs:** computed exactly by running `pywt.WaveletPacket` once and summing (coefficient length × filter length × 2 filters) over all decomposition nodes, plus 2·len per final node for RMS/STD, times 2 channels.
- **CNN MACs:** from forward hooks (1.87 M).
- **Centring:** 256 subtractions per window plus a 256-d EMA update per noise window.
- **Per-method run counts:** forward/backward passes, events and refreshes from D1.
- The ledger is labelled hardware-independent; E9 multiplies it by measured J/MAC and J/pass.

### D11. Outputs (`results/paper3/phase5/`)
- `streams/{target}_s{seed}_f{fold}.npz`: per method `pred`/`vote` (int8), `energy` (float16), `known` (bool) and `cost` (json); plus `y`, `rows` and `change_point`.
- Tables: `stream_e3.csv`, `stream_e3_harm.csv`, `trigger.csv`, `single_drone.csv`, `e5_lite.csv`, `cost_ledger.csv`, and a figure `ladder.png` (vote F1 vs backward passes per window, per target group).

## Risks / Trade-offs

- **[Cached windows are subsampled in time]:** a vote over W cached windows spans about W × 2 ms, more than W consecutive real windows. → Report W in milliseconds too. Real-time windows would make votes cheaper, not dearer.
- **[Energy trigger may also fire on drone changes inside a condition]:** chained recordings switch drones, and the energy level differs by class. → The cooldown and the guard limit the damage; the false-trigger count in the clean prefix measures it.
- **[Calibration on validation rows uses the source split]** → No labels are used, only energies. This is equivalent to a deployment-time calibration on the source's own known-good data.
- **[LOCO models are seed 0 only]** (35 models): no seed variance for E5-lite. → The CI is over folds and classes; stated as E5-*lite*.
- **[The guard's D and λ come from the dev tuning on mild cells]** → The single-drone check reports the guard's behaviour directly.
- **[The spike numbers are not results]** → All reported numbers come from this change's runs.

## Migration Plan

Additive: a new module, notebooks 7 and 8, `phase5/` and `checkpoints/paper3/loco/`. Rollback = delete them.

## Open Questions

- The method name for the paper (OG-TCST or a rename). This doesn't affect the work.
