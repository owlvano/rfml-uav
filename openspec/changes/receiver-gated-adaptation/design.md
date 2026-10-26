# Design

## Context

See proposal.md (Why, including the 2026-10-26 discovery). What exists:
- **Runner.** `rfml_uav.adapt.stream.run_stream`: a chunked prequential loop with the gate, vote, trigger and refresh, and the `continuous` method (BN momentum m, train-mode no-grad pass on the chunk's accepted windows).
- **Centring.** `rfml_uav.data.normalise.stream_normalise`: an EMA over noise windows with warm-up k, returning `(Xn, valid[, mu])`.
- **Streams.** `7_Stream_Adaptation.ipynb` builds the Phase 5 streams (`build_stream`) and stores per (target, model) files with `rows`, `y`, `change_point` and the per-method votes. The frozen hyperparameters are in `phase5/stream_hparams.json`.

## Goals / Non-Goals

**Goals:**
- Three forward-only methods and one oracle diagnostic, on the same runner and the same streams as Phase 5, so the Phase 5 rows can be compared directly.
- A dev-only, harm-aware tuning, frozen before targets.

**Non-Goals:**
- Changing any Phase 5 method or result.
- Gradient methods.
- New stream protocols.
- Single-drone streams.
- E5-lite.

## Decisions

### D1. Drift from the stream estimator
- `stream_normalise(..., return_drift=True)` adds an array `drift`, aligned with the windows: ‖μ_i − μ_0‖₁ / ‖μ_0‖₁, where μ_i is the state used for window i and μ_0 the warm-up mean. It is NaN before warm-up completes.
- The normalised output is unchanged.
- **Why relative L1 to the warm-up state:** it is dimensionless, costs 256 operations per window, and it is the measured signal from the discovery.
- **Alternatives rejected:**
  - a recent-past jump detector: it fires on every receiver-group switch;
  - the spread of centred noise windows: it doesn't separate the targets.

### D2. Runner input
- `run_stream(model, X, active, method, hp, calib, seed, drift=None)`. `drift` is per window (all windows, like `X`); the runner keeps the active entries.
- It is a label-free signal computed outside the runner, from the same centring pass that produces `X`.

### D3. Receiver gate with hysteresis
- After a chunk is predicted, the gate reads the drift of the chunk's last active window (NaN counts as 0).
- OFF → ON if drift > φ_on. ON → OFF if drift < h·φ_on, with h = 0.5 fixed.
- Updates in this chunk follow the new state, after the chunk's predictions. So a chunk is always predicted before its own update (prequential).
- An ON → OFF transition restores the deep-copied source BN running statistics.
- No drift (`None`) → the gate never opens.

### D4. Drone-balanced buffer
- A dict vote-class → list of active-window indices (accepted by the open-set gate), FIFO-capped at Bc.
- An update is allowed when at least D classes hold ≥ Bc/2 windows. The update batch is the union of those classes' windows.
- **The update:** BN in train mode with momentum m, a no-grad forward pass on the batch (in slices of ≤ 4,096), then eval.
- **Cost:** forward passes += batch size.
- **Gated methods clear the buffer on every OFF → ON transition,** so the ON statistics come from post-change windows only. The always-on method never clears.

### D5. Methods
| method | gate | update per chunk |
|---|---|---|
| `balanced` | always on | balanced batch, if the D-class condition holds |
| `gated_continuous` | receiver gate | Phase 5 continuous (the chunk's accepted windows), while ON |
| `gated_balanced` | receiver gate | balanced batch while ON, if the condition holds |
| oracle (diagnostic) | drift = 0 before the change point, 1 after | `gated_continuous` with φ_on = 0.5 |

All of them report `on_fraction` (per active window) and `updates`, besides the Phase 5 cost fields.

### D6. Streams and reuse
- `9_Gated_Adaptation.ipynb` repeats notebook 7's setup and `build_stream` (data-protocol code, not model code), adding `return_drift=True`.
- Per (target, model) it asserts that the rows equal `phase5/streams/*` rows and that source-only votes equal the stored Phase 5 votes.
- The Phase 5 rows (source, continuous, TENT, triggered) are read from the Phase 5 stream files, not re-run.

### D7. Harm-aware staged dev tuning (seed-0 models, 4 dev cells × 5 folds)
- **Objective:** J = mean target vote F1 − mean max(0, source prefix vote F1 − method prefix vote F1). Tie-break: within 0.1 pp of the best J → fewer forward passes per window.
- W, q, M, R and C stay as in `phase5/stream_hparams.json`.
- **Stages:**
  1. `balanced`: Bc ∈ {200, 500} × D ∈ {3, 5} × m ∈ {0.01, 0.1}.
  2. `gated_continuous`: φ_on ∈ {0.3, 0.5, 0.7} × m ∈ {0.01, 0.1}.
  3. `gated_balanced`: φ_on ∈ {0.3, 0.5, 0.7}, with Bc / D / m from stage 1.
- Saved to `phase6/hparams.json` before any target stream runs.
- **Disclosure:** the 2026-10-26 spike ran five `balanced`-like settings on these dev cells. That is dev data only; stated in the write-up.

### D8. Outputs (`results/paper3/phase6/`)
- `streams/{target}_s{seed}_f{fold}.npz`: per method `vote` / `pred` (int8), `known` (bool), `cost` (json, including `on_fraction` and the per-chunk ON flags as an `on` int8 array), plus `drift` (float16, active windows), `rows` and `change_point`.
- `dev_tuning.csv`, `hparams.json`.
- `gated_e3.csv`: target and prefix vote F1, and paired differences vs source-only and vs Phase 5 continuous, for the Phase 6 methods, the oracle and the Phase 5 reference rows.
- `gate.csv`: prefix / target ON fraction, detection rate, median delay.
- `cost.csv`.

## Risks / Trade-offs

- **[Post hoc]:** designed after seeing the Phase 5 targets → disclosed, reported beside the frozen rows, and tuned on dev cells only.
- **[The gate opens on receiver-group switches in the clean prefix]** (prefix drift up to ≈ 0.49) → φ_on ∈ {0.5, 0.7} sits above most of the prefix, and the prefix ON fraction and prefix harm are reported. A low φ_on is penalised by the harm-aware objective.
- **[AWGN and ADC don't move the floor]** → gated methods can't help awgn 3. That is expected; the oracle row shows how much a perfect gate would add.
- **[Balanced buffers mix pre- and post-change windows]** (always-on variant) → this dilutes adaptation (dev gain 2: +6.6 vs +26.9 pp). Gated balanced clears the buffer on ON.
- **[A gated-balanced buffer needs D drones after the change before it can update]** → it updates late. The delay is reported. A smaller Bc fills faster.
- **[The drift reference is the stream's first warm-up]** → a deployment starting under impairment would treat the impaired floor as normal. Stated as a limitation.

## Migration Plan

Additive: an optional output in `normalise.py`, an optional argument and three methods in `stream.py` (Phase 5 behaviour unchanged when `drift` is omitted), notebook 9 and `phase6/`. Rollback = revert the commit and delete `phase6/`.
