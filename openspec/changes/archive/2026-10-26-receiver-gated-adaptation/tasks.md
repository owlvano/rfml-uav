# Tasks

## 1. Drift signal and runner

- [x] 1.1 Add `return_drift` to `rfml_uav.data.normalise.stream_normalise` (D1), and extend its `__main__` self-check:
  - stationary noise → drift near 0;
  - every feature scaled by a after warm-up → drift → |a − 1|;
  - outputs identical with and without drift.

  Verify `python -m rfml_uav.data.normalise` prints its OK line.
- [x] 1.2 Add the `drift` argument, the receiver gate (D3), the balanced buffer (D4) and the methods `balanced`, `gated_continuous` and `gated_balanced` (D5) to `rfml_uav.adapt.stream`, with `on_fraction`, `updates` and per-chunk `on` flags in `cost`. Phase 5 methods must be unchanged when `drift` is omitted. Verify the Phase 5 self-check still passes.
- [x] 1.3 Extend the `__main__` self-check:
  - stable floor → gated equals source and 0 updates;
  - hysteresis keeps ON between h·φ_on and φ_on;
  - OFF restores the source BN statistics with parameters unchanged;
  - single-class stretch → no balanced update;
  - per-class cap Bc;
  - forward-only (0 backward);
  - never-ON cost = 1 forward per window;
  - no label argument (signature with `drift`).

  Verify `python -m rfml_uav.adapt.stream` prints its OK line.
- [x] 1.4 Update the `CLAUDE.md` package notes (drift output, gated/balanced methods). Verify the lines are present.

## 2. Streams and dev tuning (notebook 9, part 1)

- [x] 2.1 Create `notebooks/paper3-label-free-adaptation/9_Gated_Adaptation.ipynb` with notebook 7's setup and `build_stream` plus drift (D6). Assert, for 3 sample (target, model) pairs, that:
  - the rows equal the Phase 5 stream file's rows;
  - source-only votes equal the stored Phase 5 votes.
- [x] 2.2 Run the D7 staged harm-aware dev tuning and save `phase6/dev_tuning.csv` and `phase6/hparams.json`. Verify:
  - each stage's winner is recorded;
  - the json exists before any target stream file;
  - the target loop reads only the json.

## 3. Targets and tables (notebook 9, part 2)

- [x] 3.1 Run `balanced`, `gated_continuous`, `gated_balanced` and the oracle on 9 targets × 25 models, storing `phase6/streams/*.npz` (D8; skip existing). Verify 225 files with all 4 runs, and that a re-run of one (target, model) reproduces the votes.
- [x] 3.2 Build `phase6/gated_e3.csv`: target and prefix vote F1 (mean ± std) and paired differences vs source-only and vs Phase 5 continuous, with the Phase 5 source / continuous / TENT / triggered rows read from `phase5/streams`. Verify 9 × 8 rows.
- [x] 3.3 Build `phase6/gate.csv` (prefix and target ON fraction, detection rate, median delay per target and gated method) and `phase6/cost.csv` (forward/backward passes per window, updates per stream, per method). Verify every target row and 4 cost rows.

## 4. Write-up

- [x] 4.1 Add "Phase 6 — Receiver-gated, drone-balanced statistics (post hoc)" to `notebooks/paper3-label-free-adaptation/README.md`:
  - the post-hoc disclosure (including the dev spike);
  - the methods;
  - the selected hyperparameters;
  - the comparison table beside the Phase 5 rows;
  - the gate diagnostics and oracle;
  - cost;
  - limitations.

  Verify every number matches the CSVs.
- [x] 4.2 Add a dated "Phase 6 outcome" to `NOTES.md` (does gating/balancing resolve the Phase 5 trade-off; the RQ3 answer; deferred alternatives) and update "Next session". Verify the entry is present.

## Workflow follow-up

- Review the results, then archive with `/opsx:archive` (sync specs), commit, merge to `main`, push.
