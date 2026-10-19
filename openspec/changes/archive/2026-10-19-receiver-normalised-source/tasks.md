# Tasks

## 1. Normalisation module and shared undersampling

- [x] 1.1 Rework `src/rfml_uav/data/normalise.py` to centring (D1: `noise_mean`, `centre`, `session_normalise`, `stream_normalise`; β = 0.01, k = 50 as defaults). Verify with `python -c` that `session_normalise` on a 2-recording toy input returns float32 of the input shape.
- [x] 1.2 Update the `assert`-based `__main__` self-check to the centring spec:
  - active rows don't change the mean;
  - a per-feature additive change `f + b` applied to a whole session leaves session-normalised output unchanged within 1e-4;
  - session-normalised noise has mean 0 within 1e-4;
  - stream state is unchanged across a run of active windows;
  - a stationary noise stream of 20/β windows converges within 5% of the session mean;
  - no look-ahead: changing a later window leaves earlier outputs unchanged;
  - `valid` is False until k noise windows have arrived.

  Verify `python -m rfml_uav.data.normalise` prints its OK line.
- [x] 1.3 Add `undersample(idx, y, seed)` to `src/rfml_uav/eval/splits.py` (D4, verbatim Phase 1 logic) and extend its `__main__` self-check with a determinism assert. Verify `python -m rfml_uav.eval.splits` passes.
- [x] 1.4 Add `data/normalise.py` (centring) and `undersample` to the package section of `CLAUDE.md`. Verify the lines are present.

## 2. Impaired noise windows (notebook 4)

- [x] 2.1 Add a "DroneDetect noise windows" section to `4_Receiver_Impairment_Shift.ipynb`: CLEAN noise rows grouped by recording, then `build("dronedetect_noise", …, CELLS + [("tilt", 4)])` with the same `dd_part` / `apply_cell` (D3). Checks:
  - each file's `row` equals the CLEAN noise rows;
  - severity-0 features of 200 random noise rows equal `features.npz` within 1e-5;
  - the existing `dronedetect/*.npz` files keep their modification times.

  Verify the notebook re-runs end to end with all earlier asserts passing.

## 3. Normalised source models (notebook 6, part 1)

- [x] 3.0 Move the std-normalised run aside (D7): `results/paper3/phase4/` → `phase4_std/`, `checkpoints/paper3/noisenorm/` → `noisenorm_std/`. Verify both new directories hold 50 run files and 25 checkpoints, and the old paths no longer exist.

- [x] 3.1 Create `notebooks/paper3-label-free-adaptation/6_Receiver_Normalised_Source.ipynb`:
  - setup;
  - session-normalise all rows;
  - for each seed, assert `undersample` + `make_splits` reproduce the Phase 1 `test_idx` of all 5 folds.

  Verify the assert passes for all 25 (seed, fold).
- [x] 3.2 Train 25 linear-tail models on session-normalised CLEAN active rows with `train_best_val` (Phase 1 hyperparameters). Save checkpoints to `checkpoints/paper3/noisenorm/` and run files to `phase4/runs/`, with session `test`/`tgt` logits plus `stream_logits` and `stream_valid` for all rows (D5). Verify:
  - 25 checkpoints exist;
  - no file under `checkpoints/paper3/phase1/` changed;
  - the run files have the Phase 1 keys plus the stream keys.
- [x] 3.3 Shortcut re-test: the same CNN and protocol trained on session-normalised CLEAN noise rows, reporting 7-class and receiver-group accuracy against chance next to Phase 1 (0.43 / 0.98). Save `phase4/shortcut_test.csv`. Verify the verdict line (above / below 2 × chance) is printed.

## 4. E1 / E2 re-scoring (notebook 6, part 2)

- [x] 4.1 Build `phase4/e1_table.csv` and `phase4/e1_by_mode.csv`: per subset (and mode), mean ± std for Phase 1, normalised-session and normalised-stream, plus normalised − Phase 1 differences with `paired_test`, and the warm-up exclusion fraction for the stream rows. Phantom FY is excluded, as in Phase 1. Verify 4 subsets × 3 source rows, with differences for both normalised rows.
- [x] 4.2 Score the 25 normalised models on the E2 cells: impaired active rows normalised with the impaired noise rows (session and stream). Phase 1 numbers come from `phase2/runs`. Build `phase4/e2_table.csv` in the same paired format. Verify 13 cells × 3 source rows.
- [x] 4.2b Ablation and realistic AGC (D7–D8): add the std-normalised rows from `phase4_std/` to the E1/E2 tables; build the rare-step AGC condition (5% of rows, gain severities 1–3) and score both sources, session estimator, in `phase4/agc_rare.csv` with paired differences. Verify the E1/E2 tables carry the std columns and `agc_rare.csv` has 3 severities × paired rows.
- [x] 4.3 β sensitivity: stream rows with β ∈ {0.003, 0.03} on E1 BOTH and E2 awgn 3 only, in `phase4/beta_sensitivity.csv`. Verify 2 conditions × 3 β values.

## 5. E3 on the normalised source (notebook 5) and adoption

- [x] 5.1 Add `SOURCE = os.getenv("SOURCE", "phase1")` to notebook 5 per D6 (checkpoint directory, session-normalised `features`, output directory `phase3/noisenorm/`). Verify:
  - with the default, re-running the table cells reproduces the archived `phase3/e3_table.csv` exactly;
  - with `SOURCE="noisenorm"`, the dev-grid cells are skipped and `selected_hparams.json` is read from `phase3/`.
- [x] 5.2 Run notebook 5 with `SOURCE=noisenorm jupyter nbconvert --execute --to notebook --output 5_Adaptation_Baselines_noisenorm.ipynb` (a separate executed copy, so the committed notebook keeps its `phase1` outputs). Verify `phase3/noisenorm/e3_table.csv` has 8 × 4 rows.
- [x] 5.3 Adoption cell in notebook 6: per-run mean source-only macro-F1 over the 8 E3 targets, normalised vs Phase 1, via `paired_test`; apply the spec rule; write `phase4/adoption.json` (adopted source, pooled diff, CI, p, CLEAN diff). Verify the file exists and names exactly one source.

- [x] 5.4 (added during apply, author decision 2026-10-19) Amended adoption rule: add the supplementary `agc_3` target to notebook 5 (rare 12 dB steps on 5% of rows, the D8 mask; separate `e3_agc.csv`, 8-target tables untouched) for both sources. Re-apply adoption in notebook 6 with `gain_3` → `agc_3`, reporting the original and amended results in `adoption.json`. Verify:
  - the default notebook-5 run keeps the four `e3_*.csv` md5s;
  - `e3_agc.csv` exists for both sources;
  - `adoption.json` holds both rule results.

## 6. Write-up

- [x] 6.1 Add "Phase 4 — Receiver-normalised source (3.2)" to `notebooks/paper3-label-free-adaptation/README.md`:
  - method (session/stream, β, k);
  - shortcut re-test;
  - E1 / E2 / E3 side-by-side tables;
  - adoption decision;
  - limitations: noise labels offline, cold-start streams, CLEAN cost.

  Verify every number matches the CSVs.
- [x] 6.2 Add a dated "Phase 4 outcome" to `NOTES.md`: whether 3.2 holds, which source OG-TCST uses, and what that implies for the gate and TCST design. Update "Next session". Verify the entry is present.

## Workflow follow-up

- Review the results, then archive with `/opsx:archive` (sync specs, including the modified `impairment-shift-evaluation`), commit, and merge to `main`.
