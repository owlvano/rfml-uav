# Tasks

## 1. Stream runner (`rfml_uav.adapt.stream`)

- [x] 1.1 Add `src/rfml_uav/adapt/stream.py` with the D1 chunked prequential loop: `run_stream(model, X, active, method, hp, calib, seed)` returning per-active-window `pred`, `vote`, `energy` and `known`, plus the cost/event fields. Include the D2 sliding vote and the D3 anchored-quantile gate. Method `"source"` only for now. Verify on a synthetic stream that source-only reports 1 forward pass per window and 0 events.
- [x] 1.2 Add the D4 trigger (calibration reference, re-anchoring, cooldown) and the D5 guarded refresh (full if the buffer's vote labels cover ≥ D classes, otherwise conv-only λ-blend) as method `"triggered"`. Verify on a synthetic stream with a known mean shift that it fires after the shift, not before, and respects the cooldown.
- [x] 1.3 Add the D6 methods: `"continuous"`, `"periodic"`, `"tent"`, `"pl"`, `"vote_pl"` and `"vote_pl_head"`. Verify each runs on a synthetic stream and reports the D1 cost fields (backward > 0 only for the gradient methods).
- [x] 1.4 Add an `assert`-based `__main__` self-check for the spec scenarios. Toy setup: a BN MLP on 3 Gaussian blobs, with a stream that shifts mid-way. Checks:
  - prediction precedes update (each window's logits equal the pre-update model's);
  - no label argument;
  - vote majority and the short-history rule;
  - gated windows never enter a buffer;
  - the gate rejection stays near q on a stationary stream;
  - a stable stream gives zero events;
  - cooldown;
  - a single-class buffer gives a restricted refresh that changes only the conv-style first BN, while a diverse buffer gives a full refresh with trainable parameters unchanged;
  - vote-PL labels are block majorities with agreement ≥ ρ;
  - source-only cost.

  Verify `python -m rfml_uav.adapt.stream` prints its OK line.
- [x] 1.5 Add `adapt/stream.py` to the package section of `CLAUDE.md`. Verify the line is present.

## 2. Streams, calibration and dev tuning (notebook 7, part 1)

- [x] 2.1 Create `notebooks/paper3-label-free-adaptation/7_Stream_Adaptation.ipynb`:
  - setup;
  - loading the centred features (session for calibration, impaired sets + `agc_3` composed with the D8 mask from notebooks 5/6);
  - a `build_stream(target, seed, fold)` per D7 that returns the stream-centred `X`, `active`, `y`, `rows` and `change_point`;
  - calibration energies from each model's CLEAN validation rows.

  Asserts:
  - stream recordings ∩ the model's train/val recordings = ∅;
  - the change point is the first target window;
  - windows are time-ordered within each recording.

  Verify the asserts pass for 3 sample (target, model) pairs.
- [x] 2.2 Run the D8 staged dev tuning on dev streams (awgn_1, gain_2, adc_2, tilt_3; seed-0 models). Save `phase5/dev_tuning.csv` and `phase5/stream_hparams.json`. Verify:
  - each stage's winner is recorded;
  - the json exists before any target stream file;
  - the target loop reads hyperparameters only from the json.

## 3. Streaming E3 (notebook 7, part 2)

- [x] 3.1 Run all D6 methods plus source-only on 9 targets × 25 models, storing `phase5/streams/*.npz` per D11 (skip existing). Verify 225 stream files, each with all 8 methods, and that a re-run of one (target, model) reproduces `vote` exactly.
- [x] 3.2 Build `phase5/stream_e3.csv`: target-segment vote and per-window macro-F1 (mean ± std), best method per target, every method's paired difference to the best and to source-only (`paired_test`). Also build `phase5/stream_e3_harm.csv` with the prefix vote F1 differences. Verify 9 × 8 rows and exactly one best method per target.
- [x] 3.3 Build `phase5/trigger.csv`: false triggers in the prefix and detection delay (mean and median active windows, and ms at the cache spacing) per target, for the triggered methods. Verify every target row is present.
- [x] 3.4 Single-drone check per D7 on the shift-B targets: build `phase5/single_drone.csv` with triggered-method F1 vs source-only and the restricted/full refresh shares. Verify every shift-B target row is present.
- [x] 3.5 Build `phase5/cost_ledger.csv` per D10 (exact WPD MACs from coefficient lengths, CNN MACs from hooks, the centring cost, and per-method passes, events and refreshes from the stream files), plus the figure `phase5/ladder.png`. Verify 8 method rows, with the WPD and CNN MACs as separate fields.

## 4. E5-lite (notebook 8)

- [x] 4.1 Create `notebooks/paper3-label-free-adaptation/8_Open_Set_Gate.ipynb`: train the 35 LOCO models per D9 (seed 0; checkpoints in `checkpoints/paper3/loco/`, skip existing). Verify 35 checkpoints, and that no training row belongs to the held-out class.
- [x] 4.2 Build the CLEAN and BOTH LOCO streams and run source-only and triggered, gate on and off. Write `phase5/e5_lite.csv` per held-out class and condition (energy AUROC, unknown false-accept rate, known rejection rate, known vote macro-F1 gate on/off) plus a mean row. Verify 7 × 2 rows + mean.

## 5. Write-up

- [x] 5.1 Add "Phase 5 — Stream adaptation (RQ3) and open-set gate (RQ4)" to `notebooks/paper3-label-free-adaptation/README.md`:
  - the ladder and its protocol;
  - selected hyperparameters;
  - the streaming E3 table;
  - harm, trigger and single-drone results;
  - E5-lite;
  - the cost ledger;
  - limitations (cached-window spacing, seed-0 LOCO, energy trigger and drone changes, offline noise labels).

  Verify every number matches the CSVs.
- [x] 5.2 Add a dated "Phase 5 outcome" to `NOTES.md`:
  - which rungs earn their cost;
  - whether 3b is worth its backward passes;
  - the RQ3/RQ4 answers;
  - the alternatives deferred to later iterations (from the proposal);
  - an updated "Next session" (E9, ablations, writing).

  Verify the entry is present.

## Workflow follow-up

- Review the results, then archive with `/opsx:archive` (sync specs), commit, merge to `main`, push.
