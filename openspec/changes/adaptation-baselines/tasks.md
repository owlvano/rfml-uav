# Tasks

## 1. Baseline methods (`rfml_uav.adapt`)

- [x] 1.1 Add `src/rfml_uav/adapt/__init__.py` (empty) and `baselines.py` with `adabn(model, X, batch_size=1024, seed=0)` per design D1–D2. It returns `(adapted_copy, info)` with `fwd_per_window=1, bwd_per_window=0, n_updated=0`, and restores the BN momentum afterwards. Verify with `python -c` that the source `state_dict` is bit-identical after a call.
- [x] 1.2 Add `tent(model, X, lr, epochs, batch_size=256, seed=0)` per D3 (BN affine only, entropy loss, Adam, then the D2 statistics pass), with `info` per the spec (E + 1 forward, E backward, `n_updated` = BN affine count). Verify `n_updated == 2016` on the DroneDetect CNN.
- [x] 1.3 Add `pseudo_label(model, X, tau, lr, epochs, batch_size=256, seed=0)` per D4: labels fixed from the source pass, the τ filter, the "fewer than one batch" early return, and `info["selected"]` / `["selected_per_class"]`. Verify a τ above every max-softmax returns a model bit-identical to the source with `selected == 0`.
- [x] 1.4 Add an `assert`-based `__main__` self-check on CPU. Set-up: train a tiny BN MLP (`Linear → BatchNorm1d → ReLU → Linear`, first layer BN on the input) on 3 Gaussian blobs, then shift the inputs (scale 2, offset 1). Checks:
  - AdaBN: shifted accuracy rises above source-only; the first BN running mean equals the shifted-set mean within 1e-4; trainable parameters are unchanged;
  - TENT: only BN affine parameters change; the mean entropy on the adapt set drops;
  - PL: the pseudo-labels equal the source argmax on the selected windows, and the selected count matches the τ filter;
  - every method: the source is untouched, and a repeat run with the same seed gives identical predictions.

  Verify `python -m rfml_uav.adapt.baselines` prints its OK line.
- [x] 1.5 Add `adapt/baselines.py` to the package section of `CLAUDE.md` (3 functions, `(model, X, **hp) -> (copy, info)`, label-free, cost info). Verify the line is present.

## 2. Splits and targets (notebook 5, part 1)

- [x] 2.1 Create `notebooks/paper3-label-free-adaptation/5_Adaptation_Baselines.ipynb` with:
  - a setup cell with deterministic cuDNN flags;
  - `TARGETS = {BLUE, WIFI, BOTH, awgn_2, awgn_3, gain_3, adc_3, tilt_4}` and `DEV = {awgn_1, gain_2, adc_2, tilt_3}`, each with a stable `TARGET_ID`;
  - loaders for `features.npz`, the Phase 1 runs and checkpoints, and the Phase 2 impaired files.

  Verify the setup cells run.
- [x] 2.2 Implement `split(target, seed, fold)` per design D6 and save `phase3/splits/*.npz` for all targets and dev cells × 25 models. Asserts:
  - the adapt and test recording ids are disjoint, and every class is in both halves;
  - shift-B recordings are a subset of the model's CLEAN test recordings and disjoint from its train/val rows (rebuilt with Phase 1's `source_rows` + `make_splits`, or from `test_idx` recording ids);
  - no Phantom FY row is in a shift-A split.

  Verify all asserts pass and print per-target median adapt/test sizes.

## 3. Dev-shift selection (notebook 5, part 2)

- [x] 3.1 Run the D7 grid on the dev cells with the 5 seed-0 models. Save `phase3/dev_grid.csv` (method, hyperparameters, cell, fold, source F1, adapted F1, gain). Verify it has 4 × 5 × 19 rows.
- [x] 3.2 Select per method by mean dev gain with the D7 tie-break; write `phase3/selected_hparams.json`. Verify the file exists, and that the target loop (4.1) reads hyperparameters only from it (no grid variables in scope there).

## 4. Target evaluation and E3 report (notebook 5, part 3)

- [x] 4.1 For every target × 25 models, adapt with source-only, AdaBN, TENT and PL at their selected settings on the adapt half, and score the test half. Save `phase3/runs/{target}_s{seed}_f{fold}.npz` per design D8, plus `clean_logits` for shift B. Verify:
  - source-only macro-F1 on shift-A test halves is within the Phase 1 per-subset range;
  - re-running one (target, model) gives identical predictions.
- [x] 4.2 Sanity row: AdaBN adapted on the *clean* version of one shift-B adapt half, scored on the clean test half, for all 25 models. Report the mean gain with CI in the notebook. Verify the gain is within ±1 pp. If it isn't, flag statistics noise in the README limitations.
- [x] 4.3 Build `phase3/e3_table.csv`: per (target, method), macro-F1 and accuracy mean ± std, gain vs source-only with `paired_test` CI and p, and recovered fraction (D8, with the count of excluded runs). Mark the best baseline per target. Also save `phase3/e3_per_class.csv`. Verify 8 × 4 rows and exactly one best-baseline flag per target.
- [x] 4.4 Build `phase3/e3_auroc.csv` (energy and max-softmax AUROC per target × method) and `phase3/e3_cost.csv` (forward and backward passes per window, `n_updated`, median wall time; labelled as a proxy). Verify every row has both AUROCs and the cost table has 4 rows.
- [x] 4.5 Figure `phase3/e3_recovery.png`: recovered fraction per target (shift A | shift B) × method, with 95% CI. Verify the file is written and the colours are fixed per method.

## 5. Write-up

- [x] 5.1 Add a "Phase 3 — Adaptation baselines (E3)" section to `notebooks/paper3-label-free-adaptation/README.md`:
  - protocol (regime, splits, dev selection and the chosen hyperparameters);
  - the E3 table;
  - recovered fractions;
  - AUROC after adaptation;
  - the cost table;
  - limitations (small shift-B adapt sets, mild dev cells, offline regime).

  Verify every number matches the CSVs.
- [x] 5.2 Add a dated "Phase 3 outcome" entry to `NOTES.md`:
  - does AdaBN recover AWGN most, as the shortcut finding predicts?
  - which baseline is the bar per target;
  - what OG-TCST must beat, and at what cost;
  - an updated "Next session" list.

  Verify the entry is present.

## Workflow follow-up

- Review the results, then archive with `/opsx:archive` (sync specs), commit, and merge to `main`.
