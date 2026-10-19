# Tasks

## 1. Impairment simulator

- [x] 1.1 Add `src/rfml_uav/data/impairments.py` with `SEVERITY` (the spec table, index 0 = identity, index 4 = escalation), `KIND_ID` and `impair(x, kind, severity, rng, step=True)` per design D1–D2. Unknown kind or severity outside 0–4 raises `ValueError`. Verify `python -c` on an unknown kind and on severity 5 raises.
- [x] 1.2 Add an `assert`-based `__main__` self-check on a seeded white-noise window, real and complex, for every severity 1–4:
  - output dtype kind and length match the input;
  - severity 0 returns the input unchanged;
  - measured AWGN SNR is within 0.5 dB of the table;
  - tilt top-10% vs bottom-10% band power is within 1 dB of 0.9 × the table slope;
  - gain step RMS ratio is within 0.5 dB;
  - ADC output is within ±k·RMS and has ≤ 2^bits distinct values per component;
  - the same `default_rng([0, KIND_ID[k], s, row])` gives identical output twice.

  Verify `python -m rfml_uav.data.impairments` prints its OK line.
- [x] 1.3 Add an `assert` to the self-check that `iq_wpd_features(impair(w, "gain", s, rng, step=False))` equals `iq_wpd_features(w)` within 1e-4, documenting the DroneDetect gain invariance. Verify the self-check still passes.
- [x] 1.4 Add `impairments.py` to the package section of `CLAUDE.md` (one line: 4 impairments × severities, real/complex, identity at 0, deterministic per row). Verify the line is present.

## 2. Shared paired statistics

- [x] 2.1 Add `src/rfml_uav/eval/stats.py` with `paired_test(a, b, n_boot=10_000, seed=0) -> dict(diff, ci_lo, ci_hi, p)` per design D8. Its `__main__` self-check:
  - a constant shift of 0.05 plus small noise over 25 pairs gives p < 0.05 and a CI excluding 0;
  - identical inputs give diff = 0.

  Verify `python -m rfml_uav.eval.stats` passes, and that a notebook cell recomputing Phase 1's BLUE drop from the 25 `active_linear` run files reproduces 0.145 [0.127, 0.162].
- [x] 2.2 Add `stats.py` to the package section of `CLAUDE.md`; verify the line is present.

## 3. Impaired feature sets (notebook 4, part 1)

- [x] 3.1 Create `notebooks/paper3-label-free-adaptation/4_Receiver_Impairment_Shift.ipynb`: setup cell, paths (`phase2/{dronedetect,dronerf}`), `CELLS = product(SEVERITY, [1, 2, 3])`. Verify it runs top to bottom with no outputs yet.
- [x] 3.2 DroneDetect extraction:
  - one joblib task per CLEAN recording;
  - read the active windows once by `start`, then impair and featurise all cells with `default_rng([0, KIND_ID, s, row])`;
  - write per-recording parts, then merge into `{kind}_{s}.npz` (`X`, `row`) with tmp + `os.replace`, skipping existing files.

  Verify:
  - row counts equal the number of CLEAN active rows;
  - the severity-0 features of 200 random rows equal `features.npz` within 1e-5;
  - re-running with `n_jobs=1` on 2 recordings gives identical `X`.
- [x] 3.3 DroneRF extraction: same pattern over the `~is_dup` rows, grouped by `(bui, n)`. L and H are impaired with distinct row ids (`2·row`, `2·row+1`), then `wpd_features(L) + wpd_features(H)`. Verify the row count is ≈ 164k and that severity-0 features of 200 random rows equal `Wavelets/features.npz` within 1e-5.
- [x] 3.4 Gain-invariance cell: on 1000 rows per dataset, apply the constant gain offset only at severities 1–3 and featurise.
  - DroneDetect: assert the features are unchanged within 1e-4;
  - DroneRF: keep the features for 4.4.

  Verify the assert passes.

## 4. Source-only scoring and E2 report (notebook 4, part 2)

- [x] 4.1 Score the DroneDetect models: for each `active_linear_s{seed}_f{fold}` checkpoint (`linear_tail=True`), run `predict_logits` on the clean and impaired `test_idx` rows; save `phase2/runs/dronedetect_s{seed}_f{fold}.npz` keyed by cell. Verify the clean macro-F1 of every run equals the stored Phase 1 `test_logits` macro-F1 (diff < 1e-6).
- [x] 4.2 Score the DroneRF models: recover the test rows per design D6. Assert the recomputed `y_true` and `test_groups` equal those stored in `phase0/recording_dedup_s*_f*.npz`, then score as in 4.1 with the default tail. Verify the clean macro-F1 equals the stored Phase 0 run's.
- [x] 4.3 Build `phase2/shift_b_table.csv`: per (dataset, kind, severity), accuracy and macro-F1 mean ± std, plus the drop vs clean with `paired_test` CI and p. Also save `shift_b_per_class.csv` (per-class F1 per cell) and a drop-vs-severity figure (`shift_b_drop.png`, one line per kind, one panel per dataset). Verify the table has 1 + 12 rows per dataset.
- [x] 4.4 Add the energy and max-softmax AUROC (correct vs wrong) per DroneDetect cell to `shift_b_auroc.csv`, plus the DroneRF gain-offset-only drop from 3.4 to `gain_invariance.csv`. Verify both files exist and every DroneDetect cell has both AUROCs.
- [x] 4.5 Escalation cell (design D9): list the kinds whose DroneDetect severity-3 mean drop is < 3 pp. For each, extract severity 4 for both datasets (3.2/3.3 cells with `s=4`), score them (4.1/4.2), and append the rows marked `escalation=True`. Verify severity-4 rows exist exactly for the listed kinds.

## 5. Write-up

- [x] 5.1 Add a "Phase 2 — Receiver-impairment shift (shift B, E2)" section to `notebooks/paper3-label-free-adaptation/README.md`: protocol, severity table, the shift-B table, AUROC summary, the gain-invariance finding, escalation outcome, limitations (simulated receiver, one impairment draw, per-window AGC full scale). Verify every number in it matches the CSVs.
- [x] 5.2 Add a dated "Phase 2 outcome" entry to `NOTES.md`: interpretation (which impairment hurts most and why, how shift B compares with shift A), plus the next step (E3 adaptation baselines on shift A + B targets). Verify the entry is present.

## Workflow follow-up

- Review results, then archive the change with `/opsx:archive` and sync specs.
- Commit on a fresh branch (the current branch name refers to Phase 0).
