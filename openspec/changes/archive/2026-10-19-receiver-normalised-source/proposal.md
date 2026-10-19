# Proposal

## Why

Phases 1–3 established three facts:
- the DroneDetect source model leans on a receiver shortcut: the noise floor alone identifies the receiver group with 0.98 accuracy;
- AWGN loss is mostly that shortcut breaking (Phase 2: 40–75%);
- no adaptation baseline fixes interference (Phase 3).

The explore session of 2026-10-19 also showed that the plan's component 3.2 cannot be bolted onto the frozen Phase 1 model:
- moment-matching shift-A features to stream statistics *lowers* macro-F1 by 1–9 pp;
- per-recording statistics, which a single-drone stream converges to, collapse it to 0.08–0.19.

A spike that applied the normalisation **at training and test time**, with statistics from noise-only windows, behaved differently (seed 0, 5 folds):
- CLEAN macro-F1 0.722 → 0.687;
- BLUE 0.581 → 0.616 and BOTH 0.462 → 0.491;
- the CLEAN − BOTH gap shrank from 26.0 to 19.6 pp, with no adaptation step.

This change makes that the paper's **receiver-normalised source** (thesis 3.2) and measures it on every existing shift, before OG-TCST (3.3) builds on it. It serves **RQ2** (label-free robustness to receiver and air shifts; RQ numbering to be confirmed against the plan) and re-scores **E1, E2 and E3**. It feeds **thesis chapter 3, section 3.2**.

## What Changes

- **Receiver-noise normalisation** (new shared module). Each window's 256-d WPD vector is centred per feature on the mean of the noise-only windows of the same receiver session. Two estimators:
  - **session:** all noise windows of the recording (offline; used for training and the main tables);
  - **stream:** EMA over noise windows in time order (deployable; the plan's 3.2 formula, fed by noise windows instead of all windows).
- **Normalised source models.** The Phase 1 protocol is unchanged otherwise:
  - CLEAN active rows, the same class balancing and recording splits (identical folds), linear tail;
  - 5 seeds × 5 folds, checkpoint by validation macro-F1.

  Saved as a new set. The Phase 1 models are not touched.
- **Shortcut re-test.** The noise-only CNN is retrained on normalised noise windows, reporting 7-class and receiver-group accuracy against chance. Normalisation should remove the floor cue.
- **Impaired noise windows** for shift B: notebook 4's builder is extended to the CLEAN noise rows (same cells, same per-row seeding). Shift-B sets then carry both the active rows and the noise rows that normalise them.
- **Re-scoring with the new source**, side by side with the Phase 1 rows (paired by seed and fold):
  - **E1:** interference table (CLEAN / BLUE / WIFI / BOTH, per mode);
  - **E2:** shift-B table (4 impairments × severities 1–3, plus tilt 4);
  - **E3:** the adaptation baselines re-run on top of the normalised source, same splits and selected hyperparameters.
- **Centring, not standardisation (author decision 2026-10-19, during apply).** The first run divided by the noise std as well. That variant removed the shortcut, but the noise σ reacts to impairments differently from active windows (a 3 dB gain step shrinks it to 0.56×), and gain collapsed by 27–30 pp. A one-seed spike of centring alone gave CLEAN 0.809, BOTH 0.585 and awgn 3 0.546, against 0.722 / 0.462 / 0.405 for Phase 1, but chance level on gain (steps in every window) and 0.38 on tilt 4. The std variant is kept as an ablation.
- **Realistic AGC check.** The E2 gain cells step every window, noise windows included, which is adversarial for a noise-based correction. A rare-step condition (5% of windows stepped) is scored as well.
- **Adoption rule.** The normalised source becomes OG-TCST's base model if its macro-F1, pooled over the 8 E3 target conditions, beats the Phase 1 source with Wilcoxon p < 0.05. The CLEAN loss is reported, but it isn't a gate. Robustness, not peak clean accuracy, is the paper's objective.
- Out of scope:
  - the open-set gate, temporal-consistency self-training and the stream harness (OG-TCST change);
  - DroneRF;
  - online noise-window detection (windows keep their offline active/noise labels from `select_windows`);
  - any change to Phase 1–3 results or checkpoints.

## Capabilities

### New Capabilities
- `receiver-noise-normalisation`: per-feature centring on the receiver noise mean. Covers the session and stream estimators, their invariance (a per-feature additive receiver change cancels out) and determinism.
- `normalised-source-evaluation`: the normalised source protocol, the shortcut re-test, side-by-side E1/E2/E3 re-scoring, the std-normalised ablation, the realistic AGC check and the adoption rule.

### Modified Capabilities
- `impairment-shift-evaluation`: the DroneDetect impaired sets also cover CLEAN noise-only rows, so impaired recordings can be normalised.

## Impact

- **Code:**
  - new `src/rfml_uav/data/normalise.py` (with an `assert`-based `__main__` self-check);
  - no change to existing modules.
- **Notebooks:**
  - new `notebooks/paper3-label-free-adaptation/6_Receiver_Normalised_Source.ipynb` (training, shortcut test, E1/E2 re-scoring);
  - `4_Receiver_Impairment_Shift.ipynb`: one added extraction step for noise rows; existing outputs unchanged;
  - `5_Adaptation_Baselines.ipynb`: a `SOURCE` switch (default `phase1`, unchanged; `noisenorm` writes to `phase3/noisenorm/`).
- **Data:**
  - writes `checkpoints/paper3/noisenorm/` (25 models), `results/paper3/phase4/` and `results/paper3/phase3/noisenorm/`;
  - adds `results/paper3/phase2/dronedetect_noise/` (≈ 1.3 GB, 13 cells of 95k rows);
  - the external drive is read once more, for CLEAN noise windows.
- **Compute:**
  - ≈ 1 h GPU for 25 models plus 25 noise-only shortcut models;
  - ≈ 5 min CPU for the impaired noise windows;
  - E3 re-run ≈ minutes.
- **Dependencies:** none new.
- **Paper:** 3.2 is described as a train-and-test transform, not test-time only. The stream estimator's warm-up and its dependence on noise-window labelling go in the Limitations.
