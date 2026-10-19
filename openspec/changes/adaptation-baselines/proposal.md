# Proposal

## Why

Phases 1–2 showed what the frozen DroneDetect source model loses:
- **shift A (interference):** 14–27 pp of macro-F1;
- **shift B (simulated receiver):** up to 32 pp.

Phase 2 also found that 40–75% of the AWGN loss comes from the receiver shortcut (the out-of-band noise floor). OG-TCST, the paper's method, must beat standard label-free adaptation on these same shifts. Those standard baselines don't exist yet, so neither the bar nor the claim "adaptive normalisation recovers AWGN most" can be checked.

This change serves **RQ2** (does label-free adaptation recover the shift loss; to be confirmed against the plan's RQ list) and experiment **E3** (adaptation baselines). It feeds **thesis chapter 3**: the baselines of section 3.2 and the comparison table of section 3.3. It also records hardware-independent cost per method (passes, updated parameters), which E9 converts to joules for **thesis 5.4**.

## What Changes

- **Three standard baselines** plus the source-only reference, applied to the frozen Phase 1 models (25 runs, linear tail), with no target labels:
  - **AdaBN:** BatchNorm statistics re-estimated on target data. No gradient steps.
  - **TENT:** entropy minimisation that updates only the BatchNorm affine parameters (2,016 of 410k).
  - **Naive pseudo-labelling:** fine-tune the whole network on its own confident predictions on the target.
- **Offline batch regime.** Each method adapts once on an unlabeled *adapt* split, is frozen, and is scored on a disjoint *test* split. Streaming and event-triggered variants come with the OG-TCST change, which re-runs these baselines in its stream harness.
- **Targets (8):**
  - shift A: BLUE, WIFI, BOTH;
  - shift B: awgn 2, awgn 3, gain 3, adc 3, tilt 4.
- **Leak-free adapt/test split by recording, stratified by drone, one per (seed, fold):**
  - **Shift A:** all recordings of the subset are split in half, excluding the unseen Phantom-FY condition.
  - **Shift B:** only the model's own CLEAN *test* recordings are used (in impaired form), split in half. The model never trained on these recordings, even in clean form.
- **Hyperparameter selection on dev shifts.** A small grid per method is tuned with labels on mild shift-B cells that are *not* targets (awgn 1, gain 2, adc 2, tilt 3). The chosen setting is then frozen for every target. Target labels are used only for scoring.
- **E3 table, per (target, method):**
  - macro-F1 and accuracy, mean ± std over 25 runs;
  - gain over source-only, with a paired Wilcoxon test and 95% bootstrap CI;
  - **recovered fraction** of the drop;
  - energy-score / max-softmax AUROC after adaptation (input for the open-set gate).
- **Cost table per method:**
  - forward and backward passes per adapt window;
  - number of parameters updated;
  - measured adaptation wall time on the GPU (a proxy only).
- Out of scope:
  - OG-TCST and the open-set gate;
  - streaming / event-triggered adaptation;
  - DroneRF (author decision 2026-10-19);
  - combined shift A + B;
  - energy measurement on edge hardware (E9);
  - new source models or retraining.

## Capabilities

### New Capabilities
- `adaptation-baselines`: the three label-free adaptation methods as reusable operations on a trained classifier. Behaviour, determinism, which parameters each may change, and the source model left untouched.
- `adaptation-evaluation`: the E3 protocol. Target list, leak-free adapt/test splits, dev-shift hyperparameter selection, the E3 table with significance and recovered fraction, AUROC after adaptation, and the cost table.

### Modified Capabilities
None. Phase 1 models, features and the Phase 2 impaired sets are read unchanged.

## Impact

- **Code:**
  - new `src/rfml_uav/adapt/` package (`baselines.py` with an `assert`-based `__main__` self-check). The OG-TCST notebook will reuse it;
  - no change to existing modules.
- **Notebooks:** new `notebooks/paper3-label-free-adaptation/5_Adaptation_Baselines.ipynb`.
- **Data:**
  - reads `results/paper3/phase1/features.npz`, the Phase 1 run files and checkpoints, and `results/paper3/phase2/dronedetect/*.npz`;
  - writes `results/paper3/phase3/` (splits, logits per run, tables) and nothing under `checkpoints/`. Adapted weights aren't stored; they can be re-derived deterministically.
- **Compute:**
  - the 4 dev cells × the grid on the 5 seed-0 models, then 25 runs × 8 targets × 3 methods;
  - adapt sets hold about 7k–50k windows, and TENT/PL take ≤ 3 epochs;
  - expect well under an hour on the RTX 5070 Ti.
- **Dependencies:** none new.
