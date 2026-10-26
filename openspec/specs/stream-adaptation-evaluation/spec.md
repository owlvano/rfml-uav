# stream-adaptation-evaluation Specification

## Purpose

Defines how the stream mechanisms are compared with the label-free baselines (streaming E3, RQ3) and how the open-set gate is tested on unknown drones (E5-lite, RQ4). Every comparison is on the centred source, with shared voting, dev-shift tuning and a cost ledger that E9 turns into energy.

## Requirements

### Requirement: Receiver-condition streams
Each (model, target) stream SHALL be a clean prefix (the model's CLEAN test recordings, clean features) followed by the target segment (the target's E3 test-half recordings in target features). Recordings are chained in an order fixed by (seed, fold, target), and windows are time-ordered within each recording. Targets SHALL be BLUE, WIFI, BOTH, awgn_2, awgn_3, gain_3, agc_3, adc_3 and tilt_4.

#### Scenario: Unseen recordings only
- **WHEN** a stream is built for a model
- **THEN** none of its recordings is in that model's train or validation set

#### Scenario: Known change point
- **WHEN** a stream is built
- **THEN** the index of the first target window is recorded, and prefix and target metrics are reported separately

### Requirement: Method set with shared voting
Each stream SHALL be run with:
- source-only;
- continuous AdaBN (BN statistics updated every chunk);
- periodic refresh (every P chunks);
- triggered guarded refresh (rung 3a);
- online TENT (one step per chunk);
- online naive PL (one step per chunk on windows with max-softmax ≥ τ);
- triggered vote-PL fine-tune (rung 3b);
- the head-only vote-PL ablation.

Every method SHALL get the same temporal vote and the same gate.

#### Scenario: Same decisions pipeline
- **WHEN** two methods are compared on a stream
- **THEN** both report per-window and vote macro-F1 on identical target-segment windows

### Requirement: Dev-shift tuning, frozen before targets
W, ρ, q, θ, B, D, λ, P and the online learning rates SHALL be chosen on streams built from the dev cells (awgn_1, gain_2, adc_2, tilt_3), using the seed-0 models, and saved before any target stream is run. No target label SHALL influence a choice.

#### Scenario: Saved before targets
- **WHEN** target streams are run
- **THEN** they read every hyperparameter from the saved file, which predates their outputs

### Requirement: Streaming E3 table
Per (target, method), the report SHALL give target-segment vote macro-F1 and per-window macro-F1 (mean ± std over 25 runs), and the prefix vote macro-F1 as a harm check. Each method SHALL be compared with the best other method (by mean vote macro-F1) and with source-only, using paired Wilcoxon and 95% bootstrap CI.

#### Scenario: Best-baseline test
- **WHEN** the table is built
- **THEN** each target names its best method and gives every other method's paired difference to it

#### Scenario: No harm on clean
- **WHEN** a method is reported
- **THEN** its prefix vote macro-F1 difference from source-only is shown with CI

### Requirement: Trigger quality
For triggered methods, the report SHALL give the false-trigger count in the clean prefix and the detection delay (in active windows) after the change point, per target.

#### Scenario: Trigger table
- **WHEN** triggered methods are evaluated
- **THEN** each target row gives mean false triggers in the prefix and mean and median detection delay

### Requirement: Single-drone check
The triggered methods SHALL also run on per-recording streams (one drone each) for the shift-B targets. The report SHALL give their target macro-F1 against source-only and the share of restricted vs full refreshes.

#### Scenario: Guard holds
- **WHEN** a single-drone stream triggers
- **THEN** the report shows whether the restricted refresh was used and whether macro-F1 stayed at or above source-only

### Requirement: E5-lite open-set evaluation
For each held-out class, leave-one-class-out source models SHALL be trained on centred CLEAN features without that class (seed 0, 5 folds, otherwise the Phase 1 protocol). On CLEAN and BOTH test streams that contain the held-out class's recordings, the report SHALL give:
- the energy-score AUROC of known vs unknown windows;
- the unknown false-accept rate and the known rejection rate at the gate threshold;
- known-class vote macro-F1 with the gate on vs off.

#### Scenario: Per held-out class
- **WHEN** E5-lite completes
- **THEN** a row exists per held-out class and condition (CLEAN, BOTH), plus a mean row

### Requirement: Cost ledger
The report SHALL give, per method: forward and backward passes per active window, events per stream, full vs restricted refreshes, parameters or statistics changed per event, and MACs per window. MACs are counted for WPD features (computed analytically from the filter length, the level and the window length) and for the CNN (counted from the layer shapes), and labelled as hardware-independent inputs to E9.

#### Scenario: Ledger rows
- **WHEN** the ledger is written
- **THEN** it has one row per method, and the WPD and CNN MACs per window appear as separate fields
