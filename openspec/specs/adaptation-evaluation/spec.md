# adaptation-evaluation Specification

## Purpose

Defines the E3 protocol: how label-free adaptation methods are scored on the DroneDetect shift-A and shift-B targets without leakage or target-label tuning. The resulting table is the bar that OG-TCST is compared against.

## Requirements

### Requirement: Target set
E3 SHALL evaluate exactly these 8 targets: shift A BLUE, WIFI, BOTH; shift B awgn 2, awgn 3, gain 3, adc 3, tilt 4. Rows of a (drone, mode) with no CLEAN recording SHALL be excluded.

#### Scenario: Unseen condition excluded
- **WHEN** shift-A targets are built
- **THEN** no Phantom 4 FY row is in any adapt or test split

### Requirement: Leak-free adapt/test split
For each (seed, fold) model and target, the target recordings SHALL be split into two disjoint halves (adapt, test) by recording, stratified so every drone class appears in both halves. Shift-B targets SHALL use only that model's CLEAN test recordings, in impaired form.

#### Scenario: Recording-disjoint halves
- **WHEN** a split is generated
- **THEN** no recording identifier appears in both the adapt and the test half, and every class has at least one recording in each half

#### Scenario: Shift-B rows unseen in training
- **WHEN** a shift-B split is generated for a model
- **THEN** none of its recordings is in that model's train or validation set

#### Scenario: Same test rows for all methods
- **WHEN** methods are compared on a (model, target)
- **THEN** source-only and every adapted model are scored on the identical test rows

### Requirement: Dev-shift hyperparameter selection
Each method's hyperparameters SHALL be chosen once, by the mean macro-F1 gain over source-only on the dev shifts (shift-B awgn 1, gain 2, adc 2, tilt 3) using the 5 seed-0 models, and then frozen for all targets. No target label SHALL influence any choice.

#### Scenario: Dev cells disjoint from targets
- **WHEN** hyperparameters are selected
- **THEN** none of the dev cells is a target, and the selected settings are saved before any target is adapted

#### Scenario: Tie-break by cost
- **WHEN** two settings are within 0.1 pp mean dev gain
- **THEN** the one with fewer backward passes per window is chosen

### Requirement: E3 table
For each (target, method), the report SHALL give macro-F1 and accuracy (mean ± std over the 25 runs), the gain over source-only with paired Wilcoxon p and 95% bootstrap CI, and the recovered fraction of the drop.

#### Scenario: Table shape
- **WHEN** E3 completes
- **THEN** the table has one row per (target, method) for 8 targets × 4 methods (source-only, AdaBN, TENT, pseudo-labelling)

#### Scenario: Recovered fraction
- **WHEN** a gain is reported
- **THEN** it also gives (adapted − source) / (clean − source) macro-F1. For shift B, clean is the same model on the clean features of the same test rows. For shift A, it is the model's stored Phase 1 CLEAN test macro-F1

#### Scenario: Best baseline named
- **WHEN** E3 completes
- **THEN** the report names, per target, the baseline with the highest mean macro-F1 (the reference for OG-TCST's Wilcoxon test)

### Requirement: Separability after adaptation
For each (target, method), the report SHALL give the energy-score and max-softmax AUROC for separating correct from wrong predictions on the test half.

#### Scenario: AUROC per row
- **WHEN** E3 completes
- **THEN** every (target, method) row carries both AUROCs as mean ± std

### Requirement: Cost table
For each method at its selected hyperparameters, the report SHALL give forward and backward passes per adapt window, the number of parameters updated, and the median adaptation wall time per run.

#### Scenario: Cost rows
- **WHEN** E3 completes
- **THEN** the cost table has one row per method, labelled as a hardware-independent proxy (GPU wall time is not node energy)

### Requirement: Supplementary realistic-AGC target
Besides the 8 E3 targets, E3 SHALL evaluate a supplementary shift-B target `agc_3`: CLEAN rows where a fixed 5% of windows (chosen per row with a fixed seed, active and noise alike) carry their `gain_3` features and the rest stay clean. It SHALL use the shift-B split rule and the selected hyperparameters, and be reported in a separate table, so that the 8-target E3 table is unchanged.

#### Scenario: Main table unchanged
- **WHEN** E3 is re-run with the supplementary target for the Phase 1 source
- **THEN** the 8-target E3 table files are byte-identical to before

#### Scenario: Same stepped rows everywhere
- **WHEN** `agc_3` features are built for any source or notebook
- **THEN** the set of stepped rows is identical
