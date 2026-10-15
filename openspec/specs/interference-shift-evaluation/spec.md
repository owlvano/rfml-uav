# interference-shift-evaluation Specification

## Purpose

Defines how the DroneDetect source model is trained on clean recordings and scored under Bluetooth, Wi-Fi and combined interference, so that the accuracy loss caused by interference (RQ1) is measured without leakage, and receiver-configuration shortcuts are detected and reported.

## Requirements

### Requirement: Source trained on CLEAN only
Source models SHALL be trained and selected using CLEAN active windows only, with 7 drone classes and no Background class. Splits SHALL be by recording (each `.dat` file in exactly one of train, val, test), and checkpoint choice SHALL use validation macro-F1 only.

#### Scenario: No interference data in training
- **WHEN** a source model is trained
- **THEN** no BLUE, WIFI or BOTH row is in its train or validation set

#### Scenario: Recording-disjoint folds
- **WHEN** a fold is generated
- **THEN** train, val and test recording identifiers are pairwise disjoint, and every class has at least one test recording

### Requirement: Interference table
Each source model SHALL be scored on its CLEAN test recordings and on every BLUE, WIFI and BOTH recording. The report SHALL give accuracy, macro-F1 and per-class F1 per subset, as mean ± std over 5 seeds × 5 folds, plus the drop from CLEAN.

#### Scenario: Per-subset rows
- **WHEN** the baseline completes
- **THEN** the table has one row per subset (CLEAN, BLUE, WIFI, BOTH), each with mean ± std and the drop in macro-F1 relative to CLEAN

#### Scenario: Per-mode breakdown
- **WHEN** the baseline completes
- **THEN** a second table splits each subset by mode (ON, HO, FY)

#### Scenario: Significance of the drop
- **WHEN** the drop from CLEAN is reported for a subset
- **THEN** it carries a paired Wilcoxon p-value and 95% bootstrap CI over the 25 (seed, fold) runs

### Requirement: Unseen conditions excluded from the shift comparison
Target rows whose (drone, mode) has no CLEAN recording SHALL be excluded from the interference table and reported separately.

#### Scenario: Phantom fly
- **WHEN** WIFI or BOTH Phantom 4 FY rows are scored
- **THEN** they are absent from the interference table and appear in a separate "unseen condition" line

### Requirement: Receiver-shortcut test
The same CNN and protocol SHALL be trained on noise-only CLEAN windows. The report SHALL give its 7-class accuracy against chance (1/7), and its accuracy at telling the two receiver groups apart (AIR/DIS/PHA vs INS/MIN/MP1/MP2) against chance (1/2).

#### Scenario: Shortcut flagged
- **WHEN** noise-only 7-class accuracy exceeds 2 × chance (0.286)
- **THEN** the report marks a receiver shortcut, and the source baseline is re-run with the mitigation of the design, with both results reported

#### Scenario: No shortcut
- **WHEN** noise-only 7-class accuracy is at most 2 × chance
- **THEN** the report states that the noise floor alone does not identify the drone

### Requirement: Energy-score separability on DroneDetect
For each logit tail (default and plain linear), the report SHALL give the AUROC of the energy score and of max-softmax for separating correct from wrong predictions, on CLEAN test and on each interference subset, as mean ± std over runs.

#### Scenario: Tail choice
- **WHEN** both tails have been evaluated
- **THEN** the report names the tail used from now on: the one whose energy-score AUROC is highest on the pooled interference subsets, unless its CLEAN macro-F1 is more than 1 pp below the other tail's
