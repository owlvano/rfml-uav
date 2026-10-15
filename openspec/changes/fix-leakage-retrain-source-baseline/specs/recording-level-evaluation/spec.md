# Spec Delta

## Purpose

Defines how DroneRF classifiers are split, selected and scored so that reported accuracy reflects generalisation to unseen recordings, and the chunk-split leakage effect can be quantified side by side.

## ADDED Requirements

### Requirement: Recording-grouped splits
In `recording` mode, the split SHALL assign every chunk of a given `(bui, n)` recording to exactly one of train, val or test within a fold, stratified by class label. Test folds SHALL partition all recordings across K outer folds.

#### Scenario: No recording crosses sets
- **WHEN** a recording-mode fold is generated
- **THEN** the sets of `(bui, n)` groups in train, val and test are pairwise disjoint

#### Scenario: Every class in every test fold
- **WHEN** K = 5 outer folds are generated on DroneRF
- **THEN** each test fold contains at least one recording of each of the 4 classes

### Requirement: Legacy chunk split for comparison
In `chunk` mode, the split SHALL stratify individual chunks without regard to recording, reproducing the legacy paper-1 protocol, and SHALL still provide a separate validation set carved from train.

#### Scenario: Chunk mode leaks by design
- **WHEN** a chunk-mode fold is generated
- **THEN** train, val and test are disjoint by chunk index, and are permitted to share recordings

### Requirement: Validation-only model selection
Checkpoint choice (epoch and fold selection for export) SHALL use the validation set only. Test-fold metrics SHALL NOT influence any training or selection decision.

#### Scenario: Exported checkpoint selection
- **WHEN** a source model is chosen for ONNX export
- **THEN** the choice is made by validation macro-F1, and its test metrics are reported but not used for the choice

### Requirement: Splits are deterministic per seed
Given the same data, mode and seed, the split SHALL produce identical index sets.

#### Scenario: Re-run reproducibility
- **WHEN** the split is generated twice with seed 3
- **THEN** both runs return identical train/val/test indices for every fold

### Requirement: Source-baseline report
The baseline report SHALL give, per split mode and dedup setting, mean ± std over seeds of: accuracy, macro-F1, per-class precision/recall/F1. It SHALL also give the aggregated confusion matrix and histograms of max-softmax confidence, split into correct and incorrect predictions.

#### Scenario: Leakage table
- **WHEN** the baseline notebook completes with 5 seeds
- **THEN** a table compares `chunk` vs `recording` mode (with and without dedup), and each cell shows mean ± std over 5 seeds × 5 folds

#### Scenario: Significance of the split effect
- **WHEN** the leakage table is rendered
- **THEN** for each dedup setting it reports the paired Wilcoxon p-value and the 95% bootstrap CI of the mean accuracy and macro-F1 difference (chunk − recording) over matched (seed, fold) runs

### Requirement: Energy-score separability report
The baseline report SHALL give, for the exported recording-mode source model's configuration, histograms of the energy score `-logsumexp(logits)` split into correct and incorrect predictions, and the AUROC of both the energy score and max-softmax confidence for separating correct from incorrect predictions.

#### Scenario: Logit range check
- **WHEN** the baseline notebook completes
- **THEN** it reports energy-score and max-softmax AUROC side by side, and states whether the energy score is at least as separable as max-softmax
