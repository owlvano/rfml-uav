# normalised-source-evaluation Specification

## Purpose

Defines how the receiver-normalised DroneDetect source model is trained and compared with the Phase 1 source on every existing shift (E1, E2, E3). It also fixes the rule that decides which source OG-TCST builds on.

## Requirements

### Requirement: Normalised source protocol
Normalised source models SHALL follow the Phase 1 protocol exactly, except that inputs are session-normalised: CLEAN active rows, the same class balancing and recording folds per seed, linear tail, 5 seeds × 5 folds, checkpoint by validation macro-F1. Phase 1 checkpoints and results SHALL stay unchanged.

#### Scenario: Identical folds
- **WHEN** a normalised model is trained for (seed, fold)
- **THEN** its train, validation and test rows equal those of the Phase 1 model for the same (seed, fold)

#### Scenario: Statistics from the same recording only
- **WHEN** any window is normalised for training or scoring
- **THEN** its statistics come from noise windows of its own recording (session) or of its own stream (stream), never from another recording

### Requirement: Shortcut re-test
The noise-only shortcut test of Phase 1 SHALL be repeated on session-normalised noise windows. The report SHALL give 7-class accuracy against chance (1/7) and receiver-group accuracy against chance (1/2), next to the Phase 1 values.

#### Scenario: Shortcut verdict
- **WHEN** the re-test completes
- **THEN** the report states whether noise-only 7-class accuracy is still above 2 × chance (0.286)

### Requirement: Side-by-side re-scoring
E1 (CLEAN / BLUE / WIFI / BOTH, per mode), E2 (4 impairments × severities 1–3, plus tilt 4) and E3 (source-only, AdaBN, TENT, PL on the 8 targets, with the E3 splits and selected hyperparameters) SHALL be reported for both sources. Each row SHALL give the normalised − Phase 1 macro-F1 difference, paired by (seed, fold), with Wilcoxon p and 95% bootstrap CI.

#### Scenario: Paired rows
- **WHEN** a condition is reported
- **THEN** it shows both sources' mean ± std and their paired difference over the same 25 (seed, fold) runs

#### Scenario: Both estimators
- **WHEN** E1 and E2 are reported for the normalised source
- **THEN** rows exist for the session estimator and the stream estimator

#### Scenario: Same E3 protocol
- **WHEN** E3 is re-run on the normalised source
- **THEN** it uses the stored Phase 3 splits and `selected_hparams.json` without re-tuning

### Requirement: Adoption rule
The normalised source SHALL be adopted as OG-TCST's base model if and only if its source-only macro-F1, averaged per run over the 8 adoption targets (session estimator), exceeds the Phase 1 source's with paired Wilcoxon p < 0.05. The adoption targets SHALL be the E3 targets with `gain_3` replaced by `agc_3` (amended rule). The report SHALL also give the result of the original rule (E3 targets as they are) and the CLEAN macro-F1 difference.

#### Scenario: Decision recorded
- **WHEN** the comparison completes
- **THEN** the report names the adopted source, the pooled difference with CI and p, and the CLEAN difference

#### Scenario: Amendment disclosed
- **WHEN** the decision is reported
- **THEN** both the original-rule and the amended-rule results appear, with the reason for the amendment: `gain_3` steps every window, including the noise windows the normalisation is estimated from, and the realistic rare-step check was specified before the original rule was evaluated

### Requirement: Rejected-variant ablation
The standard-deviation-normalised variant ((f − μ) / σ), which was trained and scored first, SHALL be kept as an ablation, with its results and checkpoints moved aside rather than deleted. The report SHALL give its E1 and E2 rows next to the adopted centring variant.

#### Scenario: Ablation rows present
- **WHEN** the Phase 4 report is written
- **THEN** E1 and E2 show Phase 1, centring and std-normalised rows for every condition

### Requirement: Realistic AGC check
Besides the E2 gain cells, which step every window, the report SHALL score a rare-step AGC condition: a fixed fraction of windows (active and noise alike, chosen per row with a fixed seed) take their gain-impaired features, and all others stay clean. Both sources SHALL be scored on it.

#### Scenario: Rare steps
- **WHEN** the AGC condition is built for gain severities 1–3 at a 5% step fraction
- **THEN** about 5% of the rows carry gain-impaired features, the same rows for both sources, and the report gives both sources' macro-F1 with the paired difference
