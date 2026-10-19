# impairment-shift-evaluation Specification

## Purpose

Measures how much receiver impairments (shift B) cost the frozen paper-3 source models on DroneDetect and DroneRF (E2). The same impaired target sets are reused later as the shift-B targets for label-free adaptation.

## Requirements

### Requirement: Impaired feature sets
For every (dataset, impairment, severity), the impaired features SHALL be computed from re-read raw windows with the unchanged clean feature code. They SHALL be row-aligned with the clean feature file and carry its provenance. DroneDetect covers all CLEAN active rows; DroneRF covers all non-duplicate rows, with L and H impaired independently.

#### Scenario: Row alignment
- **WHEN** an impaired feature file is loaded
- **THEN** its row i is the impaired version of the clean row with the same index, recording and window start

#### Scenario: No impaired interference rows
- **WHEN** DroneDetect impaired sets are built
- **THEN** they hold no BLUE, WIFI or BOTH rows

#### Scenario: Clean data untouched
- **WHEN** impaired sets are built
- **THEN** the Phase 0 / Phase 1 feature files, run files and checkpoints are unchanged

### Requirement: Source-only scoring without retraining
Each of the 25 saved Phase 1 linear-tail DroneDetect models and the 25 saved Phase 0 `recording_dedup` DroneRF models SHALL be scored on the impaired version of its own test rows only. No model is trained, fine-tuned or re-selected on impaired data.

#### Scenario: Same test rows as clean
- **WHEN** a model is scored on an impaired cell
- **THEN** the row set equals that model's clean test rows

#### Scenario: Clean reference reproduced
- **WHEN** a model is scored at severity 0
- **THEN** its macro-F1 equals its stored Phase 0 / Phase 1 clean test macro-F1

### Requirement: Shift-B table
The report SHALL give, per (dataset, impairment, severity), accuracy and macro-F1 as mean ± std over the 25 runs, and the drop in macro-F1 from the same model's clean test score with a paired Wilcoxon p-value and 95% bootstrap CI.

#### Scenario: Table shape
- **WHEN** E2 completes
- **THEN** each dataset has one clean row plus one row per (impairment, severity), each with mean ± std, drop, CI and p-value

#### Scenario: Per-class view
- **WHEN** E2 completes
- **THEN** per-class F1 per cell is saved, so the classes most hurt by each impairment can be named

### Requirement: Energy-score separability under shift B
For every DroneDetect cell, the report SHALL give the AUROC of the energy score and of max-softmax for separating correct from wrong predictions, as mean ± std over runs.

#### Scenario: AUROC per cell
- **WHEN** E2 completes
- **THEN** each DroneDetect (impairment, severity) row carries energy and max-softmax AUROC

### Requirement: Severity escalation
If an impairment's severity-3 macro-F1 drop on DroneDetect is below 3 pp, severity 4 of that impairment SHALL be added for both datasets and reported. Severity 4 SHALL NOT be added otherwise.

#### Scenario: Weak impairment
- **WHEN** an impairment loses less than 3 pp mean macro-F1 at severity 3 on DroneDetect
- **THEN** its severity-4 row is computed and reported, marked as escalation

#### Scenario: Strong impairment
- **WHEN** an impairment loses 3 pp or more at severity 3 on DroneDetect
- **THEN** no severity-4 row exists for it

### Requirement: Gain-invariance check
The report SHALL state whether a constant gain change alone (no in-window step) changes the features of each dataset.

#### Scenario: DroneDetect invariant
- **WHEN** a constant gain offset alone is applied to DroneDetect windows
- **THEN** the features equal the clean features within float32 tolerance, and the report says so

#### Scenario: DroneRF sensitive
- **WHEN** a constant gain offset alone is applied to DroneRF windows
- **THEN** the report gives the resulting macro-F1 drop for that offset

### Requirement: Receiver-shortcut check under shift B
The saved Phase 1 in-band models SHALL be scored on the same DroneDetect impaired sets. The report SHALL give, per cell, the full-band and in-band macro-F1 drops and their paired difference with Wilcoxon p and 95% bootstrap CI. Under AWGN it SHALL also give the relative feature shift of in-band vs out-of-band wavelet nodes.

#### Scenario: Shortcut share of the AWGN loss
- **WHEN** E2 completes
- **THEN** each DroneDetect cell has a full-band − in-band drop difference with CI and p, and each AWGN cell has the in-band and out-of-band relative node shift
