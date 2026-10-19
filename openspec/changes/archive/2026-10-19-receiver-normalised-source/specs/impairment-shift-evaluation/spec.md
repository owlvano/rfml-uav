# Spec Delta

## MODIFIED Requirements

### Requirement: Impaired feature sets
For every (dataset, impairment, severity), the impaired features SHALL be computed from re-read raw windows with the unchanged clean feature code. They SHALL be row-aligned with the clean feature file and carry its provenance. DroneDetect covers all CLEAN active rows and, in a separate set per cell, all CLEAN noise-only rows; DroneRF covers all non-duplicate rows, with L and H impaired independently.

#### Scenario: Row alignment
- **WHEN** an impaired feature file is loaded
- **THEN** its row i is the impaired version of the clean row with the same index, recording and window start

#### Scenario: No impaired interference rows
- **WHEN** DroneDetect impaired sets are built
- **THEN** they hold no BLUE, WIFI or BOTH rows

#### Scenario: Clean data untouched
- **WHEN** impaired sets are built
- **THEN** the Phase 0 / Phase 1 feature files, run files and checkpoints are unchanged

#### Scenario: Noise rows impaired like active rows
- **WHEN** the DroneDetect noise-row set of a cell is built
- **THEN** each noise row is impaired with the same impairment, severity and per-row seeding rule as active rows, and the existing active-row files are unchanged
