# Spec Delta

## ADDED Requirements

### Requirement: Supplementary realistic-AGC target
Besides the 8 E3 targets, E3 SHALL evaluate a supplementary shift-B target `agc_3`: CLEAN rows where a fixed 5% of windows (chosen per row with a fixed seed, active and noise alike) carry their `gain_3` features and the rest stay clean. It SHALL use the shift-B split rule and the selected hyperparameters, and be reported in a separate table, so that the 8-target E3 table is unchanged.

#### Scenario: Main table unchanged
- **WHEN** E3 is re-run with the supplementary target for the Phase 1 source
- **THEN** the 8-target E3 table files are byte-identical to before

#### Scenario: Same stepped rows everywhere
- **WHEN** `agc_3` features are built for any source or notebook
- **THEN** the set of stepped rows is identical
