# gated-adaptation-evaluation Specification

## Purpose
Defines how the receiver-gated and drone-balanced adaptation methods are evaluated (Phase 6, a disclosed post-hoc follow-up to Phase 5): harm-aware dev tuning frozen before targets, comparison with the frozen Phase 5 rows on identical streams, gate diagnostics and an oracle bound.

## Requirements

### Requirement: Disclosed post-hoc status
Every Phase 6 table and write-up SHALL state that the methods were designed after the Phase 5 target results were seen. It SHALL report the Phase 5 frozen numbers alongside, never in their place.

#### Scenario: Side by side
- **WHEN** the Phase 6 comparison table is built
- **THEN** it contains the frozen Phase 5 source-only, continuous, TENT and triggered rows for the same streams

### Requirement: Identical streams
Phase 6 streams SHALL be the Phase 5 receiver-condition streams: the same recordings, order, centring and active windows for each (target, model).

#### Scenario: Reproduced source-only
- **WHEN** a Phase 6 stream is built
- **THEN** its active rows equal the Phase 5 stream file's rows, and source-only votes on it equal the Phase 5 source-only votes

### Requirement: Harm-aware dev tuning, frozen before targets
φ_on, Bc, D and m SHALL be chosen on the Phase 5 dev cells (awgn_1, gain_2, adc_2, tilt_3) with the seed-0 models. The objective is mean target vote F1 minus the mean positive part of the prefix vote F1 loss against source-only. The choice SHALL be saved before any target stream runs, and no target label SHALL influence it.

#### Scenario: Saved first
- **WHEN** target streams run
- **THEN** they read every Phase 6 hyperparameter from the saved file, which predates their outputs

### Requirement: Comparison table
Per (target, method), the report SHALL give target vote macro-F1 and prefix vote macro-F1 (mean ± std over 25 runs), and paired differences (Wilcoxon, 95% bootstrap CI) against source-only and against Phase 5 continuous AdaBN. Targets are the Phase 5 targets.

#### Scenario: Both references
- **WHEN** a Phase 6 method row is reported
- **THEN** it gives its difference to source-only and to Phase 5 continuous AdaBN, on target and on prefix

### Requirement: Gate diagnostics and oracle bound
For gated methods, the report SHALL give the fraction of active windows with the gate ON in the prefix and in the target segment, and the delay from the change point to the first ON in the target segment. An oracle run (gate ON exactly from the change point) SHALL be reported as a diagnostic upper bound, labelled as using the protocol's change point.

#### Scenario: Gate table
- **WHEN** gated methods are evaluated
- **THEN** each target row gives prefix ON fraction, target ON fraction, detection rate and median delay

### Requirement: Cost
Per method, the report SHALL give forward and backward passes per active window and updates per stream, in the Phase 5 ledger's units.

#### Scenario: Ledger rows
- **WHEN** the cost table is written
- **THEN** it has one row per Phase 6 method, comparable with `phase5/cost_ledger.csv`
