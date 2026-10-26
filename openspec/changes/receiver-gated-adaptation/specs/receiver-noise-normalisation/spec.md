# Spec Delta

## ADDED Requirements

### Requirement: Noise-floor drift
On request, the stream estimator SHALL report a drift value for every normalised window: the L1 distance between the state used for that window and the warm-up state, divided by the L1 norm of the warm-up state. Windows before warm-up completes have no drift value. Reporting drift SHALL NOT change the normalised output.

#### Scenario: Stationary noise
- **WHEN** a stream's noise windows keep the warm-up distribution
- **THEN** the reported drift stays near zero

#### Scenario: Floor change
- **WHEN** every feature of the noise windows is scaled by a constant factor a after warm-up
- **THEN** the drift approaches |a − 1| as the state converges

#### Scenario: Output unchanged
- **WHEN** the estimator runs with and without drift reporting on the same stream
- **THEN** the normalised windows and their validity flags are identical
