# Spec Delta

## Purpose

Simulates receiver-side distortions (shift B) on raw RF windows: 4 impairments at 3 fixed severities, applied the same way to real DroneRF channels and complex DroneDetect I/Q. This gives a controlled receiver shift without a second physical receiver.

## ADDED Requirements

### Requirement: Impairment catalogue
The simulator SHALL provide exactly 4 impairments (`awgn`, `gain`, `tilt`, `adc`) at severities 1–3, plus an escalation severity 4 that is used only under the escalation rule of `impairment-shift-evaluation`. Parameters are fixed by the table in the lookup scenario.

#### Scenario: Catalogue lookup
- **WHEN** an impairment name and severity 1–4 are requested
- **THEN** these parameters are applied:

| impairment | severity 1 | severity 2 | severity 3 | severity 4 (escalation) |
|---|---|---|---|---|
| awgn: SNR vs window power | 20 dB | 10 dB | 5 dB | 0 dB |
| gain: offset and in-window step | 3 dB | 6 dB | 12 dB | 20 dB |
| tilt: edge-to-edge slope | 3 dB | 6 dB | 12 dB | 20 dB |
| adc: bits / clip at k × RMS | 10 / 4 | 8 / 3 | 6 / 2 | 4 / 1.5 |


#### Scenario: Unknown request
- **WHEN** an unknown impairment name or a severity outside 0–4 is requested
- **THEN** the simulator raises an error and returns no window

### Requirement: Same output type and length
The simulator SHALL accept a 1-D real or complex window and return a window of the same length and kind (real in → real out, complex in → complex out). Complex windows SHALL get complex noise and per-component quantisation.

#### Scenario: Real window
- **WHEN** a real DroneRF L or H window is impaired
- **THEN** the output is real and has the input length

#### Scenario: Complex window
- **WHEN** a complex DroneDetect I/Q window is impaired
- **THEN** the output is complex and has the input length

### Requirement: Severity 0 is identity
Severity 0 SHALL return the input window unchanged for every impairment.

#### Scenario: Identity reproduces cached features
- **WHEN** cached Phase 0 / Phase 1 windows are passed through severity 0 and then the unchanged feature code
- **THEN** the features equal the cached features within float32 tolerance

### Requirement: Deterministic randomness
Every random draw (noise, gain-step position and sign, tilt sign) SHALL depend only on (seed, impairment, severity, row identifier), not on processing order or the number of workers.

#### Scenario: Repeat run
- **WHEN** the same window is impaired twice with the same seed, impairment, severity and row identifier
- **THEN** both outputs are identical

#### Scenario: Parallel run
- **WHEN** a feature set is built with a different number of parallel workers
- **THEN** the impaired features are identical

### Requirement: Measured impairment matches its parameter
Each impairment SHALL measurably hit its nominal parameter on a synthetic white-noise window.

#### Scenario: AWGN level
- **WHEN** `awgn` severity s is applied
- **THEN** the measured ratio of input power to added-noise power is within 0.5 dB of the table SNR

#### Scenario: Tilt slope
- **WHEN** `tilt` severity s is applied to white noise
- **THEN** the average output power in the top 10% of the band differs from the bottom 10% by 0.9 × the table slope ± 1 dB (the band centres are 0.9 of the span apart)

#### Scenario: Gain step
- **WHEN** `gain` severity s is applied
- **THEN** the RMS after the step differs from the RMS before it by the table value ± 0.5 dB

#### Scenario: ADC clipping
- **WHEN** `adc` severity s is applied
- **THEN** no output component exceeds k × the input component RMS, and the output takes at most 2^bits distinct values per component
