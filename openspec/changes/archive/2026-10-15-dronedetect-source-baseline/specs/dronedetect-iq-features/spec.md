# Spec Delta

## Purpose

Turns DroneDetect V2 I/Q recordings into labelled, provenance-carrying feature rows that the existing 256-d WPD-CNN can consume, while keeping only windows that contain drone emissions and removing absolute receiver level from each window.

## ADDED Requirements

### Requirement: Recording inventory from folder layout
The loader SHALL list every DroneDetect recording with its subset (CLEAN, BLUE, WIFI, BOTH), drone, mode (ON, HO, FY), recording index and sample count. Drone and mode SHALL come from the folder name, and the file-name code SHALL be checked against the folder.

#### Scenario: Prefix differs from folder
- **WHEN** a file `MA1_0000_00.dat` sits in folder `CLEAN/MP1_ON/`
- **THEN** its drone is `MP1`, subset `CLEAN`, mode `ON`, recording index 0

#### Scenario: Code mismatch rejected
- **WHEN** a file's 4-digit interference/mode code disagrees with its subset or mode folder
- **THEN** the inventory fails with an error naming that file

#### Scenario: Full dataset
- **WHEN** the inventory runs on DroneDetect V2
- **THEN** it lists 390 recordings over 7 drones

### Requirement: Recordings read as complex samples without full load
A recording SHALL be read as complex samples from interleaved 32-bit float I/Q, using its actual length (recordings are not all 1.2e8 samples), without loading the whole file into memory at once.

#### Scenario: Short recording
- **WHEN** the CLEAN Inspire 2 fly recording 0 (1.76 s) is read
- **THEN** its full length is used and no error is raised

### Requirement: Active windows selected by power
Each recording SHALL be cut into non-overlapping fixed-length windows (4000 samples). Windows whose mean power is at least 6 dB above that recording's quiet floor (10th-percentile window power) SHALL be marked active. At most a fixed number of active windows per recording SHALL be kept, chosen evenly over time.

#### Scenario: Bursty drone
- **WHEN** a Mavic Mini recording has bursts in about a quarter of its windows
- **THEN** only windows 6 dB above its floor are kept as active, and no kept window lies entirely in the quiet gaps

#### Scenario: Cap respected
- **WHEN** a recording has more active windows than the cap
- **THEN** exactly the cap is kept, with start positions spread across the whole recording

### Requirement: Noise-only windows kept for diagnostics
Windows at or below the recording's quiet floor SHALL be kept up to the same per-recording cap and marked as noise-only. They SHALL NOT enter source-model training or the interference table.

#### Scenario: Noise flag
- **WHEN** features are loaded
- **THEN** each row carries an active/noise flag, and filtering on it gives disjoint row sets

### Requirement: Per-window normalisation
Before feature extraction each window SHALL have its complex mean removed and SHALL be scaled to unit mean power.

#### Scenario: Receiver level removed
- **WHEN** the same window is scaled by a constant gain and offset by a constant DC term
- **THEN** its features are unchanged (within float tolerance)

### Requirement: Two-channel WPD features
Each window SHALL yield a 256-d feature vector: the DroneRF WPD features (128 per channel) of the I channel followed by those of the Q channel, computed by the same shared code that DroneRF features use.

#### Scenario: Model compatibility
- **WHEN** a feature row is passed to the CNN built through the model factory with input dimension 256
- **THEN** the forward pass succeeds without architecture changes

### Requirement: Feature rows carry provenance
The feature file SHALL store, per row: subset, drone, mode, recording index, a recording identifier unique across subsets, window start sample, and the active/noise flag. Rows of one recording SHALL appear in ascending start order.

#### Scenario: Group split possible
- **WHEN** a consumer groups rows by recording identifier
- **THEN** each group holds rows from exactly one `.dat` file, in ascending start order
