# receiver-noise-normalisation Specification

## Purpose

Centres WPD feature vectors on the mean of the receiver's noise-only windows (thesis 3.2). The classifier then sees features relative to the current receiver and air floor, not absolute ones. The same transform is used at training and at test time. Centring only: dividing by the noise standard deviation was tried and rejected (author decision 2026-10-19, see design).

## Requirements

### Requirement: Noise mean
The normaliser SHALL estimate a per-feature mean from noise-only windows only. Active windows SHALL never contribute to it.

#### Scenario: Active windows ignored
- **WHEN** the mean is estimated from a mix of noise and active windows
- **THEN** it equals the mean of the noise windows alone

### Requirement: Per-feature centring
Normalised features SHALL be f − μ per feature, with μ the noise mean of the same receiver session (session estimator) or the current stream state (stream estimator). No scaling SHALL be applied.

#### Scenario: Additive receiver change cancels
- **WHEN** every window of a session, noise and active alike, undergoes the same per-feature additive change f + b
- **THEN** the session-normalised features are unchanged within float32 tolerance

### Requirement: Session estimator
The session estimator SHALL use all noise-only windows of one recording as the mean for every window of that recording.

#### Scenario: Normalised noise is centred
- **WHEN** a recording's noise windows are normalised with its session mean
- **THEN** each feature has mean 0 over those windows, within 1e-4

### Requirement: Stream estimator
The stream estimator SHALL process windows in time order. It SHALL update μ by an exponential moving average with rate β on noise windows only, and normalise each window with the state *before* that window's own update. Its initial state SHALL be the mean of the first K noise windows of the stream (warm-up), and no window SHALL be normalised before warm-up completes.

#### Scenario: Active windows do not move the state
- **WHEN** a run of active windows passes through the stream
- **THEN** μ is unchanged between the noise windows before and after it

#### Scenario: Converges to the session mean
- **WHEN** a stationary stream of noise windows is long relative to 1/β
- **THEN** the stream state approaches the session mean within 5% per feature

#### Scenario: No look-ahead
- **WHEN** a window is normalised
- **THEN** its output depends only on that window and earlier windows

#### Scenario: Warm-up
- **WHEN** fewer than K noise windows have arrived
- **THEN** the windows seen so far are reported as not yet normalisable, and are excluded from scoring
