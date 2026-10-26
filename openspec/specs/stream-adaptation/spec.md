# stream-adaptation Specification

## Purpose

Label-free, mostly forward-only adaptation of a trained classifier over a time-ordered window stream (thesis 3.3). It covers temporal voting, an energy-score open-set gate, drift-triggered BatchNorm-statistics refresh with a class-diversity guard, and an optional vote-pseudo-label fine-tune. Every mechanism reports its compute cost.

## Requirements

### Requirement: Prequential, label-free processing
The stream runner SHALL process windows in time order. Each window's prediction SHALL come from the model state before any update that window takes part in. No mechanism SHALL read labels.

#### Scenario: Predict before update
- **WHEN** a refresh or fine-tune is triggered at a chunk
- **THEN** the predictions reported for the windows in its buffer were made by the earlier model state

#### Scenario: No label input
- **WHEN** the runner is called
- **THEN** its inputs are features, the active/noise flags, the time order and hyperparameters, with no label argument

### Requirement: Temporal vote
For every active window, the runner SHALL report a vote decision: the majority class of the last W per-window predictions in the stream, including the current one.

#### Scenario: Vote window
- **WHEN** fewer than W active windows have been seen
- **THEN** the vote uses all predictions so far

#### Scenario: Majority
- **WHEN** the last W predictions contain one class more often than any other
- **THEN** the vote decision is that class

### Requirement: Open-set energy gate
The gate SHALL flag a window as unknown when its energy score (−logsumexp of the logits) exceeds the threshold. The threshold is the (1 − q) quantile of the source calibration energies until the first refresh event. After each refresh event it is the (1 − q) quantile of the energies of all active windows in the R chunks that follow, and the previous threshold applies until those R chunks have passed. Flagged windows SHALL be excluded from refresh buffers and pseudo-labels, and reported as unidentified.

#### Scenario: Gate excludes from updates
- **WHEN** a window is flagged unknown
- **THEN** it is in no refresh buffer and no pseudo-label set

#### Scenario: Stationary stream
- **WHEN** a stream's energies follow the calibration distribution and no refresh occurs
- **THEN** the rejected fraction of windows stays near q and does not grow over the stream

#### Scenario: Gate output reported
- **WHEN** a stream is processed
- **THEN** every active window carries a known/unknown flag and its energy score

### Requirement: Drift trigger
The trigger SHALL fire when the mean energy score of the current chunk of gated windows deviates from the reference by more than θ reference standard deviations. The reference is the source calibration statistics until the first refresh, and the post-refresh statistics after it. A cooldown of C chunks SHALL follow each firing.

#### Scenario: Stable stream
- **WHEN** a stream's energy scores stay within θ reference standard deviations
- **THEN** the trigger never fires and no adaptation cost is incurred

#### Scenario: Cooldown
- **WHEN** the trigger fires
- **THEN** it cannot fire again within the next C chunks

### Requirement: Guarded BN-statistics refresh
On a trigger, the refresh SHALL re-estimate BatchNorm running statistics from a buffer of the last B gated windows, in one forward pass per window with no gradients. It SHALL run a full refresh (all BN layers) only if the buffer's vote decisions cover at least D classes. Otherwise it SHALL run a restricted refresh: convolutional BN layers only, blended as (1 − λ)·source + λ·buffer.

#### Scenario: Diverse buffer
- **WHEN** a trigger fires and the buffer's vote decisions cover at least D classes
- **THEN** all BN layers take the buffer statistics, and no trainable parameter changes

#### Scenario: Single-drone buffer
- **WHEN** a trigger fires and the buffer covers fewer than D classes
- **THEN** only the convolutional BN layers change, to the λ-blend, and the fully connected BN layers keep their current statistics

### Requirement: Vote-pseudo-label fine-tune variant
As an optional variant, on a trigger the runner SHALL fine-tune all parameters for one pass over the gated buffer. Labels are the block majority over W windows, kept where block agreement is ≥ ρ. A guarded refresh SHALL follow the fine-tune.

#### Scenario: Pseudo-label source
- **WHEN** the fine-tune runs
- **THEN** each pseudo-label is the majority of its W-block of pre-update predictions, and blocks below agreement ρ contribute no labels

### Requirement: Cost reporting
Every run SHALL report forward and backward passes per active window, the number of trigger events, the parameters or statistics changed per event, and the number of full vs restricted refreshes.

#### Scenario: Source-only cost
- **WHEN** a stream runs without adaptation
- **THEN** it reports one forward pass per window, zero backward passes and zero events
