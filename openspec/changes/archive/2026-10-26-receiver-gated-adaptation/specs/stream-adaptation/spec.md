# Spec Delta

## MODIFIED Requirements

### Requirement: Prequential, label-free processing
The stream runner SHALL process windows in time order. Each window's prediction SHALL come from the model state before any update that window takes part in. No mechanism SHALL read labels.

#### Scenario: Predict before update
- **WHEN** a refresh or fine-tune is triggered at a chunk
- **THEN** the predictions reported for the windows in its buffer were made by the earlier model state

#### Scenario: No label input
- **WHEN** the runner is called
- **THEN** its inputs are features, the active/noise flags, the time order, an optional per-window noise-floor drift signal and hyperparameters, with no label argument

## ADDED Requirements

### Requirement: Receiver gate
Gated methods SHALL keep an ON/OFF state driven by the noise-floor drift of the most recent active window in each chunk. The state switches ON when the drift exceeds φ_on and OFF when it falls below h·φ_on. Statistics updates SHALL happen only while the state is ON. Switching OFF SHALL restore the source BatchNorm statistics. Without a drift signal, a gated method SHALL behave as source-only.

#### Scenario: Stable floor
- **WHEN** the drift stays below φ_on for the whole stream
- **THEN** a gated method's predictions equal source-only predictions and it reports no updates

#### Scenario: Hysteresis
- **WHEN** the drift rises above φ_on and then stays between h·φ_on and φ_on
- **THEN** the state stays ON

#### Scenario: Revert
- **WHEN** the state switches from ON to OFF
- **THEN** every BatchNorm layer holds the source statistics again, and no trainable parameter has changed

### Requirement: Drone-balanced statistics buffer
Balanced methods SHALL keep, for each vote class, the last Bc accepted windows whose vote decision is that class. A balanced statistics update SHALL use the union of the classes holding at least Bc/2 windows, and SHALL run only if at least D classes do. Windows rejected by the open-set gate SHALL enter no class buffer.

#### Scenario: Single-drone stretch
- **WHEN** fewer than D vote classes have filled half their buffers
- **THEN** no balanced update runs and the BatchNorm statistics are unchanged

#### Scenario: Per-class capacity
- **WHEN** one vote class dominates a long stretch of the stream
- **THEN** that class contributes at most Bc windows to an update

### Requirement: Gated and balanced statistics methods
The runner SHALL provide:
- balanced continuous adaptation: on every chunk where the buffer condition holds, one gradient-free train-mode pass over the balanced buffer with BN momentum m;
- gated continuous adaptation: the existing continuous method, applied only while the receiver gate is ON;
- gated balanced adaptation: balanced continuous adaptation, applied only while the receiver gate is ON.

Each SHALL report its forward passes per window, update count and gate ON fraction.

#### Scenario: Forward-only
- **WHEN** any of these methods runs
- **THEN** it reports zero backward passes and no trainable parameter changes

#### Scenario: Gate cost
- **WHEN** a gated method runs on a stream where the gate is never ON
- **THEN** it reports one forward pass per window
