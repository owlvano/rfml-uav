# Proposal

## Why

Phase 5 (frozen protocol) left **when to adapt** as the open problem:
- Always-on BatchNorm-statistics adaptation (continuous AdaBN, TENT, PL) gains about +20 pp on gain 3 and tilt 4 and +8.5 pp on awgn 3.
- But it costs 7–11 pp on the clean prefix, on interference and on mild receiver shift.
- The energy-triggered refresh is safe but rarely fires (θ = 4 detects 0–52% of shifts).

Two mechanisms explain most of that:
1. Streams show one drone at a time, so per-chunk statistics track the current drone, not the receiver.
2. The energy score is too insensitive to tell receiver changes apart.

This change tests the fixes listed in NOTES "Phase 5 outcome". It is a **post-hoc follow-up, disclosed as such**: every target number is reported next to the frozen Phase 5 numbers, never in their place.

Discovery done before this proposal (2026-10-26, read-only or dev cells only):
- **Noise-floor drift** (relative L1 distance between the stream estimator's running noise mean and its warm-up mean) is **bimodal on the clean prefix**: about 0.04 or 0.4–0.56. The two DroneDetect receiver groups have different floors, so chaining recordings moves it.
  - On the targets it rises clearly only for gain (0.8–0.9) and tilt 4 (median 0.61).
  - It barely moves for awgn 3 and adc 3 (≈ 0.05).
  - A noise-floor gate can therefore tell severe gain/tilt shifts apart, but not AWGN, and it fires on receiver-group switches.
- The spread of centred noise windows doesn't separate any target except gain. It is dropped.
- **Drone-balanced statistics (dev cells, seed 0):** continuous AdaBN fed from a per-vote-class buffer instead of the current chunk cuts the clean-prefix harm from −10.1 to −1.9 pp (Bc = 200, D = 3) and no longer harms awgn 1 / adc 2. It gains less on gain 2 (+6.6 vs +26.9 pp).

This change serves **RQ3** (what event-triggered label-free adaptation adds, and at what cost), experiment **E3 (streaming)**. Results feed **thesis chapter 3, section 3.3**; the pass counts feed **section 5.4**.

## What Changes

- **Noise-floor drift signal.** The stream centring estimator also reports, per window, the relative L1 drift of its running noise mean from its warm-up mean. This is free: the state already exists.
- **Receiver gate with hysteresis.**
  - Adaptation switches ON when the drift exceeds φ_on.
  - It switches OFF when the drift falls below h·φ_on (h = 0.5); OFF restores the source BatchNorm statistics.
- **Drone-balanced statistics buffer.** A FIFO of the last Bc accepted windows *per vote class*. A statistics update runs only when ≥ D classes hold ≥ Bc/2 windows, so single-drone stretches can't drive the BN statistics.
- **New stream methods** in the existing runner:
  - `balanced`: continuous AdaBN on the balanced buffer, always on;
  - `gated_continuous`: Phase 5 continuous AdaBN, only while the receiver gate is ON;
  - `gated_balanced`: balanced continuous AdaBN, only while the gate is ON.
- **Oracle diagnostic.** `gated_continuous` with the gate forced ON exactly at the true change point: an upper bound on what any gate could deliver. It is a diagnostic that reads the protocol's change point, never a method.
- **Harm-aware dev tuning** on the Phase 5 dev cells (seed-0 models):
  - objective = target vote F1 − max(0, prefix harm vs source-only);
  - saved before any target runs.
- **Targets run once:** 9 targets × 25 models × 4 runs. Reported next to the frozen Phase 5 source, continuous, TENT and triggered numbers on identical streams.

### Out of scope for this iteration (alternatives noted for later iterations)
- **Noise-floor jump detectors** (drift against a recent past instead of the warm-up). They would also fire on every receiver-group switch.
- **Per-receiver-group drift references.** The groups correlate with drone identity, so this would leak labels.
- **A combined gate** (noise-floor OR energy), and a **vote-disagreement trigger**.
- **Fast/slow switching** (balanced always on, fast unbalanced updates while gated ON).
- **Single-drone per-recording streams** for the new methods (the Phase 5 check).
- **E5-lite with the new methods; gradient methods** (TENT/PL) gated or balanced.
- **Better open-set scores** (Mahalanobis, temperature-scaled energy).

## Capabilities

### New Capabilities
- `gated-adaptation-evaluation`: the Phase 6 protocol. The disclosed post-hoc status, harm-aware dev tuning frozen before targets, the comparison with the frozen Phase 5 rows on identical streams, gate diagnostics (ON time in prefix and target, delay) and the oracle bound.

### Modified Capabilities
- `stream-adaptation`: adds a receiver gate (noise-floor drift input with hysteresis, reverting to source statistics), a drone-balanced statistics buffer, and the methods that use them. The runner takes the drift signal as an input; it still takes no labels.
- `receiver-noise-normalisation`: the stream estimator can report the per-window noise-floor drift from its warm-up state.

## Impact

- **Code:**
  - `src/rfml_uav/data/normalise.py`: optional drift output of `stream_normalise`;
  - `src/rfml_uav/adapt/stream.py`: drift argument, gate, balanced buffer, 3 methods, self-check;
  - `CLAUDE.md`: package notes.
- **Notebook:** `notebooks/paper3-label-free-adaptation/9_Gated_Adaptation.ipynb`. It rebuilds the Phase 5 streams with the same code, with drift added.
- **Data:** reads Phase 3/4/5 inputs and `phase5/streams/`; writes `results/paper3/phase6/`.
- **Compute:** about 1 h (forward-only methods).
- **Paper:** if it works, 3.3 becomes "receiver-gated, drone-balanced statistics adaptation", with the frozen Phase 5 trade-off as motivation. If not, it is a disclosed negative result.
