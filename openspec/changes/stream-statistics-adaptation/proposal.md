# Proposal

## Why

Phase 4 made receiver-noise centring the paper's strongest result:
- +9 to +17 pp on CLEAN, interference, AWGN and ADC, with no gradients and no labels;
- the remaining failures are strong spectral tilt and gain steps.

An explore spike on 2026-10-19 (centred source, 25 models, the Phase 3 splits) then tested the planned OG-TCST self-training. It showed which mechanisms actually move accuracy:

| target | centred source | vote pseudo-label precision (naive → vote) | head-only update | BN-train update (vote-PL / PL / TENT) |
|---|---|---|---|---|
| BOTH | 0.585 | 0.65 → 0.87 | 0.0 pp | +1.9 / +0.5 / −1.2 |
| WIFI | 0.703 | 0.78 → 0.90 | +1.1 | +2.3 / +0.4 / −1.5 |
| awgn 3 | 0.548 | 0.66 → 0.75 | −0.1 | +15.4 / +15.6 / +16.8 |
| tilt 4 | 0.396 | 0.46 → 0.51 | −0.3 | +24.3 / +24.2 / +30.3 |
| gain 3 | 0.114 | 0.18 → 0.18 | −0.6 | +16.3 / +18.3 / +25.8 |

What the spike shows:
- **Head-only updates do nothing**, even with 87–90% correct pseudo-labels.
- **The large shift-B gains come from re-estimating BatchNorm statistics.** E3's AdaBN, which needs no labels and no gradients, already gave +27 on tilt 4 and +28 on gain 3 on this source.
- **Temporal voting fixes labels only where errors are scattered in time** (interference). It doesn't help where a whole recording is consistently wrong (receiver impairments).
- **Voting alone lifts decisions a lot:** majority over W = 25 takes BOTH from 0.587 to 0.711, with no updates.

A second spike (seed 0, BN refresh from **one recording**, i.e. one drone) exposed a trap for rung 3a: a full refresh from a single-drone buffer collapses (BOTH / awgn 3 / tilt 4: 0.16 / 0.14 / 0.14, against source 0.55 / 0.50 / 0.39). A refresh from a multi-drone buffer gives the large gains (0.57 / 0.68 / 0.66). Refreshing only the convolutional BN layers, blended 50/50 with source statistics, is safe but small (0.55 / 0.61 / 0.37). So the refresh must see several emitters, or be restricted.

So the paper becomes an **energy–accuracy ladder**:

```
rung  mechanism                                   gradients  pays on
1     input centring on the noise mean (3.2)      none       CLEAN, interference, AWGN, ADC   (done, Phase 4)
2     temporal vote over W                        none       interference decisions
3a    event-triggered BN-statistics refresh       none       tilt, gain steps, AWGN
3b    vote-PL fine-tune (variant)                 backward   ~+2 pp on interference
4     open-set energy gate                        none       unknown drones (E5-lite)
```

The adaptive core of 3.3 changes from "head-only self-training" to "event-triggered statistics refresh". Self-training (3b) is kept as a variant whose energy cost has to be justified.

**Research questions, reshuffled (author approval 2026-10-19):**
- **RQ1:** how much do interference and receiver shifts cost, and how much of that is the receiver shortcut? (E1, E2; done)
- **RQ2:** how much does zero-gradient, label-free, receiver-referenced normalisation recover? (Phase 4; done)
- **RQ3:** what does event-triggered label-free adaptation add on top, and at what cost?
- **RQ4:** can an open-set gate reject unknown drones without hurting known ones?

This change serves **RQ3 and RQ4**, experiments **E3 (streaming re-run) and E5-lite**, and records the forward/backward-pass and parameter counts that E9 turns into joules. Results feed **thesis chapter 3, section 3.3**; the cost accounting feeds **section 5.4**.

## What Changes

- **Stream harness.** A stream is one **receiver condition**, built from:
  - a clean prefix: the model's CLEAN test recordings;
  - then the target segment: the target's test-half recordings (from the E3 splits), chained in a fixed random order, so several drones appear in turn.

  Within each recording, cached windows are time-ordered, with active and noise windows interleaved. Centring uses the Phase 4 stream estimator (β = 0.01, k = 50), carried across recordings within the stream. The prefix → target boundary is a known change point for scoring the trigger. Decisions are per window and per temporal vote over W windows. The same harness runs every method, so all of them get the same voting.
- **Event trigger.** A drift detector on cheap stream signals fires a statistics refresh only when the receiver/air state appears to change. Signals: a shift of the energy score relative to a reference window, and/or vote disagreement. A stable stream costs nothing extra.
- **BN-statistics refresh (rung 3a, the main adaptive step).** On a trigger, BatchNorm running statistics are re-estimated from a buffer of recent windows that passed the gate. No labels, no gradients, one forward pass per buffered window. This is the AdaBN pass of `rfml_uav.adapt.baselines`, made event-triggered and buffer-bounded. **Class-diversity guard:**
  - a full refresh runs only if the buffer's vote labels cover at least D classes;
  - otherwise a single-drone-safe refresh runs instead: convolutional BN layers only, blended with the source statistics.
- **Single-drone check.** The same methods run on per-recording streams (one drone each), to show that the guard prevents the collapse.
- **Open-set energy gate (rung 4).** The energy score is compared against an adaptive threshold (a quantile over a sliding window). Windows flagged "unknown" are excluded from the refresh buffer and from pseudo-labels, and reported as "unidentified emitter".
- **Vote-PL variant (rung 3b).** On a trigger, the whole network is fine-tuned on vote-majority pseudo-labels from gated windows: block majority over W with agreement ≥ ρ. Reported with its backward-pass cost, as an optional rung.
- **Streaming E3.** Source-only, AdaBN, TENT and naive PL, re-run in the stream harness with the same voting and trigger budget, plus rungs 3a and 3b. All on the centred source, over the shift-A and shift-B targets (with `agc_3`).
- **E5-lite.** Leave-one-class-out source models on the centred features (one class held out per model). The gate's known-vs-unknown AUROC and the false-accept rate of the held-out class are measured, gate on vs off, alongside known-class macro-F1.
- **Cost ledger.** For every rung and method: forward and backward passes per window, parameters touched per event, number of trigger events per stream, and MACs per window for WPD features and the CNN. These are the inputs E9 turns into joules.
- **Hyperparameters.** W, ρ, the gate quantile and the trigger threshold are tuned on the Phase 3 dev shifts (seed-0 models) with the same frozen-before-targets rule as E3.

### Out of scope for this iteration (alternatives noted for later iterations)
- **Head-only updates**: refuted by the spike. They stay as one ablation row only.
- **EMA teacher, class balancing, BN + head self-training variants** (from the original OG-TCST plan). Deferred until rung 3b shows a need.
- **Gradient-based TTA beyond TENT** (EATA, SAR, SHOT).
- **Divide-by-σ normalisation**: an ablation already (Phase 4).
- **Adapting shift-B targets on impaired training recordings**, instead of the 7-recording test-half adapt sets.
- **Online noise/active labelling via the paper-2 cascade**; windows keep their offline labels.
- **Streams that carry state across receiver conditions** (one long deployment stream spanning conditions), instead of one stream per condition.
- **Per-recording streams as the main protocol** (the original proposal wording). Kept as the single-drone check only, because single-drone buffers collapse a full BN refresh.
- **Other single-drone-safe refreshes** (a running class-balanced buffer, per-class statistics) beyond the conv-only blend.
- **Combined shift A + B, real cross-receiver data (E8), DroneRF.**
- **Method naming.** "OG-TCST" over-claims self-training. A rename is left to the author.
- **Energy measurement on hardware (E9).** It needs the RPi 5 / power meter; this change only produces the cost ledger.

## Capabilities

### New Capabilities
- `stream-adaptation`: the stream harness, temporal vote, drift trigger, buffer-bounded BN-statistics refresh, open-set energy gate with adaptive threshold, and the vote-PL variant. Each mechanism must be label-free and report its forward/backward cost.
- `stream-adaptation-evaluation`: the streaming E3 protocol (methods, targets, shared voting, dev-shift tuning), E5-lite (leave-one-class-out, gate on/off), the cost ledger, and the RQ3/RQ4 tables.

### Modified Capabilities
None. The centring estimators, the baselines and the Phase 3 splits are reused unchanged.

## Impact

- **Code:**
  - new `src/rfml_uav/adapt/stream.py`, for the harness, trigger, refresh, gate and vote-PL; the OG-TCST stream harness that `CLAUDE.md` already anticipates;
  - reuses `adapt/baselines.py`, `data/normalise.py` and `eval/stats.py`.
- **Notebooks:** new `7_Stream_Adaptation.ipynb` (streaming E3 + cost ledger) and `8_Open_Set_Gate.ipynb` (E5-lite).
- **Data:**
  - reads the Phase 1/2/4 features, the noisenorm checkpoints and the Phase 3 splits;
  - writes `results/paper3/phase5/`, plus `checkpoints/paper3/loco/` for the leave-one-class-out models (7 classes × 5 folds, seed 0).
- **Compute:**
  - streaming E3 ≈ 1 h GPU (25 models × 9 targets × 6 methods, forward-dominated);
  - E5-lite training ≈ 1 h.
- **Paper:**
  - Methods: 3.2 and 3.3 rewritten as the ladder;
  - RQs reshuffled as above;
  - "OG-TCST" self-training becomes a variant;
  - the spike results motivate the design and are re-measured in this change, not cited from the spike.
