# Paper 3 — Working Notes (2026-10-13)

Findings and open threads after Phase 0. Results are in `README.md`; this file is for interpretation and next steps.

## Honest assessment after Phase 0

- **Accuracy.** The WPD-CNN reaches ~96.5% on DroneRF with the recording split, vs 97.3% with the chunk split. The MoMLeT claim survives if it is reported with the recording split.
- **Recording split may still be optimistic (unverified).** DroneRF segments `n` are likely consecutive slices of one capture session. If so, the "recording split" is really a segment split, and train/test still share room, day and receiver state.
  - TODO: confirm against Allahham et al. 2019 / the dataset docs.
  - If confirmed, list it as a limitation.
- **Background is a quiet lab.** In real air, Bebop/AR (both Wi-Fi) will likely be confused with any Wi-Fi device.
- **Errors are almost all Bebop ↔ AR** (both Parrot, Wi-Fi). Phantom F1 ≈ 1.0 (separate band/protocol).
- **Not field-usable yet.** Reasons:
  - DroneRF has two real-valued channels (low/high band), but an SDR delivers complex I/Q;
  - only one dataset and receiver;
  - no "unknown" class.
- **Still useful as research groundwork.** It provides a clean source baseline, logits for the energy score, provenance for streams and group splits, and the near-duplicate finding.
- **Near-duplicate cause.** `get_chunk_ids` runs with its default `min_packet_length=40`; the notebook's `chunk_size/2` value is never passed in. As a result ~90% of drone windows overlap >50% (median start gap 182 samples). The leakage effect on accuracy is small.

## Path to field use

| Gap | Remedy | Plan phase |
|---|---|---|
| Real-valued L/H vs complex I/Q | Train on I/Q (DroneDetect, own B200mini); WPD(I) ‖ WPD(Q) keeps the 256-d input | Phase 1 |
| One session / receiver / room | Session- or day-level splits; augmentation (gain, CFO, IQ imbalance, FIR tilt, AWGN) | Phases 0, 3 |
| Quiet background | Background = real air (Wi-Fi, BT, site emitters), recorded at the deployment site | DroneDetect interference subsets, own recordings |
| Unknown drones | Open-set gate (reject instead of forcing 1 of 4) | Paper 3 core |
| New site / receiver | Label-free adaptation on 10–30 min of local air (norm stats, BN, head) | Paper 3 core (OG-TCST) |
| Wi-Fi drones (Bebop, AR, Tello) | Spectrum ML separates them poorly from other Wi-Fi. Decode the protocol instead: MAC vendor prefix (Parrot/DJI), SSID patterns, Remote ID (Wi-Fi/BLE beacons) | Monitor repo, protocol head |

ML pays off most on what can't be cheaply decoded: proprietary links (DJI OcuSync/Lightbridge), frequency-hopping RC links, analog video.

## False alarms

**Base-rate problem.** One segment = 4000 samples @ 80 MHz = 50 µs. Background recall is 0.996, i.e. 0.4% of segments are false positives. At 100 decisions/s that is ≈ 1,400 raw false alarms per hour. Per-segment accuracy is the wrong field metric; use **false alarms per hour at a fixed detection rate**.

Remedies, cheapest first:
1. **Require persistence over time.** Alarm only after K of N positive decisions within ~1–3 s. Drone links transmit continuously; isolated false segments don't repeat. (Roadmap: debounce. Paper 3: temporal consistency.)
2. **Set thresholds by false-alarm rate.** Calibrate on hours of real site background to hit, e.g., ≤ 1 false alarm per hour.
3. **Reject option.** Low confidence / high energy score → "unidentified emitter" (logged), not "drone" (alarm). Paper 3's gate.
4. **Hard-negative mining.** Add site false positives to Background and fine-tune; repeat after deployment.
5. **Whitelist known emitters.** Decode local AP beacons and ignore known SSIDs/MACs. A router sends beacons every 102.4 ms and is static.
6. **Use signal dynamics.** Track signal strength and direction over time; a drone moves, a router doesn't.
7. **Fuse sensors (later).** Confirm with a second RF sensor, radar, acoustic or camera before a high-severity alarm.

For the papers:
- **Paper 3:** points 1–4 strengthen it. Report **false alarms per hour** on DroneDetect interference-only data next to F1.
- **Monitor repo:** points 5–7 are system features.

## Decisions so far

- MoMLeT-2025 model (`checkpoints/v2/`) is **kept as is**; it stays the reference and deployed model. The leakage-free alternative is `checkpoints/paper3/source_wpd_cnn.*`.
- Published Paper 1 state is tagged `momlet-2025` (local tag, not pushed).

## DroneDetect exploration (2026-10-15, `1_DroneDetect_Exploration.ipynb`)

- **Data.** 390 files, 374 GB on external drive (`DRONEDETECT_PATH`).
  - 5 recordings of 2 s per (subset, drone, mode).
  - DIS has no hover mode; PHA has no fly mode in CLEAN or BLUE.
- **No Background class.** There are no drone-free files, so DroneDetect supports identification only.
- **Two receiver configurations.** AIR/DIS/PHA vs INS/MIN/MP1/MP2 differ in DC offset, RMS (~10×), out-of-band noise floor and file length. This is a shortcut risk.
- **Shift A is confounded with gain.** CLEAN → BOTH also shifts the noise floor (up to 20 dB for AIR/PHA).
- **Bursty signals.** Only 15–55% of windows hold a burst, so chunking must keep active windows only.

## Phase 1 outcome (2026-10-15)

See README "Phase 1".
- **Interference cost:** macro-F1 drops 14–15 pp under BT or Wi-Fi and 27 pp under both, so there is room for adaptation.
- **Logit tail:** linear, which fixes the Phase 0 energy-score issue.
- **Receiver shortcut:** confirmed (noise-only accuracy 0.43, group accuracy 0.98). In-band cropping doesn't remove it.
- **Errors under interference** flow into group-A classes, which suggests the raised noise floor is read as "group A". Adaptive normalisation should target exactly this.

## Phase 2 outcome (2026-10-19)

See README "Phase 2".
- **Shift B is strong enough to adapt on.** On DroneDetect, AWGN severity 2–3 costs 18–32 pp, as much as or more than shift A. Gain, tilt and ADC are mild (≤ 6 pp at severity 3). Tilt needed the severity-4 escalation (6.9 pp).
- **Why AWGN hurts DroneDetect so much (confirmed, README "Receiver shortcut under AWGN").** In-band models lose 40–75% less under AWGN, and out-of-band nodes shift about 8× more than in-band ones. Most of the AWGN loss is the receiver shortcut breaking. The rest is SNR loss.
  - Paper angle: adaptive normalisation should recover AWGN most. E3 tests that directly.
  - Open: in-band models are *less* robust to a gain step. Mechanism unknown; check per-node shift under `gain` before writing.
- **Normalisation is a free defence against gain.** The unit-power I/Q features ignore a constant gain change. The un-normalised DroneRF WPD loses up to 46 pp. This is a Methods point. It also suggests re-normalising DroneRF features if DroneRF stays in the adaptation experiments.
- **The energy score doesn't clearly beat max-softmax under shift B.** It wins only at strong AWGN. The open-set gate (E5-lite) should report both scores.

## Phase 3 outcome (2026-10-19)

See README "Phase 3".
- **Standard label-free adaptation fixes AWGN, not interference.**
  - AWGN: every baseline recovers 33–50% of the drop (TENT best). This confirms the prediction from the shortcut finding.
  - Shift A: AdaBN and PL are ≈ 0, and TENT is harmful (−1.4 to −2.5 pp).
  - Interference is a per-window mixture (bursts), not a global statistics shift.
- **Implication for OG-TCST.** Normalisation adaptation alone will match the baselines on AWGN at best. The novelty must come from what the baselines can't do:
  - per-window or temporal handling of interference bursts (temporal consistency, gating out interfered windows);
  - not becoming over-confident: TENT drops energy AUROC under interference, so a gate must not be fed TENT-adapted logits.
- **The bar per target is in `e3_table.csv`** (`best_baseline`). For shift A it is effectively source-only or AdaBN (≈ 0.46–0.59). For AWGN it is TENT (0.640 / 0.573).
- **Cost.** AdaBN is about 1 forward pass per window with no backward pass. TENT and PL each need about one backward pass per window. An event-triggered, head-only update (903 parameters) should undercut both.
- **Protocol issue.** One recording per class in the shift-B adapt half biases BatchNorm methods by about 2–3 pp. If shift-B comparisons matter for OG-TCST, consider adapting on impaired *train* recordings (unlabeled, but the model trained on their clean versions). That needs an author decision.
- **Bug fixed during apply.** The AdaBN statistics pass first walked recording-sorted rows in order, which underestimated the variance (clean-data loss −4.9 pp). It now shuffles, and the self-check tests the variance.

## Phase 4 outcome (2026-10-19)

See README "Phase 4".
- **3.2 holds, but as centring at train *and* test time, not std-normalisation.** Centring on the noise mean gives +9 to +17 pp on CLEAN, interference, AWGN and ADC, with no gradients and no labels: one running mean per receiver. This is now the paper's strongest and cheapest result, and it suits *Energies*.
- **Dividing by the noise σ is wrong for this data.** The noise σ reacts to receiver changes unlike active windows (a 3 dB gain step shrinks it to 0.56×). Kept as an ablation.
- **Failure modes of centring:** strong tilt (≥ 12 dB) and gain steps inside the noise windows. TENT on top of the centred source recovers tilt 4 (0.696), so the adaptation step has a real job now.
- **The shortcut is weaker but not gone** (receiver-group accuracy from noise 0.98 → 0.91). Centring removes the floor level, not its shape. A stated limitation.
- **Source for OG-TCST: centred (amended rule, +6.6 pp, p = 6e-8).** The original rule chose Phase 1 (−1.2 pp, p = 0.07, driven by the adversarial every-window gain cell). Both are disclosed.
- **Implication for OG-TCST.** Shift A gains already come from centring (BOTH 0.462 → 0.587). OG-TCST must earn its energy cost on the remaining gaps:
  - BOTH 0.59;
  - strong tilt and gain steps (where TENT helps);
  - unknown drones (the gate).
- **Story reframe (to do in the planning doc):** lead with 3.2 (cheap, large effect). OG-TCST becomes "what event-triggered updates add on top, at what energy cost". Confirm the RQ numbering.

## Phase 5 outcome (2026-10-26)

See README "Phase 5". These are frozen-protocol numbers (dev tuning saved before any target ran). By author decision on 2026-10-26 they are reported as they are; the fixes below go to a follow-up change, disclosed as post hoc.

**Which rungs earn their cost**
- **Rung 2, the vote, is the second free win after centring.**
  - +3 to +11 pp on 8 of 9 targets (all p ≤ 3e-4), with no gradients and 24 stored predictions.
  - It fails only where errors are consistent in time (gain 3, −1.8).
- **Rung 3a, the triggered refresh, is safe but inert under the frozen settings.**
  - θ = 4 detects the shift in 0–52% of streams; the gains are only gain 3 +4.4 and tilt 4 +2.4.
  - Prefix harm ≤ 0.3 pp and +0.2% MACs.
- **Always-on statistics adaptation is a trade-off, not a free gain.**
  - Continuous AdaBN = TENT = PL: about +20 pp on gain 3 / tilt 4 and +8.5 on awgn 3.
  - But −7 to −11 pp on the clean prefix, interference and mild shift B.
- **Continuous AdaBN (+30% MACs, 0 backward) matches TENT (+90%).** The backward pass buys nothing.
- **Periodic refresh is the worst method** (−14 to −25 pp). In a stream that shows one drone at a time, a 1,000-window buffer holds 2–3 drones. Vote errors fake diversity, so the D-class guard doesn't stop full refreshes.

**Is 3b worth its backward passes?** No.
- Vote-PL at the tuned lr = 1e-5 is within 0.4 pp of the triggered refresh, except gain 3 (+1.6 pp, from a 0.148 base). Head-only equals vote-PL.
- Self-training adds nothing measurable in streams.

**RQ3 answer.** Event-triggered, label-free adaptation adds little on top of centring + vote at the frozen settings. The large gains on severe receiver shift exist (always-on BN statistics), but they cost ~9 pp wherever the receiver is fine. The open problem is **when** to adapt, not how.

**RQ4 answer.** No.
- The energy gate does not separate unknown drones: AUROC 0.64 on CLEAN, 0.55 on BOTH, below chance for 3 of 7 classes under interference. 79–83% of unknown windows are accepted.
- Its known-F1 gain (+2.6 / +2.9 pp) is a selection effect.

**Alternatives deferred to later iterations**

From the proposal:
- EMA teacher, class balancing, BN + head self-training.
- EATA / SAR / SHOT.
- Adapting shift-B targets on impaired training recordings.
- Online noise labels via the paper-2 cascade.
- Cross-condition streams.
- Per-class statistics.
- The method name.

New from Phase 5:
- **Drone-balanced buffer** (a reservoir per vote class) for the refresh: the direct fix for the periodic collapse.
- **Harm-aware dev selection:** choose on target vote F1 minus prefix harm. Stage 4 picked m = 0.1 despite −10 pp dev prefix harm.
- **Trigger tuned for detection rate** (with severe dev shifts, or a detection-rate target), or a vote-disagreement / noise-floor trigger (rung 1's μ already moves on receiver change: a cheaper, label-free drift signal).
- **Continuous AdaBN gated by that trigger:** on only after a detected receiver change, back to source on recovery.
- **A better open-set score** (Mahalanobis on penultimate features, or an energy score after temperature scaling), with gate thresholds per receiver condition.

## Next session

1. Review Phase 5. Archive `stream-statistics-adaptation`, commit, merge, push. Back up `data/drone-rf/results` and `checkpoints` (gitignored, now incl. `checkpoints/paper3/loco/`).
2. **Story:** lead with rungs 1 + 2 (centring + vote, both free). Present adaptation as a costed trade-off with an honest "when to adapt" gap. Decide whether the follow-up (noise-floor trigger + gated continuous AdaBN + balanced buffer) fits before 2027-01-20.
3. **E9** on the ledger: WPD 3.91 M + CNN 1.87 M MACs/window. Feature extraction dominates (68%), which matters for the energy story.
4. Light ablations (W, θ, update scope) on dev cells only.
5. **Hardware (none yet, 2026-10-19).**
   - Order an RPi 5 and a USB-C power meter with logging (or an INA226). This is the main E9 platform.
   - CloudJetson (AGX Orin 64GB, ~$1.70/h) is a secondary data point, if they confirm `tegrastats` per-rail power and sudo for `nvpmodel` / `jetson_clocks`. Report it as AGX Orin, not as an Orin Nano proxy.
   - Go/no-go Nov 15.
6. Writing: Methods 3.2/3.3 as the ladder; Results RQ1–RQ4.
