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

## Next session

1. Archive `dronedetect-source-baseline` after review.
2. Next change: shift B, a receiver-impairment simulator (4 impairments × 3 severities: AWGN, gain/AGC drift, frequency-response tilt, ADC bits + clipping) on cached features from both datasets (E2).
   - Plan go/no-go: if source-only loses < 3 pp, raise the severity.
3. Then the adaptation baselines (AdaBN, TENT, naive pseudo-labelling) against the DroneDetect source model (E3).
4. Hardware check for the Nov 15 go/no-go: order or borrow the Jetson and power meter.
