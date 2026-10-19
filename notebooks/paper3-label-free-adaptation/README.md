# Paper 3 — Energy-Efficient Identification of UAVs from RF Emissions with Label-Free Adaptation

Dissertation chapter 3. Target: MDPI *Energies*, section F4 (Critical Energy Infrastructure). Notebooks run in numeric order.

## Phase 0 — DroneRF source baseline (`0_Source_Baseline.ipynb`)

**Question:** how much of the inherited ~97% WPD-CNN accuracy (MoMLeT-2025) is chunk-level leakage?

**Protocol**
- Features: `Wavelets/features.npz` (WPD, 256-d, L ‖ H) with per-row provenance `(bui, n, chunk_idx, start, is_dup)`.
- `is_dup`: window overlaps the last kept window of the same recording by >50% (`rfml_uav.data.chunking.flag_duplicates`).
- Balancing as in paper 1: every class undersampled to the smallest, before splitting.
  - all windows: 102500 per class;
  - dedup: 13894 per class.
- Splits (`rfml_uav.eval.splits.make_splits`): 5 outer test folds; validation carved from each train part.
  - `chunk` = legacy paper-1 split (shares recordings across sets);
  - `recording` = grouped by `(bui, n)`.
- CNN via `get_model("CNN")`, now returning logits. Adam with lr 0.01, batch 256, 50 epochs.
- Checkpoint = epoch with best **validation** macro-F1. Test folds are never used for selection.
- 5 seeds × 5 folds per cell (100 runs, ~63 min on an RTX 5070 Ti).

**Near-duplicates.** 86–92% of drone windows per BUI are flagged `is_dup`; Background has 0%.
- In `10000L_0`, the median gap between window starts is 182 samples, so neighbouring windows overlap ~95%.
- Cause: `get_chunk_ids` in `1_Data_Processing_Chunking.ipynb` runs with its default `min_packet_length=40`; the notebook's `chunk_size/2` value is never passed in.

**Results** (`results/paper3/phase0/leakage_table.csv`, mean ± std over 25 runs)

| split | dedup | accuracy | macro-F1 | Bebop F1 | AR F1 |
|---|---|---|---|---|---|
| chunk | off | 0.9728 ± 0.0007 | 0.9728 ± 0.0007 | 0.9466 ± 0.0014 | 0.9489 ± 0.0013 |
| chunk | on | 0.9711 ± 0.0016 | 0.9710 ± 0.0016 | 0.9476 ± 0.0033 | 0.9467 ± 0.0031 |
| recording | off | 0.9627 ± 0.0073 | 0.9626 ± 0.0076 | 0.9305 ± 0.0119 | 0.9286 ± 0.0165 |
| recording | on | 0.9651 ± 0.0062 | 0.9650 ± 0.0064 | 0.9385 ± 0.0096 | 0.9357 ± 0.0118 |

Background F1 is ≥ 0.986 and Phantom F1 ≥ 0.9998 in every cell. The errors are almost entirely Bebop ↔ AR
(both Parrot, Wi-Fi). Figures: `confusion_matrices.png`, `confidence_histograms.png`.

**Gate outcome.**
- The chunk split reproduces paper 1 (97.3%).
- The recording split costs about 1 pp of accuracy (96.3–96.5%) and adds 3–5× more fold-to-fold variance.
- Removing near-duplicates barely matters.

So leakage inflates the inherited number only slightly. The claim is safe to keep, as long as it is reported with the recording split.

**Significance** (`leakage_stats.csv`; chunk − recording, paired by (seed, fold), 25 pairs, Wilcoxon + 95% bootstrap CI)

| dedup | Δ accuracy [95% CI] | p | Δ macro-F1 [95% CI] | p |
|---|---|---|---|---|
| off | 0.0100 [0.0074, 0.0128] | 1.2e-7 | 0.0101 [0.0073, 0.0130] | 1.2e-7 |
| on | 0.0060 [0.0034, 0.0087] | 3.8e-4 | 0.0061 [0.0034, 0.0088] | 4.3e-4 |

The gap is small but significant: chunk-split numbers are optimistic by 0.6–1.0 pp.

**Energy-score separability** (recording, dedup on, 25 saved checkpoints re-scored; `energy_auroc.csv`, `energy_score_histograms.png`)
- AUROC for separating correct from wrong predictions: energy 0.930 ± 0.039, max-softmax 0.957 ± 0.027.
- **Energy is less separable than max-softmax.** By the design rule, revisit the `bn7 → prelu4` logit tail in Phase 1 before building the open-set gate on the energy score.
- The histogram still shows a clear shift (correct ≈ −5, wrong ≈ −2). Correct-vs-wrong is only a proxy: the gate's real test is known vs unknown drones (E5-lite).

**Models**
- `checkpoints/v2/checkpoint_wpd_best.*`: the MoMLeT-2025 model. **Kept as is** and still the deployed / reference model.
- `checkpoints/paper3/source_wpd_cnn.{pt,onnx}`: recording split, dedup on, seed 0. Fold 1 was chosen by val macro-F1 = 0.9818 (its test acc = 0.9727). It is available as the leakage-free alternative for later phases.

Exported with `python -m rfml_uav.utils.export_onnx --ckpt … --onnx …` (softmax in graph, output `probs`).

## Phase 1 — DroneDetect source baseline and interference sensitivity (RQ1, E1)

Notebooks: `1_DroneDetect_Exploration.ipynb` → `2_DroneDetect_Features.ipynb` → `3_DroneDetect_Source_Baseline.ipynb`.

**Protocol**
- **Data:** DroneDetect V2 (`DRONEDETECT_PATH`): 7 drones, 60 MS/s complex I/Q, subsets CLEAN / BLUE / WIFI / BOTH, modes ON / HO / FY, 390 recordings of 2 s.
- **Task:** identification only (7 classes). DroneDetect has no drone-free recordings, so detection is left to the paper-2 cascade upstream.
- **Windows** (`rfml_uav.drone_detect.dataset.select_windows`): non-overlapping 4000-sample windows (67 µs), scored by DC-removed power.
  - **Active:** ≥ 6 dB above the recording's 10th-percentile floor.
  - **Noise-only:** ≤ the floor.
  - Up to 1000 of each per recording, spread over time: 390k active + 390k noise rows in total.
- **Features** (`rfml_uav.data.wavelets.iq_wpd_features`): DC removal, unit power, then the DroneRF WPD on I ‖ Q (256-d).
- **Source model:** trained on CLEAN active rows only, every class undersampled to the smallest.
  - Split by recording (`.dat` file) with `make_splits(..., "recording")`.
  - CNN, Adam lr 0.01, 50 epochs; checkpoint = best **validation** macro-F1.
  - 5 seeds × 5 folds.
- **Scoring:** each model is scored on its CLEAN test recordings and on **all** BLUE / WIFI / BOTH recordings.
- **Exclusion:** Phantom 4 FY has no CLEAN recordings, so it is kept out of the table and reported separately.

**Interference table** (`results/paper3/phase1/interference_table.csv`; linear tail, mean ± std over 25 runs; drop = CLEAN − subset macro-F1 of the same model, paired Wilcoxon + 95% bootstrap CI)

| subset | accuracy | macro-F1 | drop [95% CI] | p |
|---|---|---|---|---|
| CLEAN | 0.7310 ± 0.0463 | 0.7274 ± 0.0479 | — | — |
| BLUE | 0.5727 ± 0.0105 | 0.5829 ± 0.0105 | 0.145 [0.127, 0.162] | 6.0e-8 |
| WIFI | 0.5834 ± 0.0100 | 0.5876 ± 0.0122 | 0.140 [0.123, 0.157] | 6.0e-8 |
| BOTH | 0.4657 ± 0.0061 | 0.4618 ± 0.0076 | 0.266 [0.248, 0.283] | 6.0e-8 |

(p = 6.0e-8 is the smallest value an exact two-sided Wilcoxon test can give for 25 pairs: every run drops.)
- **Drop:** interference costs 14–15 pp of macro-F1 with Bluetooth or Wi-Fi alone, and 27 pp with both.
- **By mode** (`interference_by_mode.csv`): ON is the most robust (CLEAN 0.85 → BOTH 0.67); HO and FY fall to 0.36–0.38 under BOTH.
- **Unseen Phantom 4 FY:** accuracy 0.58 ± 0.04.
- **Errors under interference** (`confusion_linear.png`): they collapse onto group-A classes, PHA under BLUE and DIS under WIFI. In CLEAN, most errors are AIR ↔ MP1 ↔ MP2 (DJI OcuSync family).

**Logit tail** (`energy_auroc.csv`; AUROC of separating correct from wrong predictions)

| tail | CLEAN energy / MSP | pooled interference energy / MSP | CLEAN macro-F1 |
|---|---|---|---|
| default (`bn7 → prelu4`) | 0.768 / 0.878 | 0.723 / 0.746 | 0.733 |
| linear | 0.875 / 0.882 | 0.758 / 0.749 | 0.727 |

**Chosen: linear tail.** Its energy score now matches or beats max-softmax under interference, which settles the Phase 0 concern. It costs 0.6 pp of CLEAN macro-F1, within the 1 pp rule.

**Receiver shortcut** (`shortcut_test.csv`, `shortcut_test_inband.csv`, `inband_vs_fullband.csv`)
- **Full band:** a CNN trained on noise-only CLEAN windows identifies the drone at **0.43 ± 0.05**, against chance 0.14 and the 0.286 flag. It identifies the receiver group (AIR/DIS/PHA vs INS/MIN/MP1/MP2) at **0.98**. **Shortcut flagged.**
- **In-band mitigation** (only nodes ≤ 14 MHz, 120-d): noise-only accuracy 0.36, group accuracy 0.96, so the shortcut is still flagged. Macro-F1 also dropped on every subset (CLEAN 0.702, BOTH 0.414).
- **Decision (author, 2026-10-15):** keep the full-band model. Report the shortcut as a DroneDetect property. Adaptive feature normalisation (thesis 3.2) is the intended remedy, and it gets tested against this baseline.
- **Caveat:** for nearly continuous transmitters, the "noise-only" windows of some drones may still hold weak signal (AIR noise recall 0.71). Part of the 0.43 may therefore be signal, not floor.

**Model:** `checkpoints/paper3/dronedetect_source_wpd_cnn.{pt,onnx}` (7 classes, linear tail, 256-d). Chosen as seed 0, fold 4 by val macro-F1 = 0.790; its test macro-F1 is 0.673. This is the E9 energy model.

**Limitations to state in the paper**
- No Background class.
- Shift A is interference **plus** a session/gain change, not interference alone.
- The 5 recordings per condition likely come from one session, so the recording split is optimistic.
- The two receiver configurations act as a shortcut.
- 67 µs windows are much shorter than the 20 ms segments used in prior DroneDetect work. Absolute accuracy is not comparable to those papers, but the drop under interference is.

## Phase 2 — Receiver-impairment shift (shift B, E2)

Notebook: `4_Receiver_Impairment_Shift.ipynb`. Results in `results/paper3/phase2/`.

**Protocol**
- **Impairments** (`rfml_uav.data.impairments.impair`): applied per raw window, before the unchanged WPD features. Each random draw is seeded by (impairment, severity, row), so results don't depend on processing order or the number of workers. Severity 0 reproduces the cached features (checked).

  | impairment | severity 1 | severity 2 | severity 3 | severity 4 (escalation) |
  |---|---|---|---|---|
  | awgn: SNR vs window power | 20 dB | 10 dB | 5 dB | 0 dB |
  | gain: ±offset, then ±step at a random point in the middle half of the window | 3 dB | 6 dB | 12 dB | 20 dB |
  | tilt: linear-in-dB spectral slope, edge to edge | 3 dB | 6 dB | 12 dB | 20 dB |
  | adc: bits / clip at k × component RMS | 10 / 4 | 8 / 3 | 6 / 2 | 4 / 1.5 |

- **Rows:** all 95,000 CLEAN active DroneDetect windows, and all 164,264 non-duplicate DroneRF windows (L and H impaired independently).
- **Models (no retraining):**
  - the 25 Phase 1 DroneDetect models (linear tail);
  - the 25 Phase 0 DroneRF `recording_dedup` models.

  Each model is scored on the impaired version of its own test rows. Clean macro-F1 is reproduced for all 50 runs.
- **Statistics:** drop = clean − impaired macro-F1 of the same model, paired over 25 (seed, fold) runs, with a Wilcoxon test and 95% bootstrap CI (`rfml_uav.eval.stats.paired_test`).
- **Escalation:** a severity-3 drop below 3 pp on DroneDetect adds severity 4 for both datasets. Only `tilt` qualified.

**Shift-B table** (`shift_b_table.csv`; macro-F1 mean ± std over 25 runs; drop in pp [95% CI]; every drop p < 0.01 except DroneRF adc 1, p = 0.17)

| impairment | DroneDetect s1 | s2 | s3 | DroneRF s1 | s2 | s3 |
|---|---|---|---|---|---|---|
| clean | 0.727 ± 0.048 | | | 0.965 ± 0.006 | | |
| awgn | −6.2 [4.6, 7.8] | −18.5 [16.1, 20.8] | −31.8 [28.8, 34.8] | −0.3 [0.2, 0.4] | −2.2 [1.7, 2.7] | −7.9 [6.4, 9.6] |
| gain | −0.7 [0.4, 1.1] | −2.2 [1.5, 2.9] | −5.5 [4.6, 6.6] | −4.2 [3.4, 5.2] | −18.7 [17.7, 19.8] | −39.8 [38.8, 40.7] |
| tilt | −0.3 [0.1, 0.5] | −0.9 [0.3, 1.5] | −2.8 [1.8, 3.8] | −0.1 [0.1, 0.1] | −0.4 [0.3, 0.5] | −1.4 [1.0, 1.7] |
| adc | −0.1 [0.1, 0.2] | −0.5 [0.3, 0.8] | −4.1 [2.9, 5.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | −0.3 [0.2, 0.3] |
| tilt s4 (escalation) | | | −6.9 [5.5, 8.3] | | | −4.6 [3.7, 5.5] |

- **DroneDetect: AWGN dominates.** Even 20 dB SNR costs 6 pp. At 10 dB, the 18.5 pp loss exceeds Bluetooth or Wi-Fi interference (14–15 pp, Phase 1). At 5 dB, the 31.8 pp loss exceeds BOTH (26.6 pp). The worst-hit classes are INS, MP1 and PHA (`shift_b_per_class.csv`). Gain, tilt and ADC cost ≤ 6 pp even at severity 3.
- **DroneRF: gain dominates.** These WPD features are not power-normalised.
  - A constant offset alone costs 2.8 / 13.4 / 46.4 pp at 3 / 6 / 12 dB (`gain_invariance.csv`).
  - On DroneDetect, the same offset leaves the features unchanged (checked on 1,000 windows).
  - ADC barely matters on DroneRF, because its samples are already small integers.
- **Energy-score separability** (`shift_b_auroc.csv`; AUROC of separating correct from wrong predictions on DroneDetect):
  - it degrades with severity, mostly under AWGN: energy 0.875 (clean) → 0.861 / 0.771 / 0.712;
  - energy beats max-softmax only at AWGN severities 2–3 (0.771 vs 0.767, 0.712 vs 0.700). Elsewhere max-softmax is ahead by ≤ 0.01.
- Figure: `shift_b_drop.png`.

**Receiver shortcut under AWGN** (`shortcut_inband.csv`). The 25 Phase 1 in-band models (nodes ≤ 14 MHz, 120-d, clean macro-F1 0.702) were scored on the same impaired sets. Drop in pp; difference = full-band drop − in-band drop [95% CI].

| AWGN | full-band | in-band | difference | rel. feature shift, in-band / out-of-band nodes |
|---|---|---|---|---|
| 20 dB | −6.2 | −1.6 | 4.6 [2.9, 6.2] | 0.04 / 0.35 |
| 10 dB | −18.5 | −10.1 | 8.4 [6.2, 10.2] | 0.17 / 1.40 |
| 5 dB | −31.8 | −18.7 | 13.2 [10.0, 16.1] | 0.34 / 2.50 |

- **40–75% of the AWGN loss is the receiver shortcut breaking.** Noise fills the quiet out-of-band nodes, which move about 8× more than in-band nodes. The rest (10–19 pp) is genuine SNR loss.
- **Gain is the opposite:** the in-band model is *less* robust (−4.9 / −8.8 / −12.3 vs −0.7 / −2.2 / −5.5 pp). Not yet explained.
- ADC 3 hurts the full-band model more (2.7 pp [1.5, 4.1]). Tilt shows no significant difference.

**Limitations to state in the paper**
- Shift B is simulated per window, not a real second receiver (E8 is out of scope).
- One fixed impairment draw: the CI covers model variance, not draw variance.
- ADC full scale follows the window RMS (per-window AGC), so ADC and gain are measured separately, not jointly.
- CFO, IQ imbalance and slow gain drift across windows are not modelled.
