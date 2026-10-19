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

## Phase 3 — Adaptation baselines (E3)

Notebook: `5_Adaptation_Baselines.ipynb`. Results in `results/paper3/phase3/`.

**Protocol**
- **Methods** (`rfml_uav.adapt.baselines`), applied to the 25 frozen Phase 1 models (linear tail):
  - AdaBN: BatchNorm statistics re-estimated on the target;
  - TENT: entropy minimisation on the BatchNorm affine parameters, then the statistics re-estimated;
  - naive pseudo-labelling (PL): fine-tune all parameters on source predictions with max-softmax ≥ τ.
- **Offline:** each method adapts once on the unlabeled *adapt* half and is scored on the disjoint *test* half.
- **Split** by recording, per drone class: half of each class's recordings go to adapt, the rest to test.
  - Shift A (BLUE / WIFI / BOTH, no Phantom FY): 45 adapt / 50 test recordings.
  - Shift B: only the model's own CLEAN test recordings, impaired: 7 adapt (1 per class) / 12 test.
- **Hyperparameters:** tuned with labels on dev shifts that are not targets (awgn 1, gain 2, adc 2, tilt 3, seed-0 models), then frozen (`selected_hparams.json`):
  - TENT: lr 1e-2, 1 epoch;
  - PL: τ 0.8, lr 1e-4, 1 epoch.

  The dev gains are small (≤ 0.6 pp), so the selection favours light updates.

**E3 table** (`e3_table.csv`; macro-F1 mean ± std over 25 runs; gain vs source-only in pp [95% CI]; recovered = gain / (clean − source); **bold** = best baseline)

| target | source-only | AdaBN | TENT | PL |
|---|---|---|---|---|
| BLUE | **0.588 ± 0.022** | −0.4 [−0.8, 0.0] | −1.4 [−2.0, −0.8] | −0.6 [−0.8, −0.3] |
| WIFI | 0.585 ± 0.022 | **+0.1 [−0.3, 0.4]** | −2.5 [−3.3, −1.6] | −0.3 [−0.6, 0.0] |
| BOTH | 0.457 ± 0.016 | **+0.2 [−0.1, 0.5]** | −1.7 [−2.3, −1.1] | −0.2 [−0.4, 0.1] |
| awgn 2 | 0.550 ± 0.081 | +7.4 [4.1, 10.5], 33% | **+8.9 [5.9, 11.8], 44%** | +7.6 [5.0, 10.0], 38% |
| awgn 3 | 0.409 ± 0.069 | +13.7 [10.2, 17.2], 42% | **+16.5 [13.0, 20.0], 50%** | +13.5 [11.1, 16.0], 43% |
| gain 3 | 0.669 ± 0.076 | +0.4 [−1.3, 2.0] | −0.8 [−2.8, 1.2] | **+0.5 [−1.0, 2.0]** |
| adc 3 | **0.685 ± 0.074** | −1.0 [−2.4, 0.2] | −0.5 [−1.9, 0.8] | −0.3 [−1.3, 0.6] |
| tilt 4 | 0.657 ± 0.052 | −0.2 [−1.8, 1.5] | +0.5 [−1.2, 2.2] | **+0.7 [−0.4, 1.8]** |

- **Interference (shift A): no baseline helps.** AdaBN and PL stay within ±0.6 pp; TENT loses 1.4–2.5 pp. Interference changes only some windows (bursts), not the statistics of all of them, so re-normalising every window doesn't address it.
- **AWGN: every baseline recovers 33–50% of the drop.** TENT is best (+16.5 pp at 5 dB). This matches the Phase 2 shortcut finding: re-normalising fixes the noise-filled out-of-band nodes. Much of the gain is on DIS, INS and MP1 (`e3_per_class.csv`).
- **Mild impairments (gain 3, adc 3, tilt 4): no significant change.** Their drops are only 3–7 pp, so recovered fractions are noisy (2–5 runs per target excluded, where clean − source < 1 pp).
- **Separability after adaptation** (`e3_auroc.csv`):
  - energy beats max-softmax in most cells;
  - AdaBN raises energy AUROC on WIFI (0.717 → 0.770) and AWGN 3 (0.709 → 0.767);
  - TENT lowers it under interference (BLUE 0.831 → 0.709): entropy minimisation makes wrong predictions confident.

**Cost** (`e3_cost.csv`; per adapt window; GPU wall time is a proxy, not node energy)

| method | forward | backward | parameters updated | median wall time per adaptation |
|---|---|---|---|---|
| AdaBN | 1.00 | 0 | 0 | 0.013 s |
| TENT | 1.99 | 0.99 | 2,016 | 0.044 s |
| PL | 1.73 | 0.73 | 409,626 | 0.041 s |

**Sanity check: AdaBN on clean data** (`adabn_clean_diagnosis.csv`, seed-0 models; change in macro-F1 by where the statistics come from)

| statistics from | change |
|---|---|
| clean train set (~60 recordings) | +0.5 pp |
| all 19 clean test recordings | −0.1 pp |
| shift-B adapt half (7 recordings, 1 per class) | −3.5 pp |

Over 25 runs, AdaBN on the clean adapt half loses 2.3 pp [0.8, 4.0]. The implementation is sound; statistics from one recording per class are too recording-specific. **Shift-B results for AdaBN and TENT are therefore biased down by about 2–3 pp.**

**Limitations to state in the paper**
- The shift-B adapt sets are small (7 recordings), which biases BatchNorm-based baselines (above).
- The dev shifts are mild, so the tuning signal is weak and favours light updates. OG-TCST will be tuned with the same grid and cells.
- Offline regime only. Streaming versions come with OG-TCST.

## Phase 4 — Receiver-normalised source (thesis 3.2)

Notebook: `6_Receiver_Normalised_Source.ipynb`, plus E3 re-run via `SOURCE=noisenorm` (`5_Adaptation_Baselines_noisenorm.ipynb`). Results in `results/paper3/phase4/` and `results/paper3/phase3/noisenorm/`.

**Method** (`rfml_uav.data.normalise`)
- Every 256-d WPD vector is **centred** per feature on the mean of the noise-only windows of its own receiver session: `f − μ_noise`. The same transform is used at training and at test time.
- Two estimators:
  - **session:** all noise windows of the recording;
  - **stream:** EMA over noise windows in time order, β = 0.01, warm-up k = 50 noise windows, no look-ahead. Warm-up excludes 6.3% of active rows.
- Shift B: impaired active windows are centred on the *impaired* noise windows of the same recording (`phase2/dronedetect_noise/`).
- Source models: Phase 1 protocol with identical folds (checked against all 25 Phase 1 test sets), linear tail, 25 runs. Checkpoints in `checkpoints/paper3/noisenorm/`.
- **Rejected variant: (f − μ) / σ.** It was run first and is kept as an ablation (`phase4_std/`, `checkpoints/paper3/noisenorm_std/`). The noise σ reacts to receiver changes differently from active windows (a 3 dB gain step shrinks it to 0.56×), so the division inflates features.

**E1 — interference** (`e1_table.csv`; macro-F1 mean over 25 runs; Δ vs Phase 1 in pp, paired; all centred Δ p ≤ 1.2e-7)

| subset | Phase 1 | centred (session) | Δ | centred (stream) | Δ | std ablation Δ |
|---|---|---|---|---|---|---|
| CLEAN | 0.727 | **0.821** | +9.4 | 0.818 | +8.8 | −2.5 |
| BLUE | 0.583 | **0.695** | +11.2 | 0.685 | +10.2 | +4.4 |
| WIFI | 0.588 | **0.705** | +11.7 | 0.694 | +11.1 | −0.1 |
| BOTH | 0.462 | **0.587** | +12.5 | 0.580 | +11.9 | +3.0 |

**E2 — receiver impairments** (`e2_table.csv`; Δ vs Phase 1 in pp, session estimator)

| impairment | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| awgn | +14.6 | +16.6 | +15.2 | |
| adc | +9.4 | +9.6 | +7.2 | |
| tilt | +9.2 | +5.6 | **−12.6** | **−25.8** |
| gain (step in **every** window) | **−55.0** | **−58.7** | **−56.0** | |
| gain, rare steps (5% of windows; `agc_rare.csv`) | +7.8 | +7.1 | +6.8 | |

- **Gain:** the every-window construction corrupts the noise windows the correction is estimated from. With realistic, rare AGC steps, centring beats Phase 1 by about 7 pp.
- **Tilt ≥ 12 dB is a real limitation:** it changes the spectral shape of noise and signal differently, and a mean-only correction can't absorb that. TENT on top of the centred source recovers it (tilt 4: 0.696 vs Phase 1's best 0.664).
- **β barely matters** (`beta_sensitivity.csv`): BOTH 0.582 / 0.580 / 0.574 and awgn 3 0.562 / 0.562 / 0.561 for β = 0.003 / 0.01 / 0.03.

**Shortcut re-test** (`shortcut_test.csv`; CNN trained on centred noise-only windows)
- 7-class accuracy: 0.43 → **0.317** (chance 0.143, flag at 0.286).
- Receiver-group accuracy: 0.98 → **0.907**.
- **Weaker but still flagged.** Centring removes the floor *level*, not its *shape*. (The std ablation reached 0.202 / 0.654, at the cost of accuracy.)

**E3 on the centred source** (`phase3/noisenorm/e3_table.csv`, same splits and hyperparameters as Phase 3; source-only → best baseline)

| target | Phase 1 source → best | centred source → best |
|---|---|---|
| BLUE / WIFI / BOTH | 0.588 / 0.585 / 0.457 → ≈ same | **0.704 / 0.703 / 0.585** → 0.705 / 0.712 / 0.591 |
| awgn 2 / awgn 3 | 0.550 / 0.409 → 0.640 / 0.573 (TENT) | 0.698 / 0.548 → **0.753 / 0.714** (TENT) |
| adc 3 | 0.685 → 0.685 | **0.755** → 0.765 (TENT) |
| tilt 4 | 0.657 → 0.664 (PL) | 0.397 → **0.696** (TENT) |
| gain 3 (every window) | 0.669 → 0.674 (PL) | 0.114 → 0.395 (AdaBN) |
| agc 3 (rare steps; `e3_agc.csv`) | 0.720 → ≈ same | **0.785** → ≈ same |

On the centred source, adaptation finally has a job: it recovers 41–70% of the tilt-4 and gain-3 failures. AdaBN on clean data still loses 2.9 pp [0.6, 5.6] with the 7-recording shift-B adapt sets (as in Phase 3).

**Adoption decision** (`adoption.json`)
- **Original rule** (8 E3 targets): centred − Phase 1 = −1.2 pp [−2.3, −0.1], p = 0.07 → Phase 1. This is driven by gain 3 (−56 pp).
- **Amended rule** (author decision 2026-10-19; `gain_3` → `agc_3`): **+6.6 pp [5.6, 7.5], p = 6e-8 → centred source adopted.**
  - Reason: `gain_3` steps every window, including the noise windows the normalisation is estimated from. The rare-step AGC check was specified before the original rule was evaluated.
  - **The paper reports both.**
- CLEAN: +9.4 pp [7.7, 11.1].

**Limitations to state in the paper**
- The noise/active window labels are offline (`select_windows`, recording-level percentile). Online detection is the paper-2 cascade's job.
- Streams start cold per recording (6.3% of rows excluded during warm-up).
- The shortcut is weakened, not removed (receiver-group accuracy 0.907).
- Centring fails under strong spectral tilt and under gain steps that hit the noise windows themselves.
- The adoption rule was amended post hoc (disclosed above).
