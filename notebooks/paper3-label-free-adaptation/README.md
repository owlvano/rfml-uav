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

## Phase 5 — Stream adaptation (RQ3) and open-set gate (RQ4)

Notebooks: `7_Stream_Adaptation.ipynb` (streaming E3, trigger, single-drone check, cost ledger) and `8_Open_Set_Gate.ipynb` (E5-lite). Results in `results/paper3/phase5/`. Runner: `rfml_uav.adapt.stream.run_stream`.

**The ladder and its protocol**

| rung | mechanism | gradients |
|---|---|---|
| 1 | centring on the receiver-noise mean (Phase 4) | none |
| 2 | temporal vote, majority over the last W windows | none |
| 3a | event-triggered, guarded BN-statistics refresh | none |
| 3b | vote-pseudo-label fine-tune on a trigger (variant) | backward |
| 4 | open-set energy gate | none |

- **Streams.** One per (receiver condition, model):
  - a clean prefix: the model's CLEAN test recordings;
  - then the target segment: the target's E3 test-half recordings, chained in a fixed random order.
  - Windows are time-ordered within each recording. Centring uses the stream estimator (β = 0.01, k = 50) over the whole stream.
- **Runner.** Chunks of M = 250 active windows. Per chunk: predict, gate, vote, buffer, trigger, adapt, so every prediction precedes the update it feeds. No mechanism reads labels.
- **Gate.** The threshold is the (1 − q) quantile of calibration energies (the model's CLEAN validation rows), re-anchored after each refresh.
  - The original sliding quantile over accepted windows ratchets down: rejection went 0.18 → 0.84 on a stationary toy stream. It was replaced before any run (design D3).
- **Trigger.** Chunk-mean energy moves by more than θσ from the reference; cooldown C = 8 chunks.
- **Guarded refresh.**
  - Full (every BN layer) if the buffer's votes cover ≥ D classes.
  - Otherwise restricted: conv BN layers only, (1 − λ)·source + λ·buffer.
- **Comparison methods,** all on the same vote and gate: continuous AdaBN, periodic refresh, online TENT, online naive PL, vote-PL, head-only vote-PL.
- **Data.** 9 targets × 25 centred models = 225 streams.

**Selected hyperparameters** (`stream_hparams.json`, staged dev tuning on awgn_1 / gain_2 / adc_2 / tilt_3 with seed-0 models; `dev_tuning.csv`)
- W = 25 (≈ 50 ms of cached windows), q = 0.1, θ = 4, B = 1,000, D = 3, λ = 0.3.
- Continuous AdaBN: m = 0.1. Periodic: P = 20. TENT: lr = 1e-3. PL: lr = 1e-5, τ = 0.8 (Phase 3). Vote-PL: lr = 1e-5, ρ = 0.8.
- On the mild dev cells the trigger seldom fires, and more firing scored worse (θ = 2: 0.49 vs θ = 4: 0.63 dev vote F1). The insensitive θ = 4 won.

**Rung 2: the vote alone** (`vote_gain.csv`; source-only, target segment, vote − per-window macro-F1, pp)

| BLUE | WIFI | BOTH | awgn 2 | awgn 3 | gain 3 | agc 3 | adc 3 | tilt 4 |
|---|---|---|---|---|---|---|---|---|
| +9.9 | +7.7 | +10.8 | +7.8 | +6.3 | −1.8 | +4.9 | +6.7 | +3.0 |

- Every gain is significant (p ≤ 3e-4).
- No gradients; memory is the last 24 predictions.
- gain 3 is the exception: there the per-window errors are consistent, so voting cannot fix them.

**Streaming E3** (`stream_e3.csv`; target-segment vote macro-F1, mean over 25 runs; Δ vs source-only in pp, paired)

| target | source | triggered (3a) | continuous AdaBN | periodic | TENT | PL | vote-PL (3b) |
|---|---|---|---|---|---|---|---|
| BLUE | **0.761** | −0.9 | −7.3 | −21.7 | −7.2 | −7.3 | −0.8 |
| WIFI | **0.737** | −2.3 | −11.3 | −22.7 | −11.2 | −11.3 | −1.9 |
| BOTH | **0.657** | −0.0 | −8.7 | −18.6 | −8.5 | −8.6 | −0.1 |
| awgn 2 | **0.739** | −0.0 | −3.3 | −24.7 | −3.2 | −3.3 | −0.2 |
| awgn 3 | 0.588 | −0.3 | **+8.5** | −11.6 | +8.5 | +8.2 | −0.3 |
| gain 3 | 0.104 | +4.4 | +19.9 | +6.8 | **+20.0** | +20.0 | +6.0 |
| agc 3 | **0.795** | −0.5 | −8.3 | −19.4 | −7.9 | −8.3 | −0.5 |
| adc 3 | **0.781** | −0.0 | −6.0 | −23.7 | −5.8 | −6.3 | −0.0 |
| tilt 4 | 0.413 | +2.4 | +19.2 | +0.9 | **+19.5** | +18.9 | +2.2 |

- Bold = the best method per target.
- The head-only ablation equals vote-PL to within 0.01 pp on every target.
- Paired tests against the best method and against source-only are in the CSV.

**Harm on the clean prefix** (`stream_e3_harm.csv`; prefix vote F1 vs source-only, source ≈ 0.84–0.85)
- Triggered and vote-PL: within 0.3 pp (agc 3: −0.2).
- Continuous AdaBN, TENT and PL: **−7.4 to −11.0 pp** on every stream.
- Periodic: **−14.1 to −18.4 pp**.

**Trigger quality** (`trigger.csv`; triggered method; vote-PL is identical)
- **False triggers in the clean prefix:** 0.04 per stream (1 of 25 streams per target).
- **Detection rate** (streams with an event after the change point): gain 3 52%, tilt 4 20%, WIFI 16%, BLUE 4%, awgn 3 4%, and 0% on BOTH, awgn 2, agc 3 and adc 3.
- **Median delay:** gain 3 2,296 windows (≈ 4.6 s at the cached-window spacing); tilt 4 1,478 (≈ 3.0 s).

**Single-drone check** (`single_drone.csv`; per-recording streams, clean then impaired)
- Vote accuracy is within 0.2 pp of source-only on every shift-B target. At most 0.7% of streams lose more than 5 pp.
- Refreshes ran only on gain 3 (60, 43% restricted) and tilt 4 (30, 73% restricted).
- **The guard holds, but mostly because the trigger rarely fires.**
- The periodic method shows what happens without that protection:
  - its 1,000-window buffers hold 2–3 drones (seed-0 dev diagnosis);
  - vote errors make them look more diverse than they are, so the D-class guard lets full refreshes through, and accuracy collapses.

**E5-lite** (`e5_lite.csv`; 35 LOCO models, seed 0; source-only, means over held-out classes)

| condition | energy AUROC (unknown vs known) | unknown false-accept | known rejection | known vote F1, gate on / off |
|---|---|---|---|---|
| CLEAN | 0.636 | 0.833 | 0.114 | 0.866 / 0.840 |
| BOTH | 0.547 | 0.789 | 0.204 | 0.729 / 0.700 |

- Per held-out class, AUROC ranges over 0.52–0.75 (CLEAN) and 0.40–0.68 (BOTH, below chance for AIR, MIN and PHA).
- The gain with the gate on is a selection effect: the gate drops uncertain known windows. It does not reject unknowns.
- With the triggered refresh the numbers are the same, except where it fires (AIR / INS CLEAN, −6.9 / −4.7 pp known vote F1 with the gate on).

**Cost ledger** (`cost_ledger.csv`; hardware-independent, per active window)
- Feature extraction: WPD 3.91 M MACs. CNN forward: 1.87 M MACs.
- Total per window, with backward estimated at 2 × forward:

| method | total MACs | fwd passes | bwd passes | events per stream | changed per event |
|---|---|---|---|---|---|
| source | 5.78 M | 1.00 | 0 | 0 | — |
| triggered | 5.80 M (+0.2%) | 1.01 | 0 | 0.25 | 2,016 BN statistics |
| periodic | 6.14 M (+6%) | 1.19 | 0 | 8.3 | 2,016 statistics |
| continuous AdaBN | 7.50 M (+30%) | 1.91 | 0 | 175 | 2,016 statistics |
| TENT | 11.0 M (+90%) | 1.93 | 0.93 | 175 | 2,016 affine parameters |
| PL | 10.5 M (+82%) | 1.84 | 0.84 | 175 | 409,626 parameters (all) |
| vote-PL | 5.82 M (+0.6%) | 1.01 | 0.004 | 0.26 | all parameters + statistics |

The ladder figure is `ladder.png`.

**Limitations**
- **Cached-window spacing.** Windows are about 2 ms apart, so W and the delays in windows are coarser than real time.
- **Energy trigger.** Calibrated on chunk means that vary between drones, it is too insensitive for mild shifts. Drone changes inside a condition also move it.
- **E5-lite is seed 0 only** (35 models); variation is over folds and classes.
- **Noise/active labels are offline,** as in Phase 4.
- **Frozen-protocol result.** These numbers come from the pre-registered dev tuning. Fixes suggested by the results (see NOTES "Phase 5 outcome") need a new change and are disclosed as post hoc.

## Phase 6 — Receiver-gated, drone-balanced statistics (post hoc)

Notebook: `9_Gated_Adaptation.ipynb`. Results in `results/paper3/phase6/`.

**Post-hoc disclosure.** These methods were designed on 2026-10-26, after the frozen Phase 5 target results were seen. A spike ran five balanced-buffer settings on the dev cells only. Every Phase 6 row is reported beside the frozen Phase 5 rows on identical streams: the rows and the source-only votes are checked against `phase5/streams`.

**Methods** (`rfml_uav.adapt.stream`; drift from `stream_normalise(..., return_drift=True)`)
- **Noise-floor drift:** ‖μ_i − μ_0‖₁ / ‖μ_0‖₁ of the stream centring state against its warm-up mean. It is free: the state already exists.
- **Receiver gate:** ON when drift > φ_on, OFF below 0.5·φ_on. OFF restores the source BN statistics.
- **Methods:**
  - `gated_continuous`: Phase 5 continuous AdaBN, only while the gate is ON;
  - `balanced`: continuous AdaBN on a per-vote-class FIFO buffer (Bc per class; updates only when ≥ D classes are half full), always on;
  - `gated_balanced`: the balanced update, only while ON, from a buffer cleared at ON.
- **Oracle** (diagnostic): `gated_continuous` with the gate forced ON exactly at the change point.

**Selected** (`hparams.json`; harm-aware objective J = dev vote F1 − positive prefix loss; `dev_tuning.csv`)
- `gated_continuous`: φ_on = 0.7, m = 0.1. The dev prefix never opens the gate (clean-prefix drift ≤ ≈ 0.49).
- `balanced`: Bc = 200, D = 5, m = 0.01.
- `gated_balanced`: φ_on = 0.7. No dev setting ever updated, so it was a tie.

**Comparison with the frozen Phase 5 rows** (`gated_e3.csv`; target-segment vote macro-F1; Δ vs source-only in pp with 95% CI; prefix Δ in brackets)

| target | source | P5 continuous | P5 triggered | gated continuous | balanced | oracle gate |
|---|---|---|---|---|---|---|
| BLUE | 0.761 | −7.2 (−9.4) | −0.9 (0.0) | 0.0 (0.0) | +0.9 [0.3, 1.5] (+0.7) | −6.2 |
| WIFI | 0.737 | −11.3 (−8.6) | −2.3 (0.0) | 0.0 (0.0) | +0.6 [0.1, 1.2] (+0.8) | −10.1 |
| BOTH | 0.657 | −8.7 (−8.1) | −0.0 (0.0) | 0.0 (0.0) | +0.4 [−0.2, 1.1] (+0.6) | −7.8 |
| awgn 2 | 0.739 | −3.3 (−8.4) | −0.0 (0.0) | 0.0 (0.0) | +1.3 [0.5, 2.2] (+0.4) | +2.1 |
| awgn 3 | 0.588 | +8.5 (−9.1) | −0.3 (0.0) | 0.0 (0.0) | +2.8 [1.3, 4.4] (+0.7) | **+12.1** |
| gain 3 | 0.104 | +19.9 (−7.4) | +4.4 (0.0) | **+19.4 [14.6, 24.3] (0.0)** | +0.6 [−1.5, 2.4] (+0.7) | +20.0 |
| agc 3 | 0.795 | −8.3 (−8.5) | −0.5 (−0.2) | 0.0 (0.0) | −0.0 [−0.9, 0.8] (+0.3) | −4.2 |
| adc 3 | 0.781 | −6.0 (−8.0) | −0.0 (0.0) | 0.0 (0.0) | +1.6 [0.6, 2.6] (+0.4) | −2.2 |
| tilt 4 | 0.413 | +19.2 (−10.8) | +2.4 (0.0) | +2.7 [0.0, 6.8] (0.0) | +2.0 [0.9, 3.0] (+0.3) | **+22.2** |

- `gated_balanced` never updated on any target, so it equals source-only everywhere: after the gate opens, the votes never cover D = 5 classes.
- The oracle's prefix equals source by construction.

**Gate** (`gate.csv`; `gated_continuous`)
- The gate never opens in any clean prefix (0% of windows).
- **gain 3:** opens on 25/25 streams, 97% of target windows ON, median delay 60 windows (≈ 0.1 s at the cached spacing).
- **tilt 4:** opens on 3/25 streams (12%), median delay 967 windows.
- No other target moves the floor past φ_on.

**Cost** (`cost.csv`; per active window)

| method | forward passes | backward | updates per stream | gate ON |
|---|---|---|---|---|
| gated continuous | 1.03 | 0 | 5.7 | 4.6% |
| balanced | 5.73 | 0 | 155 | always |
| gated balanced | 1.00 | 0 | 0 | 4.6% |
| oracle | 1.44 | 0 | 100 | 51% |

Compare `phase5/cost_ledger.csv`: continuous AdaBN costs 1.91 forward passes per window.

**Reading**
- **The noise-floor gate turns the gain-step trade-off into a free gain.** Gated continuous AdaBN keeps continuous AdaBN's +19.4 pp on gain 3 (the oracle reaches +20.0), with zero prefix harm and zero change on every other target, at 1.03 forward passes per window.
- **Even a perfect "change happened" gate is not enough.** The oracle still loses 2–10 pp on interference, agc 3 and adc 3. Adaptation should switch on only for shifts it can fix (gain, tilt, strong AWGN), not for every change. A floor-magnitude gate does that selectively, but it misses AWGN (oracle +12.1 on awgn 3) and most tilt (+22.2) because those barely move the noise mean.
- **The balanced buffer removes the harm but most of the gain too.** Always-on balanced adaptation is harm-free (prefix +0.3 to +0.8) but gains only 0–3 pp, at 5.7× the forward cost.

**Limitations**
- **Post hoc** (disclosed above).
- **The drift reference is the stream's first warm-up,** so a deployment that starts impaired would see that floor as normal.
- **Floor drift is bimodal on clean data** (two receiver groups at ≈ 0.04 and 0.4–0.56): φ_on must sit above the between-receiver spread of the deployment.
- **The tilt detection rate is low** (12%).
- **No single-drone or E5-lite runs** for the new methods.

**Why there is no deterministic gain-correction baseline.** A reviewer may ask whether rescaling by the floor drift would match gated AdaBN on gain 3. It can't, for two reasons:
- **A constant gain offset is already invisible.** The features are unit-power per window, and the `impairments` and `wavelets` self-checks verify this.
- **Only the in-window step is left,** at a random position in each window. The only deterministic correction available, stream noise-mean centring, is already part of every row, source-only included.

So the +19.4 pp is over the best deterministic correction the pipeline has. Division by the noise std was rejected in Phase 4 because it inflates features under gain steps.
