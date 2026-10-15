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
