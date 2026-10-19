# Proposal

## Why

Phase 1 measured shift A, the interference shift on DroneDetect. Paper 3 also claims robustness to shift B: a different or degraded receiver. No recordings from a second receiver exist (real cross-receiver, E8, is out of scope), so shift B has to be simulated. The adaptation baselines (E3) and OG-TCST will be compared on both shifts, so the shift-B benchmark and its source-only numbers are needed first.

This change serves **RQ1** and experiment **E2** (receiver-impairment table). It feeds **thesis chapter 3** (section 3.1, shift sensitivity). It also provides the shifted target sets that E3 adapts on.

One design fact found while planning shapes the work. `iq_wpd_features` removes DC and scales each window to unit power, so a **constant gain change has no effect on DroneDetect features**. The DroneRF WPD (`wpd_features`) has no normalisation, so the same gain change moves every DroneRF feature. Gain impairments must therefore vary *inside* the window to matter on DroneDetect, and the two datasets will respond differently to gain. The paper should report that contrast.

## What Changes

- **Impairment simulator.** A shared function applies one receiver impairment at one of 3 severities to a raw window. It works for real windows (DroneRF L/H channels) and complex windows (DroneDetect I/Q). It is deterministic for a given seed. The 4 impairments:
  - AWGN at a fixed SNR relative to window power;
  - gain / AGC drift: a constant gain offset plus a gain step inside the window;
  - frequency-response tilt: a linear-in-dB slope across the band;
  - ADC: re-quantisation to fewer bits with clipping at a fixed multiple of window RMS.
- **Severity 0 = identity.** With no impairment, the pipeline reproduces the cached Phase 0 / Phase 1 features. This is the check that the impaired features differ from the clean ones only by the impairment.
- **Impaired feature sets.** Raw windows are re-read and impaired, then featurised with the unchanged WPD code:
  - DroneDetect: every CLEAN active window (from `DRONEDETECT_PATH` by window start);
  - DroneRF: every non-duplicate window (from `chunked/`, L and H impaired independently).
  One file per (dataset, impairment, severity), row-aligned with the clean feature file.
- **Source-only evaluation (E2).** No retraining. The 25 saved Phase 1 models (linear tail) and the 25 Phase 0 `recording_dedup` models are scored on the impaired version of their own test rows. The report gives accuracy and macro-F1 per (dataset, impairment, severity), and the drop from clean with a paired Wilcoxon test and 95% bootstrap CI.
- **Energy-score separability under shift B.** Energy and max-softmax AUROC for correct vs wrong, per cell, so the open-set gate can be judged on shift B too.
- **Severity escalation rule.** If an impairment's strongest severity costs less than 3 pp of macro-F1 on DroneDetect, one stronger level is added and reported. This is the plan's go/no-go.
- Out of scope:
  - adaptation code (E3, OG-TCST);
  - combined shift A + B (impairments on BLUE / WIFI / BOTH);
  - impairments that need a time model across windows (slow drift over the recording, CFO, IQ imbalance);
  - retraining or augmentation with impairments;
  - any change to the MoMLeT-2025 model or to Phase 0 / Phase 1 results.

## Capabilities

### New Capabilities
- `receiver-impairment-simulation`: the 4 impairments × 3 severities on raw real or complex windows, the severity table, determinism, and the identity check.
- `impairment-shift-evaluation`: impaired feature sets for both datasets, source-only scoring of the saved Phase 0 / Phase 1 models, the E2 table with significance, energy-score AUROC under shift B, and the escalation rule.

### Modified Capabilities
None. The WPD feature functions, the window selection and the models are used unchanged.

## Impact

- **Code:**
  - new `src/rfml_uav/data/impairments.py` (with a `__main__` self-check);
  - new `src/rfml_uav/eval/stats.py`: the paired Wilcoxon + bootstrap-CI helper, now inline in notebooks 0 and 3. This notebook is its third user and E3 its fourth;
  - no behaviour change to existing modules.
- **Notebooks:** new `notebooks/paper3-label-free-adaptation/4_Receiver_Impairment_Shift.ipynb`.
- **Data:**
  - reads the CLEAN subset of `DRONEDETECT_PATH` (external drive) at the cached window starts only, and `DATA_PATH/chunked/`;
  - writes impaired features and results under `DATA_PATH/results/paper3/phase2/`;
  - reads, never writes, `DATA_PATH/checkpoints/paper3/phase0/` and `phase1/`.
- **Compute:** about 12 extra feature passes per dataset (≈ 95k DroneDetect windows, ≈ 164k DroneRF windows). WPD takes ≈ 2.4 ms per window, so expect ≈ 20–30 min on 8 cores in total (DroneDetect reads from the external drive may dominate). Scoring is minutes on the GPU.
- **Dependencies:** none new.
- **Paper:**
  - shift B is simulated, not a real second receiver. This goes in Limitations;
  - the gain-invariance of the I/Q features is a design property to state in Methods.
