# Proposal

## Why

Paper 3 makes two promises to *Energies*: the method works on DroneDetect, and it is evaluated under Wi-Fi and Bluetooth interference. The current pipeline can't keep either one. The repo has no DroneDetect loader (`src/rfml_uav/drone_detect/load_data.py` is the dataset authors' demo script), no I/Q features, and no source model for this dataset.

The exploration (`1_DroneDetect_Exploration.ipynb`) found four facts that shape the work:
- there are no drone-free recordings;
- the drones were captured with two receiver configurations, which gives a classifier a possible shortcut;
- the interference subsets also shift the gain;
- the signals are bursty.

This change builds the DroneDetect source baseline and measures how much accuracy interference costs, before any adaptation code exists. It serves **RQ1** and experiment **E1** (interference table), and it feeds **thesis chapter 3, section 3.1** (interference sensitivity). It also settles the logit-tail question raised in Phase 0, which the open-set gate (E5-lite) depends on.

## What Changes

- **DroneDetect is identification-only (decision a).** 7 drone classes, no Background class. Signal-presence detection is handled upstream by the paper 2 cascade, and the paper states this.
- **Loader.** A shared DroneDetect loader parses the file inventory (subset, drone, mode, recording index) from the folder layout and reads recordings via memmap. The folder name is the label, because files in `MP1` are prefixed `MA1` and files in `MP2` are prefixed `MAV`.
- **Active-window selection.** Fixed, non-overlapping 4000-sample windows are scored by mean power. Only windows at least 6 dB above the recording's quiet floor are kept, with a fixed cap per recording spread over time. Windows below the floor are kept separately as noise-only windows for the shortcut test.
- **Per-window preprocessing.** DC removal and unit-power normalisation. This removes the absolute-level and DC differences between the two receiver configurations.
- **I/Q features.** I and Q are treated as two real channels. Each gets the existing WPD (`dmey`, level 6, RMS + STD per node), concatenated I ‖ Q into a 256-d vector. The existing CNN input is unchanged.
- **One feature file with per-row provenance**: subset, drone, mode, recording, window start, active/noise flag.
- **Source model.** Trained on CLEAN recordings only, with a recording-level split by file, 5 seeds × 5 folds, and checkpoint selection on validation only.
- **Interference table (E1).** Each source model is evaluated on CLEAN test recordings and on all BLUE, WIFI and BOTH recordings, per subset and per mode. Phantom FY is excluded from the shift-A comparison because CLEAN has no Phantom FY recordings.
- **Shortcut test.** The same CNN is trained on noise-only windows. Its drone accuracy and its accuracy at telling the two receiver groups apart are reported. Above a fixed threshold, in-band-only features are applied as mitigation and the baseline is re-run.
- **Logit-tail ablation.** The CNN gains an optional plain linear tail (no `bn7`/`prelu4` on the logits). The default stays as it is, so existing checkpoints still load. The tail used later is chosen by energy-score AUROC versus max-softmax AUROC.
- Out of scope:
  - adaptation code (OG-TCST and baselines);
  - receiver-impairment simulation (shift B);
  - the open-set gate and E5-lite;
  - energy measurement;
  - any change to DroneRF data or Phase 0 results.

## Capabilities

### New Capabilities
- `dronedetect-iq-features`: DroneDetect inventory and loading, active/noise window selection, per-window normalisation, and 2-channel WPD features with per-row provenance.
- `interference-shift-evaluation`: CLEAN-trained source protocol, per-subset and per-mode target report, exclusion rules, the receiver-shortcut test, and the energy-score separability report for this dataset.

### Modified Capabilities
- `classifier-logits-output`: adds an optional plain linear logit tail. The default architecture and checkpoint compatibility are unchanged.

## Impact

- **Code:**
  - new `src/rfml_uav/drone_detect/` package modules (inventory/loader, window selection);
  - the existing `load_data.py` demo script and `instructions.md` stay as reference;
  - `src/rfml_uav/nn/cnn.py` and `models.py` (optional tail flag);
  - WPD feature code moves into `src/rfml_uav/` so DroneRF and DroneDetect share one implementation.
- **Notebooks:** new `notebooks/paper3-label-free-adaptation/2_DroneDetect_Features.ipynb` and `3_DroneDetect_Source_Baseline.ipynb`.
- **Data:**
  - reads `DRONEDETECT_PATH` (374 GB, external drive), with one full pass of about 20–25 min;
  - writes features, results and checkpoints under `DATA_PATH/results/paper3/phase1/` and `DATA_PATH/checkpoints/paper3/phase1/`.
- **Dependencies:** none new (`pywt`, `joblib` and scikit-learn are already installed).
- **Paper:**
  - the Background-class absence and the gain confound of the interference subsets go into the Setup / Limitations sections;
  - if the shortcut test fails, the mitigation is reported as well.
