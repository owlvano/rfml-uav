# Burst-Aware Cascade Detection of UAV Radio-Frequency Signals Using Energy and Cyclostationary Analysis

This directory contains the source code and experimental notebook supporting the paper:

> **Burst-Aware Cascade Detection of UAV Radio-Frequency Signals Using Energy and Cyclostationary Analysis**

The work proposes and evaluates a **cascade-based signal presence detection framework** for UAV radio-frequency (RF) signals, explicitly accounting for **burst structure and signal duration** in time.

## Purpose of the Code

The provided code implements the burst-aware cascade detection framework described in the paper, including:

- energy-based coarse signal presence detection,
- wavelet-based intermediate detection (optional baseline),
- blind cyclostationary analysis as a selective fine-stage detector,
- burst detection and duration-aware score aggregation,
- chunk-level and segment-level performance evaluation,
- computational cost and selectivity analysis of cascade stages.

The implementation is designed to **demonstrate the practical realization of the proposed cascade model** and to support **experimental reproducibility**.

## Methodological Overview

The cascade detector operates in multiple stages:

1. **Fast energy-based screening** to reject noise-only segments.
2. **Temporal aggregation and burst detection**, enforcing minimum signal duration constraints.
3. **Selective invocation of cyclostationary analysis** only for candidate signal bursts.
4. **Segment-level decision making** using duration-constrained score aggregation.

This design significantly reduces computational load while preserving detection performance under low-SNR and intermittent transmission conditions.

## Code Structure and Execution

The main experimental workflow is implemented as a Jupyter notebook and includes:

- loading of pre-chunked RF signal data,
- estimation of background energy thresholds at a fixed false-alarm rate,
- chunk-level detector score computation,
- segment-level burst-aware aggregation,
- ROC analysis at both chunk and segment levels,
- Pd–Pfa evaluation at fixed operating points,
- timing and cascade selectivity statistics,
- visualization of burst statistics and detector performance.

The code relies exclusively on reusable detector implementations provided in the shared source package.

## Dataset Usage

The experiments uses the DroneRF dataset that is pre-chunked and saved as `.npz` files and organized by UAV identifier and segment index.

Due to size and licensing constraints, **raw RF data and derived chunks are not included** in this repository.  
Users must prepare the dataset independently by using the notebooks in the `data-processing` directory

## Environment and Dependencies

The code is written in **Python** and is designed to run within a Conda-managed environment.

All required dependencies and environment configuration are defined in the project-level `requirements.yml` file.

## Scope and Limitations

- The implementation focuses on **offline evaluation** using pre-recorded RF data.
- Real-time SDR acquisition and hardware-specific optimizations are out of scope.
- Cyclostationary analysis is implemented in a blind form and assumes no prior knowledge of signal parameters.

## Citation

If you use this code in academic work, please cite the corresponding paper once published.


## Usage and Attribution

This source code is provided for **academic and research purposes only**.  
Any reuse in publications should include appropriate citation of the original work.
