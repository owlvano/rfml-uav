# RFML-UAV  
**Radio-Frequency Machine Learning for UAV Signal Detection and Identification**

This repository contains research code, experiments, and evaluation pipelines for **radio-frequency machine learning (RFML)** applied to **unmanned aerial vehicle (UAV)** signal detection and identification.  
The project focuses on **signal presence detection**, **cascade detection models**, and **learning-assisted RF analysis** under noisy and dynamically changing spectrum conditions.

The repository is designed to support:
- reproducible experiments,
- multiple research papers,
- and long-term PhD-level development.

## Research Scope

The main research directions covered in this repository include:

- Classical RF signal presence detection  
  (energy detection, wavelet-based detection, cyclostationary analysis)
- Cascade detection frameworks for computationally efficient RF monitoring
- Chunk-level and segment-level detection evaluation
- RF feature extraction from I/Q data
- Machine-learning-assisted threshold adaptation
- UAV-oriented RF datasets (e.g., DroneRF and derived subsets)

The code bridges **classical digital signal processing (DSP)** methods and **modern machine-learning workflows**, with a strong emphasis on **experimental clarity and reproducibility**.

---

## Environment Setup

This project uses **Conda** for dependency and environment management.

### 1. Create the environment

```bash
conda env create -f requirements.yml
conda activate rfml-uav
```

### 2. Install the source code (editable mode)
```bash
pip install -e .
```

This registers the project source code as an importable Python package, allowing shared modules to be used consistently across notebooks and scripts.

### 3. (Optional) Verify installation
```bash
python -c "import rfml_uav; print(rfml_uav.__file__)"
```

## Data Policy

Due to size and licensing constraints, raw RF recordings and derived signal data are not included in this repository.

Excluded data typically includes:

- raw I/Q recordings,
- `.npz`, `.npy`, `.wav`, `.sigmf` files,
- chunked or intermediate signal representations.

Datasets used in the experiments are described in the corresponding notebooks and research papers.

##  Experiments and Publications

Experimental work is primarily conducted using Jupyter notebooks and is organized according to research purpose and target publication.

The repository includes:

- exploratory signal and metric analysis,
- dataset preparation and preprocessing pipelines,
- experiments related to UAV RF identification,
- experiments related to cascade-based signal detection methods.

Each experimental workflow relies exclusively on shared, reusable code from the main source package to ensure consistency and reproducibility.

## Usage and Attribution

This repository is intended primarily for academic and research use.
When using this code in publications, please ensure proper attribution to the original dataset authors and relevant prior work.

## Citations

This work builds upon publicly available datasets and prior research in radio-frequency signal processing and machine learning.

### Datasets

- Al-Sa’d, M., Allahham, M. S., Mohamed, A., Al-Ali, A., Khattab, T., & Erbad, A. (2019).  
  *DroneRF dataset: A dataset of drones for RF-based detection, classification, and identification.*  
  Mendeley Data, v1.  
  https://doi.org/10.17632/f4c2b4n755.1

### Background and Related Work

- Aburakhia, S., Shami, A., & Karagiannidis, G. K. (2024).  
  *On the Intersection of Signal Processing and Machine Learning: A Use Case-Driven Analysis Approach.*  
  arXiv preprint arXiv:2403.17181.  
  https://arxiv.org/abs/2403.17181


### BibTeX

If you use this repository or the referenced datasets, please cite the following works.

```bibtex
@dataset{al_sad_2019_dronerf,
  author    = {Al-Sa'd, Mohammad and Allahham, Mhd Saria and Mohamed, Amr and
               Al-Ali, Abdulla and Khattab, Tamer and Erbad, Aiman},
  title     = {DroneRF Dataset: A Dataset of Drones for RF-Based Detection, Classification, and Identification},
  year      = {2019},
  publisher = {Mendeley Data},
  version   = {1},
  doi       = {10.17632/f4c2b4n755.1}
}

@article{aburakhia2024intersection,
  author  = {Aburakhia, Shami and Karagiannidis, George K.},
  title   = {On the Intersection of Signal Processing and Machine Learning: A Use Case-Driven Analysis Approach},
  journal = {arXiv preprint arXiv:2403.17181},
  year    = {2024},
  url     = {https://arxiv.org/abs/2403.17181}
}
```