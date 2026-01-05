# Identification of Unmanned Aerial Vehicles Using RF Fingerprinting and Deep Learning Networks

This directory contains the source code and experimental scripts supporting the paper:

> **Y. P. Kondratenko, I. Sova, O. V. Kozlov, and V. Kuzmenko**,  
> *“Identification of unmanned aerial vehicles using RF fingerprinting and deep learning networks,”*  
> Proceedings of the Modern Machine Learning Technologies Workshop (MoMLeT-2025),  
> Lviv, Ukraine, June 14–15, 2025.  
> CEUR Workshop Proceedings, Vol. 4004.  
> https://ceur-ws.org/Vol-4004/paper23.pdf

## Purpose of the Code

The provided source code implements the core components of the RF fingerprinting–based UAV identification pipeline described in the paper, including:

- neural network architectures used for UAV identification,
- evaluation routines for classification performance.

The code is intended to **support reproducibility of the experimental results** and to illustrate the practical implementation of the proposed methods.

Data preprocessing and exploratory analysis logic are provided in the corresponding `data-processing` and `analysis` directories of the repository.

## Environment and Dependencies

The code is written in **Python** and is designed to run within a Conda-managed environment.

A complete list of dependencies and the environment configuration are specified in the project-level `requirements.yml` file.

## Reproducibility Notes

To reproduce the experiments described in the paper:

1. Set up the Python environment as described in the main repository README.
2. Prepare the RF dataset according to the expected directory structure.
3. Execute the preprocessing and training scripts in the order described in the accompanying notebooks or scripts.
4. Evaluate trained models using the provided evaluation routines.

Exact numerical results may vary slightly depending on hardware configuration, random initialization, and dataset partitioning.

## Citation

If you use this code in academic work, please cite the original paper:

```bibtex
@inproceedings{kondratenko_identification_2025,
  series    = {CEUR Workshop Proceedings},
  title     = {Identification of Unmanned Aerial Vehicles Using RF Fingerprinting and Deep Learning Networks},
  volume    = {4004},
  url       = {https://ceur-ws.org/Vol-4004/paper23.pdf},
  booktitle = {Proceedings of the Modern Machine Learning Technologies Workshop (MoMLeT-2025)},
  publisher = {CEUR-WS.org},
  author    = {Kondratenko, Yuriy P. and Sova, Ivan and Kozlov, Oleksiy V. and Kuzmenko, Vitalii},
  editor    = {Emmerich, Michael and Lytvyn, Vasyl and Vysotska, Victoria},
  year      = {2025},
  pages     = {312--326}
}
```