# Pre-DyGAE: Pre-training Enhanced Dynamic Graph Autoencoder for Occupational Skill Demand Forecasting

![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)

| **[Overview](#overview)**
| **[Installation](#installation)**
| **[How to Run Model](#how-to-run-model)**
| **[Dataset](#dataset)**
| **[Folder Structure](#folder-Structure)**
| **[Appendix](#appendix)**
| **[Citation](#citation)**
| **[License](#license)**

## Overview

Official code for article "[Pre-DyGAE: Pre-training Enhanced Dynamic Graph Autoencoder for Occupational Skill Demand Forecasting (IJCAI-24)](https://www.ijcai.org/proceedings/2024/0222.pdf)".

Occupational skill demand (OSD) forecasting seeks to predict dynamic skill demand specific to occupations, beneficial for employees and employers to grasp occupational nature and maintain a competitive edge in the rapidly evolving labor market. Although recent research has proposed data-driven techniques for forecasting skill demand, the focus has remained predominantly on overall trends rather than occupational granularity. In this paper, we propose a novel Pre-training Enhanced Dynamic Graph Autoencoder (Pre-DyGAE), forecasting skill demand from an occupational perspective.

![Framework](paper/articture.png)

## Installation

Create a python 3.8 environment and install dependencies:

```
  conda create -n python3.8 PreDyGAE
  source activate PreDyGAE
```

Install library

```
  pip install -r requirements.txt
```

Note that pytorch >= 1.13.0

## Dataset
The datasets used in our experiments were collected from the public information of one of the largest online recruitment platforms. We gathered JDs spanning a diverse range of occupations, covering the period from January 2020 to December 2023. Along this line, we constructed large-scale datasets from four major industries, i.e., the Daily dataset (Dai), Finance dataset (Fin), IT dataset (IT), and Manufacturing dataset (Man). They have the same format in this repository. The four datasets are stored in a form of quadruple sets < occupation − relation − skill − demand > without specific names of occupations and skills for protecting the privacy.

### Occupational Skill Demand

| Occupation | Relation | Skill |       Demand       |
| :--------: | :------: | :---: | :----------------: |
|   20057   |  100000  | 30345 | 0.1111111111111111 |
|    ...    |   ...   |  ...  |        ...        |

## Folder Structure

```tex
└── data
    └── Dai                         # Daily Dataset
        ├── task1                   # dataset for OSD graph completion
        └── task2                   # dataset for OSD forecasting
    ├── Fin                         # Fin Dataset
    ├── IT                          # IT Dataset
    └── Man                         # Man Dataset
└── code
    ├── models                      # Related Models
    ├── args.py                     # The parameters
    ├── graphSampler.py             # Inherited from DGL
    ├── mf.py                       # For matrix factorization
    ├── main.py                     # For training and test
    ├── model.py                    # OSCIA
    ├── mydataset.py                # Inherited from DGL
    ├── temporal_shift_infer.py     # Temporal Shift Module
    ├── test_task1.py               # Test code for OSD graph completion
    ├── test_task2.py               # Test code for OSD forecasting
    ├── train_task1.py              # Train code for OSD graph completion
    ├── train_task2.py              # Test code for OSD forecasting
    ├── trainer.py                  # Train and Test functions
    ├── utils.py                    # Tools
└── scripts                         # The running command
    ├── task1.py                    # The pipeline for OSD graph completion
    └── task2.py                    # The pipeline for OSD forecasting
├── LICENSE                         # The MIT LICENSE
├── README.md                       # This document
└── requirements.txt                # The dependencies
```

## How to Run Model

To run the Pre-DyGAE, you should set args in `code/models/args.py` in advance to run `script/task2.sh` for both training and testing phase:

Predict the OSD in the future.

```
sh scripts/task2.sh Dai 100 0.1 0.05 tweedie yes 1.0 yes # For Dai Dataset
sh scripts/task2.sh Fin 100 0.1 0.05 tweedie yes 1.0 yes # For Fin Dataset
sh scripts/task2.sh IT 100 0.1 0.05 tweedie yes 1.0 yes  # For IT Dataset
sh scripts/task2.sh Man 100 0.1 0.05 tweedie yes 1.0 yes # For Man Dataset
```

Complete the OSD graph (validation experiment):

```
sh scripts/task1.sh Dai 0 3 tweedie softplus 100 0.1 0.05 yes yes yes # For Dai Dataset
sh scripts/task1.sh Fin 0 3 tweedie softplus 100 0.1 0.05 yes yes yes # For Fin Dataset
sh scripts/task1.sh IT 0 3 tweedie softplus 100 0.1 0.05 yes yes yes  # For IT Dataset
sh scripts/task1.sh Man 0 3 tweedie softplus 100 0.1 0.05 yes yes yes # For Man Dataset
```

## Appendix

Due to the page limit, we have uploaded the appendix to github, strored in paper/appendix.pdf.

## Citation

If you find our work is useful for your research, please consider citing:

```
@inproceedings{ijcai2024p222,
  title     = {Pre-DyGAE: Pre-training Enhanced Dynamic Graph Autoencoder for Occupational Skill Demand Forecasting},
  author    = {Chen, Xi and Qin, Chuan and Wang, Zhigaoyuan and Cheng, Yihang and Wang, Chao and Zhu, Hengshu and Xiong, Hui},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {2009--2017},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/222},
  url       = {https://doi.org/10.24963/ijcai.2024/222},
}
```

## License

This project is licensed under the MIT License.
