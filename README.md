This is a Pytorch implementation of Pre-DyGAE: Pre-training Enhanced Dynamic Graph Autoencoder for Occupational Skill Demand Forecasting

# Pre-DyGAE: Pre-training Enhanced Dynamic Graph Autoencoder for Occupational Skill Demand Forecasting
# The data
The four datasets are stored in a form of quadruple sets < occupation − relation − skill − demand > without specific names of occupations and skills for protecting company privacy.

# The Code

## Requirements

Following is the suggested way to install the dependencies:

```
pip install -r requirements.txt
```

Note that pytorch >= 1.13.0

## Folder Structure

```tex
└── data
    └── Dai                     # Daily Dataset
        ├── task1               # dataset for OSD graph completion
        └── task2               # dataset for OSD forecasting
    ├── Fin                     # Fin Dataset
    ├── IT                      # IT Dataset
    └── Man                     # Man Dataset
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
└── scripts                     # The running command
    ├── task1.py                    # The pipeline for OSD graph completion
    └── task2.py                    # The pipeline for OSD forecasting
├── requirements.txt            # The dependencies
└── README.md                   # This document
```

## Train and Test

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
