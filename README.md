# Ventilator Pressure Prediction — Kaggle Competition

## Overview

The [Google Brain Ventilator Pressure Prediction](https://www.kaggle.com/competitions/ventilator-pressure-prediction) competition (2021) challenged participants to predict airway pressure in mechanical ventilation simulations. Each breath is an 80-step time series; the task is to predict pressure at each timestep given ventilator control signals (u_in, u_out) and lung mechanics parameters (R = resistance, C = compliance).

Kaggle profile: [illidan7](https://www.kaggle.com/illidan7)

## Approach

### 1. Physics-Informed EDA

RAPIDS KNN analysis revealed that u_in waveforms cluster into repeating patterns, and R/C parameters deterministically modulate the pressure response — C controls pressure accumulation rate, R controls spike magnitude. This physics intuition guided both feature design and model architecture.

### 2. Feature Engineering (100+ Features)

Evolved from ~38 to 100+ features in three tiers:
- **Temporal**: Lag/diff features capturing local flow dynamics
- **Statistical**: Rolling/expanding/EWM statistics and spectral features (FFT, Hilbert transform) capturing waveform shape
- **Cross-breath**: GroupBy features aggregating behavior within R-C-timestep groups and KMeans-cluster groups for cross-breath context

Built as a reusable Python module with cuDF/RAPIDS GPU acceleration.

### 3. Model Architecture: "IlliFiction"

Core model is a deep BiLSTM trunk (5 layers, 1024→128 units) with a parallel GRU branch using multiplicative skip connections and BatchNorm. Trained on TPU with 7-10 fold KFold CV, MAE loss, ReduceLROnPlateau + early stopping.

Architecture variants explored:
- **IlliDub**: Add-based residuals instead of multiplicative gating
- **Cerberus**: Multi-input model with inhale-masked branch
- **Classification**: 950-class pressure discretization experiment

### 4. KMeans-Augmented Features

Used RAPIDS KMeans at multiple cluster granularities (10 to 100K) to group similar breath waveforms, then computed cluster-aware aggregate statistics as additional features, providing cross-breath contextual information.

### 5. Ensemble Strategy

Final submission blends own models (IlliFiction, IlliFictionLast, IlliDub) with selected public kernels using adaptive mean/median blending — uses mean when fold predictions agree (spread < 0.45), median when outliers are detected. Post-processing rounds predictions to nearest known pressure step.

## Repository Structure

```
├── eda/
│   └── rapids-knn-eda.ipynb                # R/C physics understanding via KNN visualization
├── feature-engineering/
│   ├── feature-module.py                   # Reusable feature engineering module (cuDF)
│   ├── gpu-feature-pipeline.ipynb          # GPU-accelerated feature generation pipeline
│   └── knn-cluster-features.ipynb          # KMeans cluster-based groupby features
├── models/
│   ├── bilstm-pytorch-baseline.ipynb       # PyTorch BiLSTM baseline (18 versions)
│   ├── classification-experiment.ipynb     # 950-class pressure classification attempt
│   ├── illifiction-bilstm-gru.ipynb        # ★ Primary model — BiLSTM+GRU skip-net on TPU
│   ├── illidub-residual-variant.ipynb      # Add-based residual architecture variant
│   ├── cerberus-multi-input.ipynb          # Multi-input with inhale masking
│   └── illifiction-knn-final.ipynb         # Final model with KNN-augmented features
├── inference/
│   └── model-inference.ipynb               # Multi-fold inference with median aggregation
└── ensemble/
    └── ensemble-submit.ipynb               # ★ Adaptive mean/median ensemble (22 versions)
```

## Tech Stack

- **Models**: Keras/TF BiLSTM+GRU (TPU), PyTorch BiLSTM, RandomForest classifier
- **Feature Engineering**: RAPIDS/cuDF, cuML KMeans, FFT/Hilbert spectral features
- **Infrastructure**: Kaggle Notebooks (GPU, TPU v3-8)

## Competition

- **Name**: [Ventilator Pressure Prediction](https://www.kaggle.com/competitions/ventilator-pressure-prediction)
- **Type**: Time series regression
- **Metric**: Weighted MAE (inhale phase only)
- **Timeline**: September — November 2021
