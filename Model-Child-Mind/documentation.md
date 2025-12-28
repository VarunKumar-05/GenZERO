# Child Mind Institute: Advanced Hybrid BDH-Booster Model

## Overview
**[Technical Reference: Code Documentation](CODE_DOCUMENTATION.md)** | **[Deep Dive: Model Details](MODEL_DETAILS.md)**

This solution implements a state-of-the-art hybrid architecture to predict Problematic Internet Use (PCIAT scores). It combines:
1.  **Age-Gated Biological Dynamic Hebbian (BDH) Model**: A Spiking Graph Transformer that learns temporal activity patterns from Actigraphy, gated by age.
2.  **Sparse Autoencoder (SAE)**: Compresses the synaptic memory graph into a non-linear latent space, preserving rare but significant behavioral signals.
3.  **Synaptic Imputation**: Uses the similarity of BDH graphs to impute missing values in the Tabular dataset.
4.  **Ensemble Boosters**: LightGBM, XGBoost, and CatBoost trained on the hybrid features, optimized for Quadratic Weighted Kappa (QWK).

## Architecture Refinements

### 1. Age-Gated BDH (`src/model/bdh_features.py`)
*   **Mechanism**: The synaptic weights $W_{init}$ are initialized from a bank of learnable prototypes based on the subject's Age Group (<10, 10-15, 16+). This ensures the model interprets "sedentary behavior" contextually.
*   **Input**: Actigraphy Time-series (5 channels) + Age Group.
*   **Output**: 4096-dimensional Synaptic Memory Graph.

### 2. Sparse Autoencoder (`src/model/sae.py`)
*   **Goal**: Solve the "Synaptic Bottleneck".
*   **Observation**: PCA destroys non-linear clusters.
*   **Solution**: An SAE compresses the 4096-dim graph to 32 Latent Features, enforcing sparsity to isolate distinct behavioral phenotypes.

### 3. Synaptic Imputation (`src/model/imputation.py`)
*   **Goal**: Fix the "Swiss Cheese" data.
*   **Solution**: 
    1.  Compute Cosine Similarity between subjects' SAE Latent Features.
    2.  For a subject with missing tabular data, find their "Synaptic Neighbors".
    3.  Impute missing values using the weighted average of neighbors.
    *   *Logic*: If Subject A and B have identical movement patterns, their physiological stats are likely similar.

## Implementation Details
*   `train_hybrid.py`: The master script.
    *   **Pre-training**: Trains BDH and SAE.
    *   **Imputation**: Runs KNN on SAE features to fill Tabular NaNs.
    *   **Boosters**: Trains Ensemble on Imputed Tabular + SAE Latent features.
    *   **Metric**: Evaluates using **Quadratic Weighted Kappa (QWK)**.

## Usage
```bash
python train_hybrid.py
```
*Outputs: `bdh_child_mind.pth`, `sae_child_mind.pth`, and training metrics.*

### Visualization
To generate a 3D plot of the Synaptic Memory Graph:
```bash
python -m src.analysis.viz_child_mind
```
*Outputs: `child_mind_viz.png`*

