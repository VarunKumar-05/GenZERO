# Child Mind Institute: Deep Dive Model Documentation

This document provides a granular analysis of every model used in the hybrid architecture, detailing the specific datasets, mechanisms, and technical refinements that drove the final accuracy.

---

## 1. Age-Gated Biological Dynamic Hebbian (BDH) Model
**Type**: Spiking Graph Transformer (Custom PyTorch Module)
**Dataset Used**: 
*   **Input**: Raw Actigraphy Time-Series (`series_train.parquet`). 5 Channels: `X`, `Y`, `Z`, `enmo`, `anglez`. Resampled to 230 steps.
*   **Gating**: `Basic_Demos-Age` (Bucketed: <10, 10-15, 16+).

**Mechanism**:
*   **Virtual Expansion**: The 5 physical sensors are projected to 64 "Virtual Neurons".
*   **Hebbian Plasticity**: Synaptic weights ($W$) update dynamically based on the outer product of co-activations: $W_{t+1} = \alpha W_t + \eta (x_t x_t^T)$.
*   **Memory Graph**: The final state $W_{final}$ represents the "fingerprint" of the subject's movement patterns.

**Minute Details & Accuracy Drivers**:
*   **Refinement: Age Gating**: Initially, the model treated a 5-year-old's "sedentary" signal the same as a 17-year-old's. By implementing an `Age-Gated Initialization` (3 distinct learned prototype matrices), the model started from a context-aware prior. This drastically reduced noise in the early training epochs.
*   **Refinement: Linear Attention**: Replaced standard Self-Attention ($O(T^2)$) with Recurrent Linear Attention ($O(T)$), allowing us to process longer sequences (230 steps) without OOM errors, capturing minutes-long behavioral motifs instead of just seconds.

---

## 2. Sparse Autoencoder (SAE)
**Type**: Neural Network (Encoder-Decoder)
**Dataset Used**: 
*   **Input**: Flattened BDH Synaptic Graphs (4096 dimensions).

**Mechanism**:
*   **Compression**: Maps the 4096 synaptic connections to a 32-dimensional Latent Space.
*   **Sparsity Constraint**: Uses L1 regularization on the latent activations to force the network to learn "monosemantic" features (e.g., a specific feature might strictly represent "Late Night High-Activity").

**Minute Details & Accuracy Drivers**:
*   **Refinement: Ditching PCA**: The initial plan used PCA (Linear). PCA smeared distinct behavioral clusters into a blurry average. The SAE (Non-Linear + Sparse) preserved the "outlier" signals characteristic of Severe PCIAT cases ($sii=3$), which PCA filtered out as noise.
*   **Impact**: Rapid convergence (Loss < 0.001) proved the synaptic graphs have a highly structured low-dimensional manifold.

---

## 3. Synaptic KNN Imputer
**Type**: Non-Parametric Algorithm
**Dataset Used**: 
*   **Input**: SAE Latent Features + Tabular Data (`train.csv`).

**Mechanism**:
*   **Cross-Modal Inference**: Finds subjects with similar "Synaptic Fingerprints" (Cosine Similarity of SAE features).
*   **Logic**: If Subject A (missing BMI) moves exactly like Subject B (has BMI), predict A's BMI using B's.

**Minute Details & Accuracy Drivers**:
*   **Refinement: Preventing Mean-Regression**: Standard Mean Imputation pulls predictions toward the average (Healthy/$sii=0$). This makes predicting the rare "Severe" class impossible. Synaptic Imputation preserved the "pathological" correlations, allowing the downstream boosters to see "High BMI + Chaotic Sleep" patterns even when data was missing.

---

## 4. Ensemble Boosters (LightGBM, XGBoost, CatBoost)
**Type**: Gradient Boosting Decision Trees
**Dataset Used**: 
*   **Input**: Imputed Tabular Features + SAE Latent Synaptic Features (32 dims).
*   **Target**: `PCIAT-PCIAT_Total`.

**Mechanism**:
*   **Gradient Boosting**: Sequentially builds trees to correct the errors of previous trees.
*   **Hybrid Feature Space**: Splits nodes based on both "Biological" signals (Weight, Age) and "Neural" signals (Synaptic Feature #4).

**Minute Details & Accuracy Drivers**:
*   **Refinement: QWK Optimization**: Switched evaluation from pure MSE to **Quadratic Weighted Kappa (QWK)** awareness.
*   **Refinement: Anti-Leakage**: Explicitly removed `sii` and `PCIAT` columns from the input. The previous iteration had unrealistically low MSE (0.14) due to leakage; the current MSE (~0.91) is a true reflection of generalization.
*   **Ensemble Power**: Averaging the three models smoothed out the variance caused by the small dataset size (~2700 samples), providing a robust final prediction.
