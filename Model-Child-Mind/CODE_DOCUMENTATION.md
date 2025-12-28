# Child Mind Model: ROI & Code Documentation

This document provides a detailed technical reference for the codebase located in `Model-Child-Mind/`.

## Directory Structure
```
Model-Child-Mind/
├── train_hybrid.py          # Master training pipeline
├── documentation.md         # High-level architecture overview
├── src/
│   ├── data_loader/
│   │   └── child_mind.py    # Dataset & Dataloader
│   ├── model/
│   │   ├── bdh.py           # Core BDH Layer
│   │   ├── bdh_features.py  # Age-Gated Feature Extractor
│   │   ├── sae.py           # Sparse Autoencoder
│   │   └── imputation.py    # Synaptic KNN Imputer
│   └── analysis/
│       └── viz_child_mind.py # 3D Graph Visualization
└── data/                    # Dataset directory
```

---

## 1. Training Pipeline
### `train_hybrid.py`
The entry point for training the complete hybrid model.
*   **Workflow**:
    1.  **BDH Training**: Pre-trains the Spiking Graph Transformer for 2 epochs to shape synaptic manifolds.
    2.  **Feature Extraction**: Runs the full dataset through BDH to generate 4096-dim synaptic vectors.
    3.  **SAE Training**: Trains the Sparse Autoencoder to compress BDH features (4096 $\to$ 32).
    4.  **Imputation**: Uses `SynapticKNNImputer` to fill missing values in tabular data based on latent synaptic similarity.
    5.  **Booster Training**: Trains LightGBM, XGBoost, and CatBoost on the combined (Imputed Tabular + Latent Synaptic) features.
    6.  **Evaluation**: Calculates Mean Squared Error (MSE) and Quadratic Weighted Kappa (QWK).
*   **Key Functions**: `train_bdh_extractor`, `extract_features`, `train_boosters`, `main`.

---

## 2. Data Loading
### `src/data_loader/child_mind.py`
**Class**: `ChildMindDataset`
*   **Purpose**: Handles multi-modal data loading (Tabular + Time-Series).
*   **Key Features**:
    *   **Lazy Loading**: Reads Parquet files on-the-fly in `__getitem__`.
    *   **Age Gating**: Extracts `Basic_Demos-Age`, buckets it into groups (<10, 10-15, 16+), and returns `age_group` index for the model.
    *   **Data Leakage Prevention**: Explicitly excludes target columns (`PCIAT-PCIAT_Total`, `sii`) from input features.
    *   **Raw Return**: Returns raw values (with NaNs) to allow for downstream Synaptic Imputation.

---

## 3. Core Models
### `src/model/bdh.py`
**Class**: `BDHLayer`
*   **Concept**: Biological Dynamic Hebbian Layer. A Spiking Graph Transformer.
*   **Mechanism**:
    *   **Linear Attention**: $O(T)$ complexity recurrent update.
    *   **Hebbian Update**: $W_{t+1} = \alpha W_t + \eta (x_t x_t^T)$.
    *   **Dynamics**: Synaptic weights evolve based on co-activation of virtual neurons.

### `src/model/bdh_features.py`
**Class**: `BDHFeatureExtractor`
*   **Purpose**: Wraps `BDHLayer` for end-to-end learning from Actigraphy.
*   **Key Features**:
    *   **Projection**: Maps 5 physical sensors (x,y,z,enmo,anglez) $\to$ 64 Virtual Neurons.
    *   **Age-Gated Initialization**: Maintains a bank `w_init_bank` of shape `(Num_Ages, 64, 64)`. The starting brain state is selected based on the subject's age group.
*   **Output**: Flattened Synaptic Matrix (4096 dims) + Mean Activity.

### `src/model/sae.py`
**Class**: `SparseAutoencoder`
*   **Purpose**: Compresses the high-dimensional synaptic graph while preserving non-linear structures.
*   **Architecture**: `Linear(4096 -> 128) -> ReLU -> Linear(128 -> 32) -> ReLU -> Decoder`.
*   **Loss**: MSE + L1 Penalty (on latent activations) to enforce sparsity.

### `src/model/imputation.py`
**Class**: `SynapticKNNImputer`
*   **Concept**: Cross-Modal Imputation.
*   **Logic**: "If two subjects move identically (similar Synaptic Latents), their physiological stats are likely similar."
*   **Method**:
    1.  Calculates Cosine Similarity on `SAE Latents`.
    2.  Finds $k$ Nearest Neighbors.
    3.  Imputes missing tabular values using weighted average of neighbors.

---

## 4. Analysis
### `src/analysis/viz_child_mind.py`
**Function**: `visualize_synaptic_graph`
*   **Purpose**: Visual Interpretablity.
*   **Method**:
    1.  Runs inference on a sample.
    2.  Extracts the 64x64 adjacency matrix ($W_{final}$).
    3.  Thresolds weak connections.
    4.  Uses `networkx.spring_layout` (3D) to determine node positions based on connectivity strength.
    5.  Renders using Matplotlib 3D.
