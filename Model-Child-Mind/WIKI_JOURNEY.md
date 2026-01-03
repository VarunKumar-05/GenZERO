# Project Journey: The Quest for Synaptic Behavioral Modeling

## The Objective
To build a **Hybrid Biological Dynamic Hebbian (BDH) Model** capable of decoding the subtle actigraphy patterns in children to predict Problematic Internet Use (PIU). The goal was not just prediction, but *understanding*—mapping movement to "synaptic memory."

## The Adventure

### Phase 1: The Foundation
We started by constructing the **Age-Gated BDH Feature Extractor**. We hypothesized that a child's movement isn't just a time-series; it's a language. By introducing age-gated initialization, we allowed the model to interpret "stillness" differently for a 5-year-old vs. a 15-year-old.

### Phase 2: The Bottleneck
We faced the "Curse of Dimensionality." The raw synaptic graphs were too complex (4096 dimensions).
*   **Attempt 1**: PCA. *Result*: Failed to capture non-linear behavioral clusters.
*   **The Solution**: We forged a **Sparse Autoencoder (SAE)**. This compressed the graph into 32 powerful latent features, acting as the "behavioral DNA" of each subject.

## The Hurdles & The Fixes

### 1. The Chaos of Inconsistency
**The Error**: "Why is the output not consistent?"
We discovered that our model was living in a chaotic universe. Every training run yielded wildly different QWK scores (ranging from 0.4 to 0.6) because specific random seeds were not locked. The boosters (LightGBM, XGBoost) and the Neural Networks were initializing differently every time.

**The Fix**: **The Ritual of Seeding**
We implemented a strict `seed_everything(42)` protocol:
*   Locked Python, NumPy, and PyTorch (CPU & CUDA) seeds.
*   Forced deterministic algorithms in CuDNN.
*   Hard-coded `random_state` in all Gradient Boosting decision trees.
*   *Result*: Absolute reproducibility. Run A = Run B.

### 2. The Illusion of Validity (Data Leakage)
**The Error**: "Model performs suspicious well but might be memorizing."
We realized we were evaluating the model on the same data it trained on. This gave us a false sense of security.

**The Fix**: **The Great Split**
We implemented a rigorous `train_test_split` (80/20).
*   We now train strictly on the Training Set.
*   We evaluate *only* on the unseen Test Set.
*   *Result*: Honest metrics. We now know the model's true generalization power.

### 3. The Void (Missing Data)
**The Challenge**: The tabular data was full of holes ("Swiss Cheese" dataset).
**The Fix**: **Synaptic Imputation**
Instead of guessing with means, we used the *Actigraphy* features to fill the gaps. If Child A moves exactly like Child B (high, energetic synaptic similarity), we infer that Child A likely has similar physiological stats to Child B.

## The Roadmap Ahead
- [x] **Reproducibility Code**: Locked and Sealed.
- [x] **Validation Strategy**: Robust Train/Test Split.
- [ ] **Hyperparameter Tuning**: Now that we have a stable baseline, we can tune.
- [ ] **Deployment**: Integration into the final inference pipeline.
