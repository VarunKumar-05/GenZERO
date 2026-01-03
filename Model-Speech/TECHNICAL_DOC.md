# Model-Speech (SynaptoRehab): Technical Architecture Document

## 1. System Overview
**Model-Speech** (codenamed *SynaptoRehab*) is an **Unsupervised Spiking Neural Network (SNN)** designed for **Bio-Behavioral Anomaly Detection**. Unlike standard classifiers that Map $X \to Y$, this system monitors the temporal evolution of brain-like states to detect neurological deviations (Tremors, Disfluency, Fatigue) in real-time.

### 1.1 Core Philosophy
The model operates on the principle of **"Comparison to Self"**. It does not have hardcoded thresholds for what constitutes a "tremor." Instead, it learns a baseline of the user's normal activity and flags statistically significant deviations.

---

## 2. Architecture Breakdown

### A. Input Processing Layer
The system processes multi-modal data frame-by-frame:
1.  **Acoustic Features**: eGeMAPS (88 dimensions) - Pitch, Jitter, Shimmer, Loudness.
2.  **Semantic Features**: Word Embeddings/Counts from transcripts.
3.  **Normalization**: A `LayerNorm` (128-dim) standardizes these mixed signals to ensure stability before entering the SNN.

### B. The BDH Spiking Core (`SpikingNeuralNetwork`)
The heart of the system is a recurrent SNN with **Hebbian Plasticity**.

*   **Structure**: 128 Input Neurons $\to$ 256 Hidden Neurons (Recurrent).
*   **Dynamics**:
    $$ h_t = \text{ReLU}(W_{in} x_t + 0.1 \cdot W_{syn} h_{t-1}) $$
    The hidden state $h_t$ represents the current "neural activation pattern."

### C. Stabilized Hebbian Learning
To prevent the "weight explosion" common in unsupervised learning, we implemented a **Decay-Based Hebbian Rule**:

$$ W_{new} = \alpha \cdot W_{old} + \eta \cdot (h_t \otimes h_t^T) $$

*   **Decay Factor ($\alpha = 0.99$)**: This "forgetting factor" ensures that old, irrelevant correlations fade over time, keeping the weights bounded.
*   **Soft Clamping**: If any weight exceeds a magnitude of 5.0, it is scaled down. This ensures numerical stability over long sessions.

### D. Calibrated Concept Probes (`CalibratedDetector`)
Instead of training a classifier, we use **Z-Score Anomaly Detection** on specific neuron clusters ("Concepts").

1.  **Calibration Phase**: For the first 200 frames ($\sim$6 seconds), the system does **not** detect errors. Instead, it computes the Mean ($\mu$) and Standard Deviation ($\sigma$) of every neuron's firing rate.
2.  **Detection Phase**: For all subsequent frames, we calculate the Z-Score:
    $$ Z = \frac{h_t - \mu}{\sigma} $$
3.  **Trigger Logic**:
    *   **Tremor / Hyperactivity**: If selected neurons exceed $Z > 2.0$ (2 Sigma event).
    *   **Fatigue / Hypoactivity**: If selected neurons fall below $Z < -1.5$.

---

## 3. Key Advantages

### 1. Zero-Shot Adaptation
The model does not need to be pre-trained on a specific patient's data. Because of the **Dynamic Calibration**, it adapts to *any* user within 6 seconds. If a user naturally speaks loudly, the baseline $\mu$ shifts up, and the system correctly ignores it, only flagging *relative* spikes.

### 2. Clinical Interpretability
The Learned Synapses ($W_{syn}$) capture word-state correlations. By inspecting the weights, we found the model automatically learned that:
*   **"uh" / "um"** $\to$ Strongly excitatory connections to the **Disfluency Concept**.
*   **"just" / "so"** $\to$ Weak connections (fillers).

This confirms the unsupervised learning successfully identified linguistic markers of hesitation without being explicitly told what "stuttering" looks like.

### 3. Lightweight & Real-Time
The entire architecture uses simple matrix multiplications and requires no backpropagation during inference. It can run on edge devices (phones/tablets) with negligible latency.
