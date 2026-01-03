# Model-Physcosis: Technical Architecture Document

## 1. Architecture Overview (MultiHeadBDHNet)
The **Model-Physcosis** is built upon the **MultiHead Baby Dragon Hatchling (BDH) Network**, a specialized Spiking Neural Network (SNN) designed for temporal graph classification. Unlike traditional RNNs or Transformers, it leverages **Hebbian Plasticity** to "learn" the connectivity of the brain dynamically as it processes an fMRI scan.

### 1.1 Core Components
The model consists of three distinct computational blocks:

1.  **Multi-Head BDH Layer (`MultiHeadBDHLayer`)**: The core plasticity engine.
2.  **Trajectory Attention Pooling (`TrajectoryPooling`)**: A dynamic temporal aggregator.
3.  **Fused Classification Head**: A dense layer combining structural and dynamic features.

---

## 2. In-Depth Component Analysis

### A. Multi-Head Plasticity Layer
Instead of fixed weights, the network maintains **Evolving Synaptic Weights** ($W_t$) that change for every time step $t$ in the fMRI sequence.

*   **Input**:
    *   `x`: Spike train sequence $(B, T, N)$, where $N=105$ (Brain Regions).
    *   `w_init`: Initial Functional Network Connectivity (FNC) matrix $(B, N, N)$.
*   **Mechanism**:
    The layer operates **4 Parallel Plasticity Heads**, each with a distinct **Decay Rate ($\alpha$)**:
    *   **Head 0 ($\alpha=0.9$)**: Long-term memory (stable connectivity).
    *   **Head 1 ($\alpha=0.7$)**: Medium-term adaptation.
    *   **Head 2 ($\alpha=0.5$)**: Short-term adaptation.
    *   **Head 3 ($\alpha=0.1$)**: Rapid transient response.

    **Hebbian Update Rule**:
    For each head $h$, the synaptic weight matrix $W^{(h)}$ updates as:
    $$ W^{(h)}_{t+1} = \alpha_h \cdot W^{(h)}_t + \eta \cdot (x_t \otimes x_t^T) $$
    
    *   $\alpha_h$: The learnable decay factor for head $h$.
    *   $\eta$: The learnable learning rate (plasticity coefficient).
    *   $x_t \otimes x_t^T$: The outer product of activations (Hebb's rule: *neurons that fire together, wire together*).

*   **Output**:
    *   **Final Weights ($W_{final}$)**: A tensor of shape $(B, 4, 105, 105)$ representing the *learned brain topology* after viewing the entire scan.
    *   **Output Sequence ($y_{seq}$)**: A sequence $(B, T, 4, 105)$ representing the dynamic brain states.

### B. Trajectory Attention Pooling
To capture the temporal dynamics of the "thought process" (trajectory), we use an attention mechanism rather than simple averaging.

*   **Input**: The sequence of hidden states $y_{seq}$.
*   **Mechanism**:
    1.  **Query ($Q$)**: A global learnable vector representing "what important moments look like."
    2.  **Key/Value ($K, V$)**: Projections of the sequence $y_{seq}$.
    3.  **Attention**: Computes a weighted sum of all time steps based on relevance to the Query.
    $$ \text{Context} = \text{Softmax}(\frac{Q K^T}{\sqrt{d}}) V $$
*   **Advantage**: This allows the model to focus on specific moments of high synchronous activity (e.g., a sudden burst of connectivity) while ignoring noise.

### C. Feature Fusion
The model combines two complementary views of the data for classification:
1.  **Structural View**: The flattened final weights ($W_{final}$), representing the **"Brain Wiring Diagram"** learned over the session.
2.  **Dynamic View**: The pooled trajectory context, representing the **"Flow of Activity"**.

These are concatenated and passed to a dense classifier:
$$ \text{Logits} = \text{MLP}(\text{Concat}(\text{Flatten}(W_{final}), \text{Pooled}(y_{seq}))) $$

---

## 3. Utilizing the BDH Advantage

The model exploits the BDH architecture to solve the specific challenges of fMRI analysis:

### 1. solving the Non-Stationarity Problem
**Challenge**: Brain connectivity is not static; it changes rapidly (seconds) vs slowly (minutes).
**BDH Advantage**: By using **Multi-Head Plasticity with diverse $\alpha$ values**, the model captures **Multi-Scale Temporal Dynamics**.
*   High $\alpha$ heads capture the stable "traits" of the subject (e.g., Schizophrenia baseline).
*   Low $\alpha$ heads capture the transient "states" (e.g., momentary thought patterns).

### 2. Solving the Data Scarcity Problem
**Challenge**: fMRI datasets are small (hundreds of samples) but high-dimensional.
**BDH Advantage**: **Self-Supervised Hebbian Learning**.
The effective parameters (weights) are not learned via Gradient Descent but are *generated* by the data itself via the Hebbian rule. The model only learns the *hyperparameters* ($\alpha, \eta$) governing this generation. This makes the model extremely **Data-Efficient** and resistant to overfitting compared to an LSTM or Transformer with millions of fixed parameters.

### 3. Solving the Interpretability Problem
**Challenge**: Deep learning models are black boxes.
**BDH Advantage**: **Explicit Topology**.
The final output $W_{final}$ is literally a connectivity matrix. We can inspect this matrix to see exactly which brain regions strengthened their connections during the scan. For example, if Regions A and B have a high weight in $W_{final}$, it means they fired synchronously throughout the session.

---

## 4. Performance Metrics

### Validated Accuracy
Our final model, trained with **Stratified Splitting** and **Weighted Cross-Entropy Loss** to handle class imbalance, achieves:

*   **Overall Validation Accuracy**: **62.11%** (Macro-Average: 62.12%)
*   **Class-Wise Performance**:
    *   **Bipolar Disorder (BP)**: 62.2% Recall, 51.1% Precision.
    *   **Schizophrenia (SZ)**: 62.1% Recall, 72.0% Precision.
*   **Confusion Matrix (Validation)**:
    *   **BP**: 23 Correct / 14 Missed
    *   **SZ**: 36 Correct / 22 Missed

**Significance**: Unlike previous iterations that achieved ~55% by guessing the majority class (SZ), this model is **perfectly balanced**, demonstrating a true ability to discriminate between the two conditions despite the noise and complexity of fMRI data.
