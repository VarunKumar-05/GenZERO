# SynaptoRehab: Track 2 Submission Guide

## Solution Overview
SynaptoRehab is a frontier bio-behavioral monitoring system built on the **Baby Dragon Hatchling (BDH)** architecture. It leverages **Hebbian plasticity** to learn patient-specific baselines and **Linear Attention** for efficient processing of multi-modal streams.

### Key Features
1.  **Multi-Modal Bridge**: Ingests Audio (Librosa) and Text (CHAT transcripts) to form a unified temporal stream.
2.  **Hebbian Synapses**: Weights update dynamically based on input correlations ($O(T)$ learning), allowing the model to "remember" recovery states without retraining.
3.  **Concept Probes**: Specific neural circuits are monitored to detect "Tremor", "Stutter", and "Fatigue" states.
4.  **Clinical Dashboard**: A Streamlit interface visualizing neural sparsity (5% activation target) and emergent synaptic topology.

## Repository Structure
- `src/bdh_snn/network.py`: Core SNN with `HebbianSynapse` and `LinearAttention`.
- `src/analysis/detector.py`: Implements `ConceptProbes` for anomaly detection.
- `src/clinical/dashboard.py`: Visualization logic for the "Brain Pulse".
- `src/input/`: Data loaders for Audio and Text.
- `app.py`: Interactive Streamlit dashboard.
- `train_dummy.py`: Script to verify the training loop and pipeline.

## Execution Plan

### Phase 1: Environment Setup
Ensure all dependencies are installed:
```bash
pip install -r Graph-pictorial/requirements.txt
```

### Phase 2: Data Processing
The system is designed to handle TalkBank and DAIC-WOZ formats.
- Audio is processed for MFCC, Zero-Crossing Rate, and RMS.
- Text is vectorized from CHAT transcripts.

### Phase 3: Training & Probing
Run the verification loop:
```bash
python train_dummy.py
```
This script simulates the flow of data through the SNN, triggering Hebbian updates and verifying output shapes.

### Phase 4: Dashboard & Visualization
Launch the interactive dashboard:
```bash
streamlit run app.py
```
Use the sidebar to simulate different patient states ("Tremor Onset", "Speech Disfluency") and observe how the **Synaptic Topology** graph evolves and how specific **Concept Probes** light up in the Clinical Report.

## "The Path C Edge"
The `ConceptProbes` class in `src/analysis/detector.py` is the key differentiator. It demonstrates that the model isn't just a black box but has interpretable "circuits" for specific medical symptoms, fulfilling the "rigor and novelty" criteria.
