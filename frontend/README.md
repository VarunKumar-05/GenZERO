# GenZERO Frontend

A comprehensive Streamlit-based frontend for the GenZERO Bio-Behavioral Monitoring System.

## 🚀 Quick Start

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📁 Structure

```
frontend/
├── app.py                     # Main application entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── pages/
    ├── 1_🎤_Speech_Analysis.py    # Speech emotion recognition
    ├── 2_👶_Child_Mind.py          # Child mental health assessment
    ├── 3_🧬_Psychosis_Detection.py # Schizophrenia/Bipolar classification
    ├── 4_📊_Clinical_Dashboard.py  # Visualization & reports
    └── 5_⚙️_Settings.py            # System configuration
```

## 🧠 Features

### 1. Speech Emotion Recognition
- Real-time neural simulation with configurable patient states
- Audio file upload and analysis (WAV, MP3, FLAC)
- DAIC-WOZ dataset session processing
- Concept probe visualization (Tremor, Stutter, Fatigue)

### 2. Child Mind Assessment
- Age-gated BDH feature extraction
- Actigraphy data processing (5 channels: X, Y, Z, enmo, anglez)
- PCIAT score prediction with ensemble boosters
- SII classification (None/Mild/Moderate/Severe)

### 3. Psychosis Detection
- rs-fMRI connectome analysis (105 ICN regions)
- BDH network training with Hebbian plasticity
- Single-subject inference for SZ/BP classification
- Visualization tools for functional connectivity

### 4. Clinical Dashboard
- Comprehensive metrics overview
- Clinical report generation (PDF/HTML/Markdown)
- Neural activation sparsity visualization
- Synaptic topology graphs

### 5. System Settings
- BDH parameter configuration (α, η, hidden dimensions)
- Sparse Autoencoder settings
- Concept probe thresholds
- Model weight management

## 🏗️ Architecture

This frontend integrates with the GenZERO backend components:

- **BDH Spiking Neural Network** - Hebbian plasticity with linear attention
- **Sparse Autoencoder** - Dimensionality reduction with L1 sparsity
- **Concept Probes** - Interpretable clinical concept detection
- **Clinical Dashboard** - Real-time monitoring and reporting

## 📊 Model Tracks

| Track | Input | Output |
|-------|-------|--------|
| Speech | Audio (MFCC, ZCR, RMS) | Emotion/Clinical indicators |
| Child Mind | Actigraphy (5 channels) | PCIAT Score, SII Class |
| Psychosis | rs-fMRI (FNC + ICN) | SZ/BP Classification |

## 🎨 UI Features

- Dark mode optimized design
- Glassmorphism aesthetics
- Responsive layout
- Real-time progress indicators
- Interactive visualizations

## 📝 License

Part of the GenZERO project - Bio-Behavioral Intelligence Platform.
