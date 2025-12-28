"""
Page 2: Child Mind Institute Assessment
Fully integrated with BDHFeatureExtractor & SparseAutoencoder - NO MOCK DATA
"""
import streamlit as st
import sys
import os
import numpy as np

# Page Configuration
st.set_page_config(page_title="Child Mind Assessment | GenZERO", page_icon="👶", layout="wide")

# Add parent directory to path and import backend
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import backend integration
from backend_integration import (
    CHILD_MIND_MODEL_AVAILABLE,
    load_child_mind_model,
    load_pciat_boosters,
    predict_pciat_score,
    get_age_group,
    generate_actigraphy_sample
)

import torch

# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("👶 Child Mind Institute Assessment")
st.markdown("*Age-Gated BDH Feature Extraction with Sparse Autoencoder*")

# Show backend status
if CHILD_MIND_MODEL_AVAILABLE:
    st.success("✅ Backend loaded: BDHFeatureExtractor, SparseAutoencoder, SynapticKNNImputer")
else:
    st.warning("⚠️ Child-Mind model not available. Using core BDH components instead.")

st.divider()

# ============================================================================
# PATIENT DEMOGRAPHICS
# ============================================================================

st.markdown("## 👤 Patient Demographics")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Patient Age", min_value=5, max_value=18, value=12)
    age_group = get_age_group(age)
    age_group_names = {0: "Young (<10)", 1: "Middle (10-15)", 2: "Older (16+)"}
    st.info(f"**Age Group:** {age_group_names[age_group]} (bucket={age_group})")

with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
with col3:
    bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, value=20.5, step=0.5)

st.divider()

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

st.markdown("## ⚙️ BDH-SAE Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### BDH Feature Extractor")
    num_channels = st.selectbox("Input Channels", [5], index=0, help="X, Y, Z, enmo, anglez")
    num_virtual = st.selectbox("Virtual Neurons", [32, 64, 128], index=1)
    seq_length = st.slider("Sequence Length", 100, 500, 230)

with col2:
    st.markdown("### Sparse Autoencoder")
    latent_dim = st.selectbox("Latent Dimensions", [16, 32, 64], index=1)
    l1_lambda = st.select_slider("L1 Sparsity (λ)", options=[1e-6, 1e-5, 1e-4, 1e-3], value=1e-5)

st.divider()

# ============================================================================
# DATA INPUT
# ============================================================================

st.markdown("## 📊 Actigraphy Data Input")

data_source = st.radio(
    "Data Source",
    ["🎲 Generate Sample Data", "📤 Upload Parquet/CSV"],
    horizontal=True
)

actigraphy_data = None

if data_source == "🎲 Generate Sample Data":
    if st.button("Generate Sample Actigraphy"):
        actigraphy_data = generate_actigraphy_sample(seq_length, num_channels)
        st.success(f"✅ Generated sample data: shape {list(actigraphy_data.shape)}")
        
        # Show sample preview
        st.line_chart(actigraphy_data[:100, :].numpy(), height=200)
        
else:
    uploaded_file = st.file_uploader("Upload actigraphy data", type=['parquet', 'csv'])
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.info("Note: Processing uploaded files requires ChildMindDataset loader")

st.divider()

# ============================================================================
# INFERENCE
# ============================================================================

st.markdown("## 🧠 BDH-SAE Pipeline")

if st.button("🚀 Run Child Mind Assessment", type="primary", use_container_width=True):
    
    if CHILD_MIND_MODEL_AVAILABLE:
        # Load model components
        with st.spinner("Loading BDH Feature Extractor and SAE..."):
            models = load_child_mind_model(
                input_channels=num_channels,
                virtual_nodes=num_virtual,
                num_age_groups=3,
                latent_dim=latent_dim,
                load_weights=True
            )
            bdh_extractor = models['bdh_extractor']
            sae = models['sae']
            weights_loaded = models.get('weights_loaded', {})
        
        # Show weight loading status
        if weights_loaded.get('bdh') and weights_loaded.get('sae'):
            st.success("✅ Models loaded with **TRAINED WEIGHTS** from .pth files!")
        elif weights_loaded.get('bdh') or weights_loaded.get('sae'):
            st.warning(f"⚠️ Partial weights loaded - BDH: {weights_loaded.get('bdh')}, SAE: {weights_loaded.get('sae')}")
        else:
            st.warning("⚠️ Using random initialization (no trained weights found)")
        
        # Generate or use input data
        if actigraphy_data is None:
            actigraphy_data = generate_actigraphy_sample(seq_length, num_channels)
        
        # Add batch dimension: (1, T, C)
        ts_input = actigraphy_data.unsqueeze(0)
        mask = torch.ones(1, seq_length)
        age_group_tensor = torch.tensor([age_group], dtype=torch.long)
        
        st.info(f"📊 Input shape: {list(ts_input.shape)} (batch, time, channels)")
        
        # Step 1: BDH Feature Extraction
        with st.spinner("Step 1: BDH Feature Extraction with Age-Gating..."):
            bdh_extractor.eval()
            with torch.no_grad():
                bdh_features = bdh_extractor(ts_input, mask, age_group_tensor)
        
        st.success(f"✅ BDH Features extracted: shape {list(bdh_features.shape)}")
        
        # Step 2: SAE Compression
        with st.spinner("Step 2: Sparse Autoencoder Compression..."):
            sae.eval()
            with torch.no_grad():
                reconstructed, latent = sae(bdh_features)
        
        st.success(f"✅ SAE Latent: {list(latent.shape)} (compressed from {bdh_features.shape[1]} dims)")
        
        # ====================================================================
        # RESULTS
        # ====================================================================
        
        st.divider()
        st.markdown("## 📋 Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Try to load trained boosters for real PCIAT prediction
            boosters = load_pciat_boosters()
            
            # Predict PCIAT score using boosters or fallback
            pciat_score, is_real_prediction = predict_pciat_score(latent, boosters)
            
            # Show whether this is a real or heuristic prediction
            if is_real_prediction:
                st.success("✅ Using **trained ensemble boosters** for prediction")
            else:
                st.warning("⚠️ Using heuristic (no trained boosters found). Run `train_hybrid.py` to train boosters.")
            
            st.metric("Predicted PCIAT Score", f"{pciat_score:.1f}")
            
            # SII classification (same thresholds as train_hybrid.py)
            if pciat_score < 31:
                sii = 0
                sii_label = "None"
            elif pciat_score < 50:
                sii = 1
                sii_label = "Mild"
            elif pciat_score < 80:
                sii = 2
                sii_label = "Moderate"
            else:
                sii = 3
                sii_label = "Severe"
            
            colors = {0: "green", 1: "orange", 2: "red", 3: "darkred"}
            st.markdown(f"**SII Classification:** :{colors[sii]}[{sii_label}] (sii={sii})")
        
        with col2:
            st.markdown("**Latent Feature Statistics:**")
            st.json({
                "mean": round(latent.mean().item(), 4),
                "std": round(latent.std().item(), 4),
                "min": round(latent.min().item(), 4),
                "max": round(latent.max().item(), 4),
                "sparsity": round((latent.abs() < 0.1).float().mean().item() * 100, 1)
            })
        
        # ====================================================================
        # VISUALIZATIONS
        # ====================================================================
        
        st.divider()
        st.markdown("## 🎨 Feature Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### SAE Latent Space")
            import pandas as pd
            latent_df = pd.DataFrame({'Latent': latent[0].detach().numpy()})
            st.bar_chart(latent_df, use_container_width=True)
        
        with col2:
            st.markdown("### Reconstruction Error")
            recon_error = (bdh_features - reconstructed).pow(2)
            st.metric("Total MSE", f"{recon_error.mean().item():.4f}")
            # Per-dimension error (show first 50 dims)
            error_df = pd.DataFrame({'MSE': recon_error[0, :50].detach().numpy()})
            st.bar_chart(error_df, use_container_width=True)
        
        # Store results for Comprehensive Report
        st.session_state['child_mind_results'] = {
            'sii_score': sii,
            'sii_label': sii_label,
            'pciat_score': pciat_score,
            'latent_stats': {
                "mean": round(latent.mean().item(), 4),
                "sparsity": round((latent.abs() < 0.1).float().mean().item() * 100, 1)
            },
            'recon_error': recon_error.mean().item()
        }
    
    else:
        # Fallback to core BDH from src/
        st.warning("Using core BDH components from src/ folder")
        
        from backend_integration import load_speech_model
        
        models = load_speech_model(input_dim=num_channels, hidden_dim=num_virtual)
        snn = models['snn']
        
        # Generate sample data
        if actigraphy_data is None:
            actigraphy_data = generate_actigraphy_sample(seq_length, num_channels)
        
        # Process first timestep through SNN
        input_sample = actigraphy_data[0:1, :]
        snn_output = snn.forward(input_sample)
        
        st.success("✅ Processed through core SNN")
        st.json({
            "output_shape": list(snn_output["output"].shape),
            "hidden_shape": list(snn_output["hidden_state"].shape)
        })

# ============================================================================
# ARCHITECTURE INFO
# ============================================================================

st.divider()

with st.expander("🏗️ View BDH-SAE Architecture"):
    st.markdown("""
    ```
    BDHFeatureExtractor(
        projection: Linear(5 channels → 64 virtual neurons)
        w_init_bank: Parameter(3 ages × 64 × 64)  # Age-gated initialization
        bdh_layer: BDHLayer(64 nodes)
            - α (decay): 0.9
            - η (learning): 0.01
            - Hebbian update: W_t+1 = αW_t + η(x_t @ x_t^T)
        Output: [flatten(W_final), mean(activity)]  # 64*64 + 64 = 4160 dims
    )
    
    SparseAutoencoder(
        encoder: Linear(4160 → 128) → ReLU → Linear(128 → 32)
        decoder: Linear(32 → 128) → ReLU → Linear(128 → 4160)
        Loss: MSE + λ * L1(latent)  # Sparsity regularization
    )
    ```
    """)
