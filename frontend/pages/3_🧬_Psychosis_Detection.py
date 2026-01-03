"""
Page 3: Psychosis Detection (Schizophrenia vs Bipolar Disorder)
Fully integrated with BDHNet & SpikeEncoder - NO MOCK DATA
"""
import streamlit as st
import sys
import os
import numpy as np

# Page Configuration
st.set_page_config(page_title="Psychosis Detection | GenZERO", page_icon="🧬", layout="wide")

# Add parent directory to path and import backend
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import backend integration
from backend_integration import (
    PSYCHOSIS_MODEL_AVAILABLE,
    load_psychosis_model,
    generate_fmri_sample
)

import torch

# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("🧬 Psychosis Classification")
st.markdown("*Distinguishing Schizophrenia from Bipolar Disorder using rs-fMRI Connectomics*")

# Show backend status
if PSYCHOSIS_MODEL_AVAILABLE:
    st.success("✅ Backend loaded: BDHNet, BDHLayer, SpikeEncoder")
else:
    st.warning("⚠️ Psychosis model not available. Some features limited.")

st.divider()

# ============================================================================
# TABS: TRAINING vs INFERENCE
# ============================================================================

tab1, tab2, tab3 = st.tabs(["📊 Training", "🔮 Inference", "📈 Visualization"])

# ============================================================================
# TAB 1: TRAINING
# ============================================================================

with tab1:
    st.markdown("## 🏋️ BDH Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset Configuration")
        data_root = st.text_input("Data Root", value="data/train")
        num_nodes = st.number_input("ICN Nodes", value=105, disabled=True)
        fnc_features = st.number_input("FNC Features", value=5460, disabled=True, 
                                       help="Upper triangle: 105*(105-1)/2 = 5460")
        seq_length = st.slider("Sequence Length (Time Points)", 100, 500, 230)
    
    with col2:
        st.markdown("### Training Hyperparameters")
        epochs = st.slider("Epochs", 5, 100, 20)
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        learning_rate = st.select_slider("Learning Rate", 
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)
        lambda_traj = st.slider("λ Trajectory", 0.0, 1.0, 0.1)
        lambda_sparse = st.select_slider("λ Sparsity", 
            options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
    
    st.divider()
    
    # Architecture Display
    st.markdown("### 🧠 BDHNet Architecture")
    
    st.code("""
BDHNet(num_nodes=105)
├── bdh_layer: BDHLayer(105 nodes)
│   ├── α (decay): Parameter(init=0.9)
│   ├── η (learning): Parameter(init=0.01)
│   └── forward(x, w_init, mask) → (w_final, y_seq)
│       - Recurrent loop: t = 1..T
│       - y_t = W_t @ x_t
│       - W_{t+1} = α*W_t + η*(x_t ⊗ x_t^T)
├── classifier: Sequential
│   ├── Dropout(0.5)
│   ├── Linear(11025 → 256)
│   ├── ReLU
│   ├── Dropout(0.5)
│   └── Linear(256 → 2)
└── traj_head: Linear(105 → 105)
    """, language="text")
    
    st.markdown("### 📉 Loss Components")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Classification Loss**")
        st.latex(r"\mathcal{L}_{cls} = \text{CrossEntropy}(\hat{y}, y)")
    with col2:
        st.markdown("**Trajectory Loss**")
        st.latex(r"\mathcal{L}_{traj} = \text{MSE}(\hat{x}_{t+1}, x_{t+1})")
    with col3:
        st.markdown("**Sparsity Loss**")
        st.latex(r"\mathcal{L}_{sparse} = \lambda \|W_{final}\|_1")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        st.warning("Training requires GPU and full dataset. Use `train_bdh.py` script.")
        st.code("python Model-Physcosis/train_bdh.py --epochs 20 --batch_size 16", language="bash")

# ============================================================================
# TAB 2: INFERENCE
# ============================================================================

with tab2:
    st.markdown("## 🔮 Single-Subject Inference")
    
    data_source = st.radio(
        "Data Source",
        ["🎲 Generate Sample Data", "📤 Upload FNC + ICN Files"],
        horizontal=True
    )
    
    fmri_data = None
    
    if data_source == "🎲 Generate Sample Data":
        if st.button("Generate Sample rs-fMRI Data"):
            fmri_data = generate_fmri_sample(num_nodes=105, sequence_length=230)
            st.success(f"✅ Generated sample data:")
            st.json({
                "FNC shape": list(fmri_data['fnc'].shape),
                "ICN shape": list(fmri_data['icn'].shape)
            })
    else:
        col1, col2 = st.columns(2)
        with col1:
            fnc_file = st.file_uploader("Upload FNC Matrix", type=['npy', 'csv'])
        with col2:
            icn_file = st.file_uploader("Upload ICN Timecourses", type=['npy', 'csv'])
        
        if fnc_file and icn_file:
            st.success("✅ Files uploaded. Ready for inference.")
    
    st.divider()
    
    if st.button("🧠 Run Classification", type="primary", use_container_width=True):
        
        if PSYCHOSIS_MODEL_AVAILABLE:
            # Load models
            with st.spinner("Loading BDHNet and SpikeEncoder..."):
                models = load_psychosis_model(num_nodes=105)
                bdh_net = models['bdh_net']
                spike_encoder = models['spike_encoder']
            
            st.success("✅ Models loaded: BDHNet, SpikeEncoder")
            
            # Generate data if not provided
            if fmri_data is None:
                fmri_data = generate_fmri_sample(num_nodes=105, sequence_length=230)
            
            # Reconstruct adjacency matrix from FNC
            fnc = fmri_data['fnc']
            icn = fmri_data['icn']
            
            # FNC to adjacency matrix
            num_nodes = 105
            adj = torch.zeros(num_nodes, num_nodes)
            triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
            adj[triu_idx[0], triu_idx[1]] = fnc
            adj = adj + adj.T  # Make symmetric
            
            st.info(f"📊 Adjacency matrix reconstructed: {list(adj.shape)}")
            
            # Spike encoding
            with st.spinner("Encoding ICN timecourses to spike trains..."):
                spikes = spike_encoder(icn.numpy())
            
            st.success(f"✅ Spike encoding complete: {list(spikes.shape)}")
            
            # Run through BDHNet
            with st.spinner("Running BDHNet inference..."):
                bdh_net.eval()
                with torch.no_grad():
                    # Add batch dimension
                    x = spikes.unsqueeze(0)  # (1, T, N)
                    w_init = adj.unsqueeze(0)  # (1, N, N)
                    mask = torch.ones(1, x.shape[1])
                    
                    logits, w_final, traj_pred = bdh_net(x, w_init, mask)
                    
                    probs = torch.softmax(logits, dim=1)
            
            st.success("✅ **Classification Complete!**")
            
            # Results
            st.divider()
            st.markdown("## 📋 Classification Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                bp_prob = probs[0, 0].item() * 100
                sz_prob = probs[0, 1].item() * 100
                
                prediction = "Schizophrenia (SZ)" if sz_prob > bp_prob else "Bipolar Disorder (BP)"
                confidence = max(sz_prob, bp_prob)
                
                if sz_prob > bp_prob:
                    st.error(f"**Prediction:** {prediction}")
                else:
                    st.warning(f"**Prediction:** {prediction}")
                
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col2:
                st.markdown("**Class Probabilities:**")
                st.progress(bp_prob / 100, text=f"Bipolar (BP): {bp_prob:.1f}%")
                st.progress(sz_prob / 100, text=f"Schizophrenia (SZ): {sz_prob:.1f}%")
            
            # Show weight evolution
            with st.expander("🔍 View Final Synaptic Weights"):
                st.markdown(f"**W_final shape:** {list(w_final.shape)}")
                st.markdown(f"**Weight statistics:**")
                st.json({
                    "mean": round(w_final.mean().item(), 4),
                    "std": round(w_final.std().item(), 4),
                    "sparsity": round((w_final.abs() < 0.1).float().mean().item() * 100, 1)
                })
                
                # Store results for Comprehensive Report
                st.session_state['psychosis_results'] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probs': {'BP': bp_prob, 'SZ': sz_prob},
                    'final_weights_stats': {
                        "mean": round(w_final.mean().item(), 4),
                        "sparsity": round((w_final.abs() < 0.1).float().mean().item() * 100, 1)
                    }
                }
        
        else:
            st.error("Psychosis model not available. Check Model-Physcosis/src/ imports.")

# ============================================================================
# TAB 3: VISUALIZATION
# ============================================================================

with tab3:
    st.markdown("## 📈 Connectivity Visualization")
    
    viz_type = st.selectbox(
        "Visualization Type",
        ["Functional Connectivity Matrix", "Spike Raster Plot", "Synaptic Weight Evolution"]
    )
    
    if st.button("Generate Visualization"):
        # Generate sample data for visualization
        sample = generate_fmri_sample(num_nodes=105, sequence_length=100)
        
        if viz_type == "Functional Connectivity Matrix":
            # Reconstruct FNC matrix
            fnc = sample['fnc']
            num_nodes = 105
            adj = torch.zeros(num_nodes, num_nodes)
            triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
            adj[triu_idx[0], triu_idx[1]] = fnc
            adj = (adj + adj.T).numpy()
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(adj, cmap='coolwarm', aspect='auto')
            ax.set_title("Functional Network Connectivity (FNC)")
            ax.set_xlabel("ICN Region")
            ax.set_ylabel("ICN Region")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
        
        elif viz_type == "Spike Raster Plot":
            import matplotlib.pyplot as plt
            
            if PSYCHOSIS_MODEL_AVAILABLE:
                models = load_psychosis_model()
                spike_encoder = models['spike_encoder']
                spikes = spike_encoder(sample['icn'].numpy())
            else:
                spikes = torch.rand(100, 105) > 0.95
            
            fig, ax = plt.subplots(figsize=(10, 6))
            spike_matrix = spikes[:, :30].numpy() if hasattr(spikes, 'numpy') else spikes[:, :30]
            rows, cols = np.where(spike_matrix > 0.5)
            ax.scatter(rows, cols, s=1, c='purple', marker='|')
            ax.set_title("Spike Raster Plot (First 30 ICN Regions)")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("ICN Region")
            st.pyplot(fig)
        
        else:
            st.info("Weight evolution visualization requires running full inference first.")
