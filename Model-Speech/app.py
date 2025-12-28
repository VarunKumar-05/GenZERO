import streamlit as st
import torch
import numpy as np
import time
from src.bdh_snn.network import SpikingNeuralNetwork
from src.analysis.detector import AnalysisLayer
from src.clinical.dashboard import ClinicalDashboard

# Page Config
st.set_page_config(page_title="SynaptoRehab: BDH Monitor", layout="wide")

st.title("SynaptoRehab: Bio-Behavioral Monitoring System")
st.markdown("""
**Frontier Architecture:** Baby Dragon Hatchling (BDH) | **Mechanism:** Hebbian Plasticity & Linear Attention
""")

# Sidebar for Controls
st.sidebar.header("Simulation Controls")
simulation_mode = st.sidebar.selectbox("Patient State", ["Baseline Recovery", "Tremor Onset", "Speech Disfluency", "Cognitive Fatigue"])
steps = st.sidebar.slider("Time Steps", 1, 50, 10)

# Initialize System Components (Cached)
@st.cache_resource
def load_system():
    model = SpikingNeuralNetwork(input_dim=10, hidden_dim=20)
    analyzer = AnalysisLayer()
    dashboard = ClinicalDashboard()
    return model, analyzer, dashboard

model, analyzer, dashboard = load_system()

# Simulation Logic
if st.button("Run Simulation"):
    st.write(f"Running simulation for state: **{simulation_mode}**")
    
    # Generate synthetic input based on mode
    input_dim = 10
    if simulation_mode == "Baseline Recovery":
        # Random noise, low amplitude
        input_data = torch.randn(1, input_dim) * 0.1
    elif simulation_mode == "Tremor Onset":
        # High frequency oscillation pattern in first few dimensions
        input_data = torch.randn(1, input_dim) * 0.1
        input_data[0, 0:3] += 2.0 # Strong signal in tremor channels
    elif simulation_mode == "Speech Disfluency":
        # Irregular spikes in middle dimensions
        input_data = torch.randn(1, input_dim) * 0.1
        input_data[0, 3:5] += 2.0
    elif simulation_mode == "Cognitive Fatigue":
        # Low overall activity, drift
        input_data = torch.randn(1, input_dim) * 0.05
    
    # Run Model
    with st.spinner("Processing Neural Dynamics..."):
        # Forward pass
        snn_output = model.forward(input_data)
        hidden_state = snn_output["hidden_state"]
        
        # Update Synapses (Hebbian Learning)
        weights = model.update_synapses(input_data, hidden_state)
        
        # Analyze
        analysis_results = analyzer.analyze(snn_output, weights)
        
        time.sleep(1) # UX pause

    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Clinical Report")
        report = dashboard.generate_report(analysis_results)
        st.markdown(report)
        
        st.subheader("Neural Sparsity")
        fig_sparsity = dashboard.plot_sparsity(hidden_state)
        st.pyplot(fig_sparsity)

    with col2:
        st.subheader("Synaptic Topology (Hebbian Growth)")
        fig_topology = dashboard.plot_topology(weights)
        st.pyplot(fig_topology)

    # Raw Data Expander
    with st.expander("View Raw System State"):
        st.json({
            "input_vector": input_data.tolist(),
            "output_vector": snn_output["output"].tolist(),
            "analysis_debug": analysis_results
        })

else:
    st.info("Click 'Run Simulation' to start the bio-behavioral monitoring loop.")
