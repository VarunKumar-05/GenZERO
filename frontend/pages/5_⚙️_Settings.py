"""
Page: System Settings
Configure model parameters and system preferences
"""
import streamlit as st

st.set_page_config(page_title="Settings | GenZERO", page_icon="⚙️", layout="wide")

st.title("⚙️ System Settings")
st.markdown("*Configure model parameters and system preferences*")

st.divider()

tab1, tab2, tab3 = st.tabs(["🧠 Model Settings", "💾 Storage", "🎨 Display"])

with tab1:
    st.markdown("### Neural Network Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**BDH Parameters**")
        decay = st.slider("Decay Factor (α)", 0.5, 1.0, 0.9, 0.01)
        learning_rate = st.slider("Learning Rate (η)", 0.001, 0.1, 0.01, 0.001)
        hidden_dim = st.number_input("Hidden Dimensions", 64, 1024, 256, 64)
    
    with col2:
        st.markdown("**Sparse Autoencoder**")
        latent_dim = st.number_input("Latent Dimensions", 8, 128, 32, 8)
        l1_lambda = st.select_slider("L1 Lambda", options=[1e-6, 1e-5, 1e-4, 1e-3], value=1e-5)
    
    st.markdown("### Concept Probe Thresholds")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.slider("Tremor Threshold", 0.1, 0.9, 0.5, 0.05)
    with col2:
        st.slider("Stutter Threshold", 0.1, 0.9, 0.5, 0.05)
    with col3:
        st.slider("Fatigue Threshold", 0.1, 0.9, 0.4, 0.05)

with tab2:
    st.markdown("### Model Weights")
    st.text_input("BDH Model Path", value="bdh_model.pth")
    st.text_input("SAE Model Path", value="sae_child_mind.pth")
    
    st.markdown("### Cache Settings")
    st.checkbox("Enable Synaptic State Caching", value=True)
    st.number_input("Max Cached Sessions", 10, 1000, 100)
    
    if st.button("🔄 Reload Models"):
        st.success("Models reloaded successfully")

with tab3:
    st.markdown("### Theme Settings")
    st.selectbox("Color Scheme", ["Dark Mode (Default)", "Light Mode", "High Contrast"])
    st.checkbox("Enable Animations", value=True)
    st.checkbox("Show Debug Info", value=False)
    
    st.markdown("### Visualization")
    st.selectbox("Chart Library", ["Matplotlib", "Plotly", "Altair"])
    st.slider("Chart DPI", 72, 300, 150)
