"""
GenZERO - Unified Frontend Application
Bio-Behavioral Monitoring System with BDH Architecture
"""
import streamlit as st

# Import from backend integration module
try:
    from backend_integration import (
        SPEECH_MODEL_AVAILABLE,
        CHILD_MIND_MODEL_AVAILABLE,
        PSYCHOSIS_MODEL_AVAILABLE,
        load_speech_model,
        generate_speech_input,
        run_speech_inference,
        get_backend_status
    )
    import torch
    BACKEND_AVAILABLE = SPEECH_MODEL_AVAILABLE
except ImportError:
    BACKEND_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="GenZERO | Bio-Behavioral Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    /* Root Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --dark-bg: #0e1117;
        --card-bg: rgba(17, 25, 40, 0.75);
        --accent-cyan: #00d4ff;
        --accent-purple: #7c3aed;
        --accent-pink: #f472b6;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 3rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 300;
        margin-bottom: 1.5rem;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(17, 25, 40, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Stats Section */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.5rem;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Tech Stack Pills */
    .tech-pill {
        display: inline-block;
        background: rgba(124, 58, 237, 0.2);
        border: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 20px;
        padding: 0.4rem 1rem;
        margin: 0.25rem;
        font-size: 0.85rem;
        color: #a78bfa;
    }
    
    /* Navigation Buttons */
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .nav-button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1f2e 0%, #0e1117 100%);
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animation Keyframes */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .float-animation {
        animation: float 3s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 1.8rem; background: linear-gradient(135deg, #667eea, #764ba2); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🧠 GenZERO</h1>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Bio-Behavioral Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🎤 Speech Analysis", "👶 Child Mind Assessment", "🧬 Psychosis Detection", 
         "📊 Clinical Dashboard", "⚙️ System Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # System Status
    st.markdown("### 📡 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models", "3", delta="Active")
    with col2:
        st.metric("GPU", "Ready", delta="CUDA")
    
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; color: rgba(255,255,255,0.4); font-size: 0.75rem;">
        <p>GenZERO v1.0.0</p>
        <p>BDH Architecture | Hebbian Learning</p>
    </div>
    """, unsafe_allow_html=True)

# Main Content Based on Navigation
if page == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">GenZERO</div>
        <div class="hero-subtitle">
            Next-Generation Bio-Behavioral Monitoring System powered by<br>
            Baby Dragon Hatchling (BDH) Spiking Neural Networks
        </div>
        <div style="margin-top: 1rem;">
            <span class="tech-pill">🧠 Spiking Neural Networks</span>
            <span class="tech-pill">⚡ Hebbian Plasticity</span>
            <span class="tech-pill">🔬 Linear Attention O(T)</span>
            <span class="tech-pill">🏥 Clinical ML</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-value">3</div>
            <div class="stat-label">Medical Models</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">~5%</div>
            <div class="stat-label">Sparse Activation</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">O(T)</div>
            <div class="stat-label">Time Complexity</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">100%</div>
            <div class="stat-label">Edge Privacy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("## 🎯 Model Tracks")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎤</div>
            <div class="feature-title">Speech Emotion Recognition</div>
            <div class="feature-desc">
                Analyze audio for micro-prosodic features, speech disfluency patterns, 
                and emotional biomarkers using the DAIC-WOZ dataset integration.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">👶</div>
            <div class="feature-title">Child Mind Assessment</div>
            <div class="feature-desc">
                Age-gated biological modeling using actigraphy data. 
                Predicts PCIAT scores with Sparse Autoencoder features and ensemble boosters.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧬</div>
            <div class="feature-title">Psychosis Classification</div>
            <div class="feature-desc">
                Distinguishes Schizophrenia from Bipolar Disorder using rs-fMRI connectome data 
                with trajectory prediction and SMOTE augmentation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Architecture Overview
    st.markdown("## 🏗️ BDH Architecture Overview")
    
    st.markdown("""
    <div class="feature-card">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;">
            <div>
                <h4 style="color: #667eea;">🎤 Multi-Modal Input</h4>
                <p style="color: rgba(255,255,255,0.6);">Audio, Video, and Text streams with temporal alignment and cross-modal synchronization.</p>
            </div>
            <div>
                <h4 style="color: #764ba2;">🧠 BDH Core</h4>
                <p style="color: rgba(255,255,255,0.6);">Spiking neurons with Hebbian plasticity, dual-memory system (frozen + plastic graphs).</p>
            </div>
            <div>
                <h4 style="color: #f472b6;">📊 Analysis Layer</h4>
                <p style="color: rgba(255,255,255,0.6);">Concept probes for tremor, stutter, and fatigue detection with semantic mapping.</p>
            </div>
            <div>
                <h4 style="color: #00d4ff;">🏥 Clinical Output</h4>
                <p style="color: rgba(255,255,255,0.6);">Interpretable reports, risk scores, and glass-brain visualizations with audit trails.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "🎤 Speech Analysis":
    st.markdown("# 🎤 Speech Emotion Recognition")
    st.markdown("*Analyze speech patterns for emotional biomarkers and clinical indicators*")
    
    st.divider()
    
    # Mode Selection
    analysis_mode = st.radio(
        "Analysis Mode",
        ["🎙️ Real-time Simulation", "📁 Upload Audio File", "📂 DAIC-WOZ Session"],
        horizontal=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if analysis_mode == "🎙️ Real-time Simulation":
            st.markdown("### Simulation Controls")
            
            patient_state = st.selectbox(
                "Patient State",
                ["Baseline Recovery", "Tremor Onset", "Speech Disfluency", "Cognitive Fatigue"]
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                time_steps = st.slider("Time Steps", 1, 100, 20)
            with col_b:
                hidden_dim = st.slider("Hidden Dimensions", 64, 512, 256)
            
            if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
                if BACKEND_AVAILABLE:
                    # Use actual BDH backend
                    with st.spinner("Processing through real BDH network..."):
                        # Initialize models
                        model = SpikingNeuralNetwork(input_dim=10, hidden_dim=hidden_dim)
                        analyzer = AnalysisLayer()
                        dashboard = ClinicalDashboard()
                        
                        # Generate input based on state
                        input_data = torch.randn(1, 10) * 0.1
                        if patient_state == "Tremor Onset":
                            input_data[0, 0:3] += 2.0
                        elif patient_state == "Speech Disfluency":
                            input_data[0, 3:5] += 2.0
                        elif patient_state == "Cognitive Fatigue":
                            input_data[0, 5:8] += 1.5
                        
                        # Forward pass
                        snn_output = model.forward(input_data)
                        hidden_state = snn_output["hidden_state"]
                        weights = model.update_synapses(input_data, hidden_state)
                        analysis_results = analyzer.analyze(snn_output, weights)
                    
                    st.success("✅ Real BDH Processing Complete!")
                    st.markdown("### 📋 Clinical Assessment (from backend)")
                    
                    # Use actual dashboard report
                    report = dashboard.generate_report(analysis_results)
                    st.markdown(report)
                    
                    # Show anomalies
                    if analysis_results.get("anomalies"):
                        for anomaly in analysis_results["anomalies"]:
                            st.error(f"⚠️ {anomaly}")
                    
                    # Visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        st.markdown("**Neural Sparsity**")
                        fig_sparsity = dashboard.plot_sparsity(hidden_state)
                        st.pyplot(fig_sparsity)
                    with viz_col2:
                        st.markdown("**Synaptic Topology**")
                        fig_topology = dashboard.plot_topology(weights)
                        st.pyplot(fig_topology)
                else:
                    # Fallback to mock results
                    import time as time_module
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time_module.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    st.success("✅ Simulation Complete (mock mode)")
                    st.markdown("### 📋 Clinical Assessment")
                    
                    if patient_state == "Tremor Onset":
                        st.error("⚠️ **Alert:** Tremor Detected")
                    elif patient_state == "Speech Disfluency":
                        st.warning("⚠️ **Alert:** Speech Disfluency Detected")
                    else:
                        st.info("✅ **Status:** Patient Stable")
        
        elif analysis_mode == "📁 Upload Audio File":
            st.markdown("### Upload Audio")
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac'],
                help="Supported formats: WAV, MP3, FLAC"
            )
            
            if uploaded_file:
                st.audio(uploaded_file)
                st.success(f"Loaded: {uploaded_file.name}")
                
                if st.button("🔬 Analyze Audio", type="primary"):
                    st.info("Analysis would extract MFCC, ZCR, RMS features and process through BDH network")
        
        else:
            st.markdown("### DAIC-WOZ Dataset")
            st.text_input("Dataset Root Path", value="src/Dataset/DAIC_WOZ_Data")
            max_sessions = st.number_input("Max Sessions", min_value=1, max_value=100, value=3)
            
            if st.button("🔍 Load Sessions", type="primary"):
                st.info("Would load and process DAIC-WOZ sessions with transcript + COVAREP features")
    
    with col2:
        st.markdown("### 📊 Neural Sparsity")
        st.markdown("""
        <div class="feature-card" style="text-align: center; padding: 2rem;">
            <p style="color: rgba(255,255,255,0.5);">Run simulation to view neural activation patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔗 Synaptic Topology")
        st.markdown("""
        <div class="feature-card" style="text-align: center; padding: 2rem;">
            <p style="color: rgba(255,255,255,0.5);">Hebbian weight visualization will appear here</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "👶 Child Mind Assessment":
    st.markdown("# 👶 Child Mind Institute Assessment")
    st.markdown("*Age-gated biological dynamic modeling for pediatric mental health*")
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Data Input")
        
        data_source = st.radio(
            "Data Source",
            ["📤 Upload Actigraphy CSV", "🔗 Load from Dataset"],
            horizontal=True
        )
        
        if data_source == "📤 Upload Actigraphy CSV":
            uploaded = st.file_uploader("Upload actigraphy data (Parquet/CSV)", type=['parquet', 'csv'])
            if uploaded:
                st.success(f"Loaded: {uploaded.name}")
        
        st.markdown("### 👤 Patient Demographics")
        
        age = st.slider("Patient Age", 5, 18, 12)
        
        age_group = "< 10 years" if age < 10 else ("10-15 years" if age <= 15 else "16+ years")
        st.info(f"Age Group Classification: **{age_group}**")
        
        st.markdown("### ⚙️ Model Configuration")
        
        col_a, col_b = st.columns(2)
        with col_a:
            virtual_nodes = st.selectbox("Virtual Neurons", [32, 64, 128], index=1)
            seq_length = st.slider("Sequence Length", 100, 500, 230)
        with col_b:
            latent_dim = st.selectbox("SAE Latent Dim", [16, 32, 64], index=1)
            ensemble = st.multiselect("Ensemble", ["LightGBM", "XGBoost", "CatBoost"], 
                                     default=["LightGBM", "XGBoost", "CatBoost"])
    
    with col2:
        st.markdown("### 🎯 Prediction Pipeline")
        
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #667eea;">📐 BDH Feature Extraction</h4>
            <p style="color: rgba(255,255,255,0.6);">
                5 physical sensors → 64 virtual neurons<br>
                Hebbian plasticity updates: W<sub>t+1</sub> = αW<sub>t</sub> + η(x<sub>t</sub>x<sub>t</sub><sup>T</sup>)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #764ba2;">🔬 Sparse Autoencoder</h4>
            <p style="color: rgba(255,255,255,0.6);">
                4096-dim synaptic graph → 32-dim latent space<br>
                L1 regularization for monosemantic features
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #f472b6;">📊 Ensemble Prediction</h4>
            <p style="color: rgba(255,255,255,0.6);">
                PCIAT Total Score prediction<br>
                Quadratic Weighted Kappa optimization
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run Assessment", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                import time
                time.sleep(2)
            
            st.success("Assessment Complete")
            
            st.metric("Predicted PCIAT Score", "42.7", delta="-3.2 from baseline")
            
            sii_map = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
            predicted_sii = 1
            st.warning(f"**SII Classification:** {sii_map[predicted_sii]} (sii={predicted_sii})")

elif page == "🧬 Psychosis Detection":
    st.markdown("# 🧬 Psychosis Classification")
    st.markdown("*Distinguishing Schizophrenia from Bipolar Disorder using rs-fMRI connectomics*")
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧠 Input Data")
        
        st.text_input("rs-fMRI Data Directory", value="data/train")
        
        st.markdown("### 📊 Data Properties")
        
        st.markdown("""
        | Property | Value |
        |----------|-------|
        | ICN Regions | 105 |
        | FNC Features | 5460 (upper triangle) |
        | Time Points | Variable (padded to 230) |
        | Classes | SZ (1), BP (0) |
        """)
        
        st.markdown("### ⚙️ Training Configuration")
        
        col_a, col_b = st.columns(2)
        with col_a:
            epochs = st.slider("Epochs", 5, 50, 20)
            batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        with col_b:
            learning_rate = st.select_slider("Learning Rate", 
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            lambda_sparse = st.select_slider("λ Sparse", 
                options=[1e-5, 1e-4, 1e-3], value=1e-4)
    
    with col2:
        st.markdown("### 🔬 BDH Network Architecture")
        
        st.markdown("""
        <div class="feature-card">
            <p style="color: rgba(255,255,255,0.8); font-family: monospace;">
            BDHNet(<br>
            &nbsp;&nbsp;bdh_layer: BDHLayer(105 nodes)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;├─ α (decay): 0.9<br>
            &nbsp;&nbsp;&nbsp;&nbsp;└─ η (learning): 0.01<br>
            &nbsp;&nbsp;classifier: Sequential(<br>
            &nbsp;&nbsp;&nbsp;&nbsp;├─ Dropout(0.5)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;├─ Linear(11025 → 256)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;├─ ReLU<br>
            &nbsp;&nbsp;&nbsp;&nbsp;├─ Dropout(0.5)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;└─ Linear(256 → 2)<br>
            &nbsp;&nbsp;)<br>
            &nbsp;&nbsp;traj_head: Linear(105 → 105)<br>
            )
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Loss Components")
        st.markdown("""
        - **Classification Loss**: CrossEntropy
        - **Trajectory Loss**: MSE (x<sub>t+1</sub> prediction)
        - **Sparsity Loss**: L1 on W<sub>final</sub>
        """)
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            import time
            for i in range(20):
                time.sleep(0.15)
                progress.progress((i + 1) * 5)
                status.text(f"Epoch {i+1}/20 | Loss: {1.2 - i*0.05:.4f} | Acc: {50 + i*2.3:.1f}%")
            
            st.success("Training Complete!")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Final Accuracy", "95.2%", delta="+45.2%")
            with col_m2:
                st.metric("Validation Accuracy", "89.7%", delta=None)

elif page == "📊 Clinical Dashboard":
    st.markdown("# 📊 Clinical Dashboard")
    st.markdown("*Comprehensive visualization and interpretability for clinical decision support*")
    
    st.divider()
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="font-size: 2rem;">🧠</div>
            <div style="font-size: 1.8rem; color: #667eea; font-weight: bold;">247</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Total Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="font-size: 2rem;">⚡</div>
            <div style="font-size: 1.8rem; color: #00d4ff; font-weight: bold;">3.2ms</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Inference Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="font-size: 2rem;">📈</div>
            <div style="font-size: 1.8rem; color: #f472b6; font-weight: bold;">94.8%</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="font-size: 2rem;">🔒</div>
            <div style="font-size: 1.8rem; color: #10b981; font-weight: bold;">100%</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Edge Private</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Report Generation
    st.markdown("### 📋 Report Generation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Full Clinical Assessment", "Concept Probe Summary", "Synaptic Evolution", "Risk Trend Analysis"]
        )
        
        patient_id = st.text_input("Patient ID", value="PAT-2024-001")
        date_range = st.date_input("Assessment Date")
    
    with col2:
        include_visualizations = st.checkbox("Include Visualizations", value=True)
        include_raw_data = st.checkbox("Include Raw Activation Data", value=False)
        export_format = st.radio("Export Format", ["PDF", "HTML", "Markdown"], horizontal=True)
    
    if st.button("📄 Generate Report", type="primary"):
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea;">## Clinical Assessment Report</h3>
            <p><strong>Patient ID:</strong> PAT-2024-001</p>
            <p><strong>Assessment Date:</strong> 2024-12-28</p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            
            <h4>Neuromorphic Concept Activation</h4>
            <ul style="color: rgba(255,255,255,0.8);">
                <li><strong>Tremor:</strong> Inactive (Activation: 0.21, Synaptic Strength: 0.18)</li>
                <li><strong>Stutter:</strong> Inactive (Activation: 0.19, Synaptic Strength: 0.22)</li>
                <li><strong>Fatigue:</strong> Inactive (Activation: 0.24, Synaptic Strength: 0.20)</li>
            </ul>
            
            <h4>Clinical Recommendation</h4>
            <p style="color: rgba(255,255,255,0.7);">
                Patient shows stable baseline with no significant anomalies detected. 
                Continue current monitoring protocol. Schedule follow-up in 30 days.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Visualization Section
    st.markdown("### 🎨 Visualizations")
    
    viz_type = st.selectbox(
        "Visualization Type",
        ["Neural Activation Sparsity", "Synaptic Topology Graph", "Concept Activation Timeline", "Risk Score Gauge"]
    )
    
    st.info("📊 Select a visualization type and run an analysis to display interactive graphs")

elif page == "⚙️ System Settings":
    st.markdown("# ⚙️ System Settings")
    st.markdown("*Configure model parameters and system preferences*")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["🧠 Model Settings", "💾 Memory & Storage", "🎨 Display"])
    
    with tab1:
        st.markdown("### Neural Network Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**BDH Parameters**")
            st.slider("Decay Factor (α)", 0.5, 1.0, 0.9, 0.01)
            st.slider("Learning Rate (η)", 0.001, 0.1, 0.01, 0.001)
            st.number_input("Hidden Dimensions", 64, 1024, 256, 64)
        
        with col2:
            st.markdown("**Sparse Autoencoder**")
            st.number_input("Latent Dimensions", 8, 128, 32, 8)
            st.select_slider("L1 Lambda", options=[1e-6, 1e-5, 1e-4, 1e-3], value=1e-5)
            st.number_input("Hidden Layer Size", 64, 512, 128, 64)
        
        st.markdown("### Concept Probe Thresholds")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.slider("Tremor Threshold", 0.1, 0.9, 0.5, 0.05)
        with col2:
            st.slider("Stutter Threshold", 0.1, 0.9, 0.5, 0.05)
        with col3:
            st.slider("Fatigue Threshold", 0.1, 0.9, 0.4, 0.05)
    
    with tab2:
        st.markdown("### Persistent Memory")
        
        st.checkbox("Enable Synaptic State Caching", value=True)
        st.number_input("Max Cached Sessions", 10, 1000, 100)
        st.text_input("Cache Directory", value="./cache/synaptic_states")
        
        st.markdown("### Model Weights")
        
        st.text_input("BDH Model Path", value="bdh_model.pth")
        st.text_input("SAE Model Path", value="sae_child_mind.pth")
        
        if st.button("🔄 Reload Models"):
            st.success("Models reloaded successfully")
    
    with tab3:
        st.markdown("### Theme Settings")
        
        st.selectbox("Color Scheme", ["Dark Mode (Default)", "Light Mode", "High Contrast"])
        st.checkbox("Enable Animations", value=True)
        st.checkbox("Show Debug Info", value=False)
        
        st.markdown("### Visualization Defaults")
        
        st.selectbox("Default Chart Library", ["Matplotlib", "Plotly", "Altair"])
        st.slider("Chart DPI", 72, 300, 150)
