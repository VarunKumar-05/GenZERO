"""
Page 1: Speech Emotion Recognition
Fully integrated with backend BDH components with trained weights
Supports audio file upload (WAV, MP3) with automatic feature extraction
"""
import streamlit as st
import sys
import os
import numpy as np
import tempfile

# Page Configuration
st.set_page_config(page_title="Speech Analysis | GenZERO", page_icon="🎤", layout="wide")

# Add parent directory to path and import backend
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import backend integration
from backend_integration import (
    SPEECH_MODEL_AVAILABLE,
    load_speech_model,
    generate_speech_input,
    run_speech_inference,
    analyze_words_during_states,
    generate_daic_style_report
)

import torch

# Check if audio processing is available
try:
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("🎤 Speech Emotion Recognition")
st.markdown("*BDH Spiking Neural Network with Hebbian Plasticity & Concept Probes*")

# Show backend status
if SPEECH_MODEL_AVAILABLE:
    audio_status = "✅ Audio processing available" if AUDIO_AVAILABLE else "⚠️ Audio processing unavailable (librosa not installed)"
    st.success(f"✅ Backend loaded: SpikingNeuralNetwork, AnalysisLayer, ClinicalDashboard | {audio_status}")
else:
    st.error("❌ Backend not available. Check that src/ folder is accessible.")
    st.stop()

st.divider()

# ============================================================================
# DATA INPUT
# ============================================================================

st.markdown("## 📊 Input Data")

# Trained model uses input_dim=40, hidden_dim=256
TRAINED_INPUT_DIM = 40
TRAINED_HIDDEN_DIM = 256

if AUDIO_AVAILABLE:
    data_source = st.radio(
        "Data Source",
        ["🎤 Upload Audio File (WAV/MP3)", "🎲 Simulate Patient State"],
        horizontal=True
    )
else:
    data_source = st.radio(
        "Data Source",
        ["🎲 Simulate Patient State"],
        horizontal=True
    )
    st.warning("⚠️ Audio upload requires librosa. Install with: `pip install librosa`")

input_data = None
audio_features = None

if "Upload Audio" in data_source:
    st.markdown("### 🎤 Upload Audio File")
    
    uploaded_audio = st.file_uploader(
        "Upload audio file for analysis",
        type=['wav', 'mp3', 'ogg', 'flac'],
        help="Supported formats: WAV, MP3, OGG, FLAC"
    )
    
    if uploaded_audio:
        st.audio(uploaded_audio, format='audio/wav')
        
        # Save to temp file and extract features
        with st.spinner("Extracting audio features (MFCC, ZCR, RMS)..."):
            try:
                # Save uploaded file to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(uploaded_audio.read())
                    tmp_path = tmp.name
                
                # Load and extract features using librosa
                y, sr = librosa.load(tmp_path, sr=22050)
                
                # Extract MFCC (13 coefficients)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = mfccs.mean(axis=1)  # Average over time
                mfcc_std = mfccs.std(axis=1)
                
                # Extract spectral features
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
                
                # Extract ZCR and RMS
                zcr = librosa.feature.zero_crossing_rate(y).mean()
                rms = librosa.feature.rms(y=y).mean()
                
                # Combine into 40-dim feature vector
                # 13 MFCC mean + 13 MFCC std + 3 spectral + ZCR + RMS + padding = 40
                features = np.concatenate([
                    mfcc_mean,           # 13
                    mfcc_std,            # 13
                    [spectral_centroid / 10000],  # 1 (normalized)
                    [spectral_bandwidth / 10000], # 1 (normalized)
                    [spectral_rolloff / 10000],   # 1 (normalized)
                    [zcr * 10],          # 1 (scaled)
                    [rms * 10],          # 1 (scaled)
                    np.zeros(9)          # 9 (padding to reach 40)
                ])
                
                audio_features = features[:TRAINED_INPUT_DIM]  # Ensure exactly 40
                
                # Store temp path for transcription later
                st.session_state['audio_temp_path'] = tmp_path
                
                st.success(f"✅ Extracted {TRAINED_INPUT_DIM} features from audio")
                
                # Show feature preview
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**MFCC Features:**")
                    st.bar_chart(mfcc_mean)
                with col2:
                    st.markdown("**Spectral Features:**")
                    st.json({
                        "Spectral Centroid": round(spectral_centroid, 2),
                        "Spectral Bandwidth": round(spectral_bandwidth, 2),
                        "ZCR": round(zcr, 4),
                        "RMS Energy": round(rms, 4)
                    })
                # Text/Transcript upload for word-level analysis
                st.markdown("---")
                st.markdown("### 📝 Transcript Input (Optional)")
                st.markdown("*Upload transcript for word-level analysis like daic_analysis_report.txt*")
                
                transcript_file = st.file_uploader(
                    "Upload transcript (CSV or TXT)",
                    type=['csv', 'txt'],
                    help="CSV should have text in first column, or TXT with raw text"
                )
                
                if transcript_file:
                    try:
                        if transcript_file.name.endswith('.csv'):
                            import pandas as pd
                            df = pd.read_csv(transcript_file)
                            # Get text from first column or 'text'/'value' column
                            if 'text' in df.columns:
                                transcript = ' '.join(df['text'].astype(str).tolist())
                            elif 'value' in df.columns:
                                transcript = ' '.join(df['value'].astype(str).tolist())
                            else:
                                transcript = ' '.join(df.iloc[:, 0].astype(str).tolist())
                        else:
                            transcript = transcript_file.read().decode('utf-8')
                        
                        # Tokenize
                        words = transcript.lower().split()
                        clean_words = [w.strip('.,?!;:') for w in words if w.strip('.,?!;:')]
                        
                        st.success(f"✅ Loaded transcript with {len(clean_words)} words")
                        st.session_state['transcript'] = transcript
                        st.session_state['words'] = clean_words
                        
                        with st.expander("📝 View Transcript"):
                            st.text(transcript[:500] + "..." if len(transcript) > 500 else transcript)
                            
                    except Exception as e:
                        st.error(f"Error loading transcript: {e}")
                        st.session_state['transcript'] = None
                        st.session_state['words'] = None
                else:
                    st.session_state['transcript'] = None
                    st.session_state['words'] = None
                    
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                audio_features = None

else:
    st.markdown("### Patient State Simulation")
    
    patient_state = st.selectbox(
        "Select Patient State",
        ["Baseline Recovery", "Tremor Onset", "Speech Disfluency", "Cognitive Fatigue"]
    )
    
    st.warning(f"⚠️ Will generate **simulated** {TRAINED_INPUT_DIM}-dim feature vector")

st.divider()

# ============================================================================
# INFERENCE
# ============================================================================

st.markdown("## 🧠 BDH Neural Inference")

if st.button("🚀 Run BDH Analysis", type="primary", use_container_width=True):
    
    # Load model components with trained weights
    with st.spinner("Loading BDH model with trained weights..."):
        models = load_speech_model(
            input_dim=TRAINED_INPUT_DIM, 
            hidden_dim=TRAINED_HIDDEN_DIM,
            load_weights=True
        )
        snn = models['snn']
        analyzer = models['analyzer']
        dashboard = models['dashboard']
        weights_loaded = models.get('weights_loaded', False)
    
    # Show weight status
    if weights_loaded:
        st.success("✅ Models loaded with **TRAINED WEIGHTS** from ser_bdh_model.pth")
    else:
        st.warning("⚠️ Using random weights (trained weights not found)")
    
    # Prepare input data
    if "Upload Audio" in data_source:
        if audio_features is not None:
            input_data = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0)
            st.success("✅ Using **REAL audio features** extracted from uploaded file")
        else:
            st.error("Please upload an audio file first")
            st.stop()
    else:
        with st.spinner("Generating simulated input..."):
            input_data = generate_speech_input(patient_state, input_dim=TRAINED_INPUT_DIM)
        st.info(f"⚠️ Using **simulated** input based on '{patient_state}'")
    
    st.info(f"📊 Input tensor shape: {list(input_data.shape)}")
    
    # Run inference
    with st.spinner("Processing through BDH Spiking Neural Network..."):
        results = run_speech_inference(snn, analyzer, dashboard, input_data)
    
    st.success("✅ **BDH Processing Complete!**")
    
    # ========================================================================
    # CLINICAL ASSESSMENT REPORT
    # ========================================================================
    
    st.divider()
    st.markdown("## 📋 Clinical Assessment Report")
    
    # Get analysis data
    anomalies = results['analysis'].get('anomalies', [])
    concept_probes = results['analysis'].get('concept_probes', {})
    
    # Status Banner
    if anomalies:
        st.error("### ⚠️ Anomalies Detected")
        for anomaly in anomalies:
            st.markdown(f"- {anomaly}")
    else:
        st.success("### ✅ Patient Status: Stable")
        st.markdown("No clinical anomalies detected in speech patterns.")
    
    st.markdown("---")
    
    # Concept Probe Results in Columns
    st.markdown("### 🧠 Neuromorphic Concept Probes")
    
    cols = st.columns(3)
    
    probe_config = {
        'tremor': {'icon': '🫨', 'color': '#ff6b6b', 'description': 'Voice tremor/shakiness'},
        'stutter': {'icon': '🗣️', 'color': '#feca57', 'description': 'Speech disfluency'},
        'fatigue': {'icon': '😴', 'color': '#48dbfb', 'description': 'Cognitive fatigue'}
    }
    
    for idx, (concept, data) in enumerate(concept_probes.items()):
        config = probe_config.get(concept, {'icon': '🔬', 'color': '#667', 'description': ''})
        is_active = data.get('active', False)
        activation = data.get('activation_level', 0)
        synaptic = data.get('synaptic_strength', 0)
        
        with cols[idx % 3]:
            # Status indicator
            status_emoji = "🔴" if is_active else "🟢"
            status_text = "ACTIVE" if is_active else "Normal"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {'#ff6b6b33' if is_active else '#2ed57322'}, transparent);
                border-radius: 12px;
                padding: 16px;
                border-left: 4px solid {config['color']};
                margin-bottom: 12px;
            ">
                <h4 style="margin: 0;">{config['icon']} {concept.capitalize()}</h4>
                <p style="color: #888; font-size: 12px; margin: 4px 0;">{config['description']}</p>
                <p style="font-size: 24px; margin: 8px 0;">{status_emoji} {status_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.metric("Activation", f"{activation:.3f}", delta=None)
            st.metric("Synaptic Strength", f"{synaptic:.3f}", delta=None)
    
    # Full Report (Collapsible)
    with st.expander("📄 View Full Clinical Report"):
        st.markdown(results['report'])
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    st.divider()
    st.markdown("## 🎨 Neural Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Neural Activation Sparsity")
        fig_sparsity = dashboard.plot_sparsity(results['hidden_state'])
        st.pyplot(fig_sparsity)
    
    with col2:
        st.markdown("### Synaptic Topology")
        fig_topology = dashboard.plot_topology(results['weights'])
        st.pyplot(fig_topology)
    
    # ========================================================================
    # DOWNLOADABLE CLINICAL REPORT
    # ========================================================================
    
    st.divider()
    st.markdown("## 📄 Detailed Analysis Report")
    
    # Generate comprehensive report similar to daic_analysis_report.txt
    from datetime import datetime
    
    hidden_state = results['hidden_state']
    weights = results['weights']
    
    # Calculate statistics
    hidden_norm = torch.norm(hidden_state).item()
    hidden_mean = hidden_state.mean().item()
    hidden_std = hidden_state.std().item()
    sparsity = (hidden_state.abs() < 0.1).float().mean().item() * 100
    
    # Detect active states
    detected_states = []
    for concept, data in concept_probes.items():
        if data.get('active', False):
            detected_states.append(concept.capitalize())
    
    # Get transcript and words from session state (if available)
    transcript = st.session_state.get('transcript', None)
    words = st.session_state.get('words', None)
    
    # Perform word analysis if words are available
    word_analysis = None
    if words:
        word_analysis = analyze_words_during_states(words, detected_states, concept_probes)
        
        # Display word analysis summary
        st.markdown("### 📊 Word Analysis Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Words", word_analysis.get('total_words', 0))
        with col2:
            st.metric("Unique Words", word_analysis.get('unique_words', 0))
        with col3:
            st.metric("Valid (Non-Stop)", word_analysis.get('valid_words', 0))
        
        # Top determining words
        top_words = word_analysis.get('top_determining_words', [])
        if top_words:
            st.markdown("**Top Determining Words:**")
            word_str = ', '.join([f"`{w}`({c})" for w, c in top_words[:10]])
            st.markdown(word_str)
    
    # Generate DAIC-style report
    input_source = "Uploaded Audio" if "Upload Audio" in data_source else "Simulated Patient State"
    report_text = generate_daic_style_report(
        input_source=input_source,
        detected_states=detected_states,
        concept_probes=concept_probes,
        hidden_norm=hidden_norm,
        hidden_mean=hidden_mean,
        sparsity=sparsity,
        transcript=transcript,
        word_analysis=word_analysis
    )
    
    # Store results in session state for Comprehensive Report
    st.session_state['speech_results'] = {
        'detected_states': detected_states,
        'concept_probes': {k: {'active': v.get('active', False), 'activation': v.get('activation_level', 0)} for k, v in concept_probes.items()},
        'neural_stats': {
            'hidden_norm': hidden_norm,
            'hidden_mean': hidden_mean,
            'sparsity': sparsity
        },
        'word_analysis': word_analysis,
        'report_text': report_text
    }
    
    # Display report preview
    with st.expander("📋 Preview Full Report"):
        st.code(report_text, language="text")
    
    # Download button
    st.download_button(
        label="📥 Download Full Report (.txt)",
        data=report_text,
        file_name=f"speech_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # ========================================================================
    # RAW DATA (Hidden)
    # ========================================================================
    
    with st.expander("🔍 View Raw System State"):
        st.markdown("**Tensor Shapes:**")
        st.json({
            "input_shape": list(input_data.shape),
            "hidden_state_shape": list(results['hidden_state'].shape),
            "weights_shape": list(results['weights'].shape)
        })

# ============================================================================
# ARCHITECTURE INFO
# ============================================================================

st.divider()
st.markdown("## 🏗️ Architecture Reference")

with st.expander("View BDH-SNN Architecture"):
    st.markdown(f"""
    **Trained Model:** input_dim={TRAINED_INPUT_DIM}, hidden_dim={TRAINED_HIDDEN_DIM}
    
    **Audio Feature Extraction:**
    - 13 MFCC coefficients (mean + std) = 26 features
    - Spectral: Centroid, Bandwidth, Rolloff = 3 features
    - Temporal: ZCR, RMS Energy = 2 features
    - Padding = 9 features
    - **Total: 40 features**
    
    ```
    SpikingNeuralNetwork(
        input_layer: Linear(40 → 256)
        synapses: HebbianSynapse(256)
        attention: LinearAttention(256)
        output_layer: Linear(256 → 10)
    )
    ```
    """)
