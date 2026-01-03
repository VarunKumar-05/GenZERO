"""
GenZERO Backend Integration Module
Provides unified access to all model components from the three tracks.
"""
import sys
import os

# ============================================================================
# PATH SETUP - Add all model directories to Python path
# ============================================================================

# Get the project root (parent of frontend folder)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file in project root
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


# Add paths for each model directory
PATHS_TO_ADD = [
    PROJECT_ROOT,                                      # For src/ imports
    os.path.join(PROJECT_ROOT, 'Model-Child-Mind'),    # For Child-Mind model
    os.path.join(PROJECT_ROOT, 'Model-Physcosis'),     # For Psychosis model  
    os.path.join(PROJECT_ROOT, 'Model-Speech'),        # For Speech model
]

for path in PATHS_TO_ADD:
    if path not in sys.path:
        sys.path.insert(0, path)

# ============================================================================
# IMPORTS - Core Components
# ============================================================================

import torch
import numpy as np

# ============================================================================
# TRACK 1: SPEECH MODEL (src/ folder)
# ============================================================================

try:
    from src.bdh_snn.network import SpikingNeuralNetwork, HebbianSynapse, LinearAttention
    from src.analysis.detector import AnalysisLayer, ConceptProbes
    from src.clinical.dashboard import ClinicalDashboard
    from src.memory.storage import PersistentMemory
    SPEECH_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Speech model import failed: {e}")
    SPEECH_MODEL_AVAILABLE = False

# Audio processing is optional (requires librosa)
try:
    from src.input.audio import AudioStream
    from src.input.text import TextInput
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

# ============================================================================
# TRACK 2: CHILD-MIND MODEL
# ============================================================================

try:
    from src.model.bdh_features import BDHFeatureExtractor
    from src.model.sae import SparseAutoencoder, train_sae
    from src.model.bdh import BDHLayer as ChildMindBDHLayer
    from src.model.imputation import SynapticKNNImputer
    CHILD_MIND_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Child-Mind model import failed: {e}")
    CHILD_MIND_MODEL_AVAILABLE = False

# ============================================================================
# TRACK 3: PSYCHOSIS MODEL
# ============================================================================

try:
    from src.model.bdh import BDHLayer as PsychosisBDHLayer, BDHNet
    from src.data_loader.spike_encoder import SpikeEncoder
    PSYCHOSIS_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Psychosis model import failed: {e}")
    PSYCHOSIS_MODEL_AVAILABLE = False

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_speech_model(input_dim=40, hidden_dim=256, load_weights=True):
    """Load and return Speech model components with trained weights."""
    if not SPEECH_MODEL_AVAILABLE:
        raise RuntimeError("Speech model components not available")
    
    snn = SpikingNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
    analyzer = AnalysisLayer()
    dashboard = ClinicalDashboard()
    
    # Load trained weights if available
    weights_loaded = False
    
    if load_weights:
        weight_paths = [
            os.path.join(PROJECT_ROOT, 'Model-Speech', 'ser_bdh_model.pth'),
            os.path.join(PROJECT_ROOT, 'ser_bdh_model.pth')
        ]
        
        for weight_path in weight_paths:
            if os.path.exists(weight_path):
                try:
                    snn.load_state_dict(torch.load(weight_path, map_location='cpu'))
                    weights_loaded = True
                    print(f"[INFO] Loaded Speech SNN weights from {weight_path}")
                    break
                except Exception as e:
                    print(f"[WARNING] Could not load weights from {weight_path}: {e}")
    
    return {
        'snn': snn,
        'analyzer': analyzer,
        'dashboard': dashboard,
        'weights_loaded': weights_loaded
    }

def load_child_mind_model(input_channels=5, virtual_nodes=64, num_age_groups=3, latent_dim=32, load_weights=True):
    """Load and return Child-Mind model components with trained weights."""
    if not CHILD_MIND_MODEL_AVAILABLE:
        raise RuntimeError("Child-Mind model components not available")
    
    bdh_extractor = BDHFeatureExtractor(
        input_channels=input_channels,
        virtual_nodes=virtual_nodes,
        num_age_groups=num_age_groups
    )
    
    # BDH output: virtual_nodes^2 + virtual_nodes (weights + activity)
    sae_input_dim = virtual_nodes * virtual_nodes + virtual_nodes
    sae = SparseAutoencoder(input_dim=sae_input_dim, latent_dim=latent_dim)
    
    # Load trained weights if available
    weights_loaded = {'bdh': False, 'sae': False}
    
    if load_weights:
        bdh_weights_path = os.path.join(PROJECT_ROOT, 'Model-Child-Mind', 'bdh_child_mind.pth')
        sae_weights_path = os.path.join(PROJECT_ROOT, 'Model-Child-Mind', 'sae_child_mind.pth')
        
        if os.path.exists(bdh_weights_path):
            try:
                bdh_extractor.load_state_dict(torch.load(bdh_weights_path, map_location='cpu'))
                weights_loaded['bdh'] = True
                print(f"[INFO] Loaded BDH weights from {bdh_weights_path}")
            except Exception as e:
                print(f"[WARNING] Could not load BDH weights: {e}")
        
        if os.path.exists(sae_weights_path):
            try:
                sae.load_state_dict(torch.load(sae_weights_path, map_location='cpu'))
                weights_loaded['sae'] = True
                print(f"[INFO] Loaded SAE weights from {sae_weights_path}")
            except Exception as e:
                print(f"[WARNING] Could not load SAE weights: {e}")
    
    return {
        'bdh_extractor': bdh_extractor,
        'sae': sae,
        'imputer': SynapticKNNImputer(k=5),
        'weights_loaded': weights_loaded
    }

def load_pciat_boosters():
    """Load trained PCIAT booster models if available."""
    import joblib
    
    booster_dir = os.path.join(PROJECT_ROOT, 'Model-Child-Mind')
    booster_files = {
        'lgbm': os.path.join(booster_dir, 'lgbm_pciat.pkl'),
        'xgb': os.path.join(booster_dir, 'xgb_pciat.pkl'),
        'catboost': os.path.join(booster_dir, 'catboost_pciat.pkl')
    }
    
    boosters = {}
    for name, path in booster_files.items():
        if os.path.exists(path):
            try:
                boosters[name] = joblib.load(path)
                print(f"[INFO] Loaded {name} booster from {path}")
            except Exception as e:
                print(f"[WARNING] Could not load {name} booster: {e}")
    
    return boosters if boosters else None

def predict_pciat_score(latent_features, boosters=None):
    """
    Predict PCIAT score from SAE latent features.
    Uses trained boosters if available, otherwise uses a deterministic formula.
    """
    # latent_features: (1, latent_dim) tensor
    latent_np = latent_features.detach().cpu().numpy()
    
    if boosters and len(boosters) > 0:
        # Use trained boosters - ensemble prediction
        predictions = []
        for name, model in boosters.items():
            try:
                pred = model.predict(latent_np)
                predictions.append(pred[0])
            except Exception as e:
                print(f"[WARNING] {name} prediction failed: {e}")
        
        if predictions:
            pciat_score = np.mean(predictions)
            return float(pciat_score), True  # True = real prediction
    
    # Fallback: Use feature statistics (NOT a real prediction)
    # This is a placeholder - won't be accurate without training
    latent_mean = latent_np.mean()
    latent_std = latent_np.std()
    latent_max = latent_np.max()
    
    # More sophisticated heuristic based on latent statistics
    pciat_score = 40 + latent_mean * 15 + latent_std * 8 + latent_max * 2
    pciat_score = max(0, min(100, pciat_score))
    
    return float(pciat_score), False  # False = heuristic

def load_psychosis_model(num_nodes=105, load_weights=True):
    """Load and return Psychosis model components with trained weights."""
    if not PSYCHOSIS_MODEL_AVAILABLE:
        raise RuntimeError("Psychosis model components not available")
    
    bdh_net = BDHNet(num_nodes=num_nodes)
    spike_encoder = SpikeEncoder(method='rate')
    
    # Load trained weights if available
    weights_loaded = False
    
    if load_weights:
        # Try Model-Physcosis folder first, then root
        weight_paths = [
            os.path.join(PROJECT_ROOT, 'Model-Physcosis', 'bdh_model.pth'),
            os.path.join(PROJECT_ROOT, 'bdh_model.pth')
        ]
        
        for weight_path in weight_paths:
            if os.path.exists(weight_path):
                try:
                    bdh_net.load_state_dict(torch.load(weight_path, map_location='cpu'))
                    weights_loaded = True
                    print(f"[INFO] Loaded BDHNet weights from {weight_path}")
                    break
                except Exception as e:
                    print(f"[WARNING] Could not load weights from {weight_path}: {e}")
    
    return {
        'bdh_net': bdh_net,
        'spike_encoder': spike_encoder,
        'weights_loaded': weights_loaded
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_speech_input(patient_state, input_dim=10):
    """Generate input tensor based on patient state for Speech model."""
    input_data = torch.randn(1, input_dim) * 0.1
    
    if "Tremor" in patient_state:
        input_data[0, 0:3] += 2.0  # Strong tremor signal in neurons 0-2
    elif "Disfluency" in patient_state or "Stutter" in patient_state:
        input_data[0, 3:5] += 2.0  # Stutter signal in neurons 3-4
    elif "Fatigue" in patient_state:
        input_data[0, 5:8] += 1.5  # Fatigue signal in neurons 5-7
    
    return input_data

def run_speech_inference(snn, analyzer, dashboard, input_data):
    """Run complete inference through Speech model pipeline."""
    # Forward pass through SNN
    snn_output = snn.forward(input_data)
    hidden_state = snn_output["hidden_state"]
    
    # Update synaptic weights with Hebbian learning
    weights = snn.update_synapses(input_data, hidden_state)
    
    # Analyze for concept probes
    analysis_results = analyzer.analyze(snn_output, weights)
    
    # Generate clinical report
    report = dashboard.generate_report(analysis_results)
    
    return {
        'hidden_state': hidden_state,
        'weights': weights,
        'analysis': analysis_results,
        'report': report
    }

def get_age_group(age):
    """Determine age group bucket for Child-Mind model."""
    if age < 10:
        return 0  # Young
    elif age < 16:
        return 1  # Middle
    else:
        return 2  # Older

def generate_actigraphy_sample(sequence_length=230, num_channels=5):
    """Generate sample actigraphy data for testing."""
    # Simulate 5-channel actigraphy: X, Y, Z, enmo, anglez
    data = np.random.randn(sequence_length, num_channels) * 0.5
    
    # Add realistic patterns
    t = np.linspace(0, 4*np.pi, sequence_length)
    data[:, 0] += 0.3 * np.sin(t)  # X has periodic motion
    data[:, 1] += 0.3 * np.cos(t)  # Y has periodic motion
    data[:, 3] = np.abs(data[:, 3])  # enmo is non-negative
    
    return torch.tensor(data, dtype=torch.float32)

def generate_fmri_sample(num_nodes=105, sequence_length=230):
    """Generate sample rs-fMRI data for testing."""
    # FNC: Upper triangle of correlation matrix (5460 values for 105 nodes)
    fnc = np.random.randn(num_nodes * (num_nodes - 1) // 2) * 0.3
    
    # ICN timecourses: (T, nodes)
    icn = np.random.randn(sequence_length, num_nodes) * 0.5
    
    return {
        'fnc': torch.tensor(fnc, dtype=torch.float32),
        'icn': torch.tensor(icn, dtype=torch.float32)
    }

# ============================================================================
# SPEECH-TO-TEXT TRANSCRIPTION
# ============================================================================

# Check if speech recognition is available
try:
    import speech_recognition as sr
    TRANSCRIPTION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False

# Stop words to filter out for word analysis
STOP_WORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", 
    "of", "to", "in", "for", "with", "by", "as", "it", "this", "that",
    "be", "are", "was", "were", "been", "being", "have", "has", "had",
    "do", "does", "did", "from", "up", "down", "out", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", 
    "where", "why", "how", "all", "any", "both", "each", "few", 
    "more", "most", "other", "some", "such", "no", "nor", "not", 
    "only", "own", "same", "so", "than", "too", "very", "can", 
    "will", "just", "don", "should", "now", "i", "you", "we", "they",
    "he", "she", "my", "your", "me", "um", "uh", "like", "know"
}

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using SpeechRecognition.
    Returns transcript string and word list.
    """
    if not TRANSCRIPTION_AVAILABLE:
        return None, None, "SpeechRecognition not installed"
    
    try:
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        
        # Use Google Web Speech API (free for limited use)
        transcript = recognizer.recognize_google(audio)
        
        # Tokenize into words
        words = transcript.lower().split()
        clean_words = [w.strip('.,?!;:') for w in words if w.strip('.,?!;:')]
        
        return transcript, clean_words, None
        
    except sr.UnknownValueError:
        return None, None, "Could not understand audio"
    except sr.RequestError as e:
        return None, None, f"API error: {e}"
    except Exception as e:
        return None, None, f"Transcription failed: {e}"

def analyze_words_during_states(words, detected_states, concept_probes):
    """
    Analyze which words occur during detected anomaly states.
    Returns word frequency data similar to daic_analysis_report.txt.
    """
    from collections import Counter
    
    if not words:
        return {}
    
    # Count all non-stop words
    valid_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    word_counts = Counter(valid_words)
    
    # Get top determining words (most frequent non-stop words)
    top_words = word_counts.most_common(15)
    
    # Build result similar to daic_analysis_report.txt format
    result = {
        'total_words': len(words),
        'valid_words': len(valid_words),
        'unique_words': len(set(valid_words)),
        'top_determining_words': top_words,
        'state_indicators': {}
    }
    
    # For each detected state, track words
    for state in detected_states:
        # In real DAIC, words are tracked per-frame during that state
        # Here we just use all words as indicators since we have full audio
        result['state_indicators'][state] = top_words[:10]
    
    # Extract sample sentences (split by common sentence endings)
    sentences = []
    if words:
        transcript = ' '.join(words)
        for sep in ['. ', '? ', '! ']:
            transcript = transcript.replace(sep, '.|')
        sentences = [s.strip() for s in transcript.split('|') if len(s.strip()) > 5][:5]
    
    result['sample_sentences'] = sentences
    
    return result

def generate_daic_style_report(
    input_source,
    detected_states,
    concept_probes,
    hidden_norm,
    hidden_mean,
    sparsity,
    transcript=None,
    word_analysis=None
):
    """
    Generate a detailed report similar to daic_analysis_report.txt
    """
    from datetime import datetime
    
    report = f"""GenZERO Speech Analysis Report
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INPUT DATA
----------
Data Source: {input_source}
Model: BDH Spiking Neural Network (ser_bdh_model.pth)

"""
    
    # Detected States Section
    report += "DETECTED STATES\n"
    report += "-" * 20 + "\n"
    if detected_states:
        for state in detected_states:
            report += f"⚠️ {state} Detected\n"
    else:
        report += "✅ No anomalies detected - Patient Stable\n"
    
    report += "\n"
    
    # Neural Statistics
    report += f"""NEURAL STATISTICS
-----------------
Hidden Norm Mean: {hidden_norm:.4f}
Mean Activation: {hidden_mean:.6f}
Sparsity: {sparsity:.1f}%

"""
    
    # Word Analysis Section (like daic_analysis_report.txt)
    if word_analysis:
        report += """GLOBAL STATE-DETERMINING INDICATORS
------------------------------------
"""
        # Top determining words
        top_words = word_analysis.get('top_determining_words', [])
        if top_words:
            word_str = ', '.join([f"{w}({c})" for w, c in top_words])
            report += f"Top Determining Words: {word_str}\n\n"
        
        report += "\n"
        
        # Per-state word indicators
        state_indicators = word_analysis.get('state_indicators', {})
        if state_indicators:
            report += "STATE INDICATORS\n"
            report += "-" * 20 + "\n"
            for state, words in state_indicators.items():
                if words:
                    word_str = ', '.join([f"{w}({c})" for w, c in words])
                    report += f"{state}: {word_str}\n"
    
    report += "\n"
    
    # Concept Probe Analysis
    report += "CONCEPT PROBE ANALYSIS\n"
    report += "-" * 20 + "\n"
    for concept, data in concept_probes.items():
        status = "ACTIVE ⚠️" if data.get('active', False) else "Normal ✓"
        report += f"""
{concept.upper()}:
  Status: {status}
  Activation Level: {data.get('activation_level', 0):.4f}
  Synaptic Strength: {data.get('synaptic_strength', 0):.4f}
"""
    
    report += f"""
{'=' * 50}
Report generated by GenZERO BDH Speech Analysis System
"""
    
    return report

# ============================================================================
# GEMINI INTEGRATION
# ============================================================================

def get_gemini_api_key():
    """Retrieve Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")

# ============================================================================
# STATUS CHECK
# ============================================================================

def get_backend_status():
    """Return status of all backend components."""
    return {
        'speech': SPEECH_MODEL_AVAILABLE,
        'child_mind': CHILD_MIND_MODEL_AVAILABLE,
        'psychosis': PSYCHOSIS_MODEL_AVAILABLE,
        'transcription': TRANSCRIPTION_AVAILABLE,
        'gemini': bool(get_gemini_api_key())
    }

if __name__ == "__main__":
    print("Backend Integration Status:")
    status = get_backend_status()
    for model, available in status.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {model}: {'Available' if available else 'Not Available'}")

