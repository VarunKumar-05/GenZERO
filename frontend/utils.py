"""
Utilities module for the GenZERO frontend
Provides integration with the backend models and components
"""
import sys
import os

# Add parent directory to path for backend imports
BACKEND_PATH = os.path.join(os.path.dirname(__file__), '..')
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

import numpy as np

def get_backend_path():
    """Returns the path to the backend source directory"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_snn_model(input_dim=128, hidden_dim=256):
    """Load the Spiking Neural Network model"""
    try:
        from src.bdh_snn.network import SpikingNeuralNetwork
        return SpikingNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
    except ImportError as e:
        print(f"Warning: Could not import SNN model: {e}")
        return None

def load_analysis_layer():
    """Load the Analysis Layer for concept probing"""
    try:
        from src.analysis.detector import AnalysisLayer
        return AnalysisLayer()
    except ImportError as e:
        print(f"Warning: Could not import AnalysisLayer: {e}")
        return None

def load_clinical_dashboard():
    """Load the Clinical Dashboard for visualization"""
    try:
        from src.clinical.dashboard import ClinicalDashboard
        return ClinicalDashboard()
    except ImportError as e:
        print(f"Warning: Could not import ClinicalDashboard: {e}")
        return None

def load_memory_storage():
    """Load the Persistent Memory module"""
    try:
        from src.memory.storage import PersistentMemory
        return PersistentMemory()
    except ImportError as e:
        print(f"Warning: Could not import PersistentMemory: {e}")
        return None

def generate_mock_analysis_results(patient_state="Baseline"):
    """Generate mock analysis results for demo purposes"""
    
    base_activations = {
        "tremor": {"active": False, "activation_level": 0.18, "synaptic_strength": 0.21},
        "stutter": {"active": False, "activation_level": 0.22, "synaptic_strength": 0.19},
        "fatigue": {"active": False, "activation_level": 0.25, "synaptic_strength": 0.24}
    }
    
    anomalies = []
    
    if "Tremor" in patient_state:
        base_activations["tremor"] = {"active": True, "activation_level": 0.87, "synaptic_strength": 0.62}
        anomalies.append("Tremor Detected")
    elif "Disfluency" in patient_state or "Stutter" in patient_state:
        base_activations["stutter"] = {"active": True, "activation_level": 0.91, "synaptic_strength": 0.73}
        anomalies.append("Speech Disfluency Detected")
    elif "Fatigue" in patient_state:
        base_activations["fatigue"] = {"active": True, "activation_level": 0.61, "synaptic_strength": 0.55}
    
    return {
        "raw_output": np.random.randn(10).tolist(),
        "anomalies": anomalies,
        "concept_probes": base_activations
    }

def format_clinical_report(analysis_results):
    """Format analysis results as a clinical report"""
    anomalies = analysis_results.get("anomalies", [])
    probes = analysis_results.get("concept_probes", {})
    
    report = "## Clinical Assessment Report\n\n"
    
    if anomalies:
        report += f"**Alerts:** {', '.join(anomalies)}\n\n"
    else:
        report += "**Status:** Patient Stable\n\n"
    
    report += "### Neuromorphic Concept Activation:\n"
    for concept, data in probes.items():
        status = "Active" if data["active"] else "Inactive"
        report += f"- **{concept.capitalize()}**: {status} "
        report += f"(Activation: {data['activation_level']:.2f}, "
        report += f"Synaptic Strength: {data['synaptic_strength']:.2f})\n"
    
    return report
