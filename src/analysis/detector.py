import torch
import numpy as np

class ConceptProbes:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Calibration stats
        self.calibrated = False
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        self.calibration_buffer = []
        self.CALIBRATION_FRAMES = 200 # First 200 frames used for baseline
        
        # Define concepts with Z-Score thresholds instead of raw values
        # "Tremor" = indices [0..2], threshold = 2.0 std devs above mean
        self.concepts = {
            "tremor": {"indices": [0, 1, 2], "z_threshold": 2.0},
            "stutter": {"indices": [3, 4], "z_threshold": 2.0},
            "fatigue": {"indices": [5, 6, 7], "z_threshold": -1.5} # Low activity?
        }
        print("Initialized Calibrated Concept Probes")

    def calibrate(self, hidden_state):
        # Accumulate data
        if isinstance(hidden_state, torch.Tensor):
            vals = hidden_state.detach().cpu().numpy()
        else:
            vals = np.array(hidden_state)
            
        self.calibration_buffer.append(vals)
        
        if len(self.calibration_buffer) >= self.CALIBRATION_FRAMES:
            # Compute stats
            all_data = np.concatenate(self.calibration_buffer, axis=0) # (N_frames, Hidden)
            self.baseline_mean = np.mean(all_data, axis=0)
            self.baseline_std = np.std(all_data, axis=0) + 1e-6 # Avoid div0
            self.calibrated = True
            print(f"Calibration Complete. Mean Norm: {np.mean(self.baseline_mean):.3f}")
            # Clear memory
            self.calibration_buffer = []

    def probe(self, hidden_state, weights):
        """
        Checks for anomalies using Z-Scores relative to baseline.
        """
        results = {}
        
        # If not calibrated, return neutral
        if not self.calibrated:
            self.calibrate(hidden_state)
            return {}

        if isinstance(hidden_state, torch.Tensor):
            activations = hidden_state.detach().cpu().numpy().flatten()
        else:
            activations = np.array(hidden_state).flatten()

        # Calculate Z-Scores for the whole layer
        z_scores = (activations - self.baseline_mean) / self.baseline_std

        for concept_name, config in self.concepts.items():
            indices = config["indices"]
            z_thresh = config["z_threshold"]
            
            if max(indices) < len(activations):
                # Check metrics on specific neurons
                concept_z = np.mean(z_scores[indices])
                
                # Logic: 
                # Positive threshold -> Trigger if ABOVE (hyperactivity/tremor)
                # Negative threshold -> Trigger if BELOW (hypoactivity/fatigue)
                if z_thresh > 0:
                    is_active = concept_z > z_thresh
                else:
                    is_active = concept_z < z_thresh
                
                # Synaptic Strength analysis (optional/supplementary)
                if isinstance(weights, torch.Tensor):
                    w = weights.detach().cpu().numpy()
                else:
                    w = weights
                    
                concept_weight = np.mean(w[indices, :][:, indices]) if w is not None else 0.0
                
                # Raw activation for reporting (unnormalized)
                raw_act = np.mean(activations[indices])
                
                results[concept_name] = {
                    "active": bool(is_active),
                    "z_score": float(concept_z),
                    "activation_level": float(raw_act),
                    "synaptic_strength": float(concept_weight)
                }
            else:
                results[concept_name] = {"active": False, "error": "Index out of bounds"}
                
        return results

class AnalysisLayer:
    def __init__(self):
        print("Initialized Analysis & Detection Layer (Calibrated)")
        self.probes = None

    def analyze(self, snn_output, weights=None):
        hidden_state = snn_output.get("hidden_state")
        
        # Init probes lazily
        if self.probes is None and hidden_state is not None:
            dim = hidden_state.shape[1] if hidden_state.ndim > 1 else hidden_state.shape[0]
            self.probes = ConceptProbes(dim, dim)
            
        results = {
            "raw_output": snn_output.get("output"),
            "anomalies": []
        }
        
        if self.probes:
            probe_results = self.probes.probe(hidden_state, weights)
            results["concept_probes"] = probe_results
            
            if not probe_results:
                return results # Still calibrating
                
            # Logic
            if probe_results.get("tremor", {}).get("active"):
                results["anomalies"].append("Tremor Detected")
                
            if probe_results.get("stutter", {}).get("active"):
                results["anomalies"].append("Speech Disfluency Detected")
                
            if probe_results.get("fatigue", {}).get("active"):
                results["anomalies"].append("Cognitive Fatigue")
                
        return results
