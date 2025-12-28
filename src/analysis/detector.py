import torch
import numpy as np

class ConceptProbes:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Define "concepts" as specific patterns or neuron indices
        self.concepts = {
            "tremor": {"indices": [0, 1, 2], "threshold": 0.5},
            "stutter": {"indices": [3, 4], "threshold": 0.5},
            "fatigue": {"indices": [5, 6, 7], "threshold": 0.4}
        }
        print("Initialized Concept Probes for SNN")

    def probe(self, hidden_state, weights):
        """
        Checks if specific concepts are active based on hidden state and synaptic weights.
        """
        results = {}
        
        # Convert hidden_state to numpy if it's a tensor
        if isinstance(hidden_state, torch.Tensor):
            activations = hidden_state.detach().numpy().flatten()
        else:
            activations = np.array(hidden_state).flatten()

        for concept_name, config in self.concepts.items():
            indices = config["indices"]
            threshold = config["threshold"]
            
            # Check mean activation of the concept neurons
            if len(indices) <= len(activations):
                concept_activation = np.mean(activations[indices])
                is_active = concept_activation > threshold
                
                # Also check synaptic strength (plasticity) for these neurons
                # This simulates "learning" the concept
                # We look at the diagonal or sum of weights for these indices
                if isinstance(weights, torch.Tensor):
                    w = weights.detach().numpy()
                else:
                    w = weights
                
                # Average weight strength for connections to these neurons
                # Assuming weights is (hidden, hidden)
                concept_weight = np.mean(w[indices, :][:, indices])
                
                results[concept_name] = {
                    "active": bool(is_active),
                    "activation_level": float(concept_activation),
                    "synaptic_strength": float(concept_weight)
                }
            else:
                results[concept_name] = {"active": False, "error": "Index out of bounds"}
                
        return results

class AnalysisLayer:
    def __init__(self):
        print("Initialized Analysis & Detection Layer")
        self.probes = None

    def analyze(self, snn_output, weights=None):
        print("Analyzing spike patterns for anomalies...")
        
        hidden_state = snn_output.get("hidden_state")
        
        # Initialize probes if not already done (lazy init based on state size)
        if self.probes is None and hidden_state is not None:
            dim = hidden_state.shape[1]
            self.probes = ConceptProbes(dim, dim)
            
        results = {
            "raw_output": snn_output.get("output"),
            "anomalies": []
        }
        
        if self.probes:
            probe_results = self.probes.probe(hidden_state, weights)
            results["concept_probes"] = probe_results
            
            # High-level detection logic
            if probe_results["tremor"]["active"]:
                results["anomalies"].append("Tremor Detected")
            if probe_results["stutter"]["active"]:
                results["anomalies"].append("Speech Disfluency Detected")
                
        return results
