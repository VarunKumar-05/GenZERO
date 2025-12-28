import torch
import torch.nn as nn

class HebbianSynapse(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(size, size))
        self.plasticity = nn.Parameter(torch.ones(size, size) * 0.1)

    def update(self, pre_synaptic, post_synaptic):
        # Simple Hebbian update rule: delta_w = eta * pre * post
        delta_w = self.plasticity * torch.matmul(pre_synaptic.t(), post_synaptic)
        self.weights.data += delta_w

class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Placeholder for Linear Attention components

    def forward(self, x):
        # O(T) attention mechanism placeholder
        return x

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        print("Initialized BDH Spiking Neural Network with Hebbian Plasticity")
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.synapses = HebbianSynapse(hidden_dim)
        self.attention = LinearAttention(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 10) # Example output classes

    def forward(self, input_features):
        # Convert input to tensor if needed
        if not isinstance(input_features, torch.Tensor):
            # Handle list or numpy array
            import numpy as np
            if isinstance(input_features, list):
                input_features = np.array(input_features)
            input_features = torch.tensor(input_features, dtype=torch.float32)
        
        print("Processing features through SNN...")
        
        # Forward pass
        hidden = torch.relu(self.input_layer(input_features))
        
        # Apply attention
        context = self.attention(hidden)
        
        # Output
        output = self.output_layer(context)
        
        return {"output": output, "hidden_state": hidden}

    def update_synapses(self, input_features, hidden_state):
        """
        Updates synaptic weights based on Hebbian learning rule.
        """
        # Ensure input is tensor
        if not isinstance(input_features, torch.Tensor):
             input_features = torch.tensor(input_features, dtype=torch.float32)
        
        # We need to map input to hidden dimensions for the update if they are different sizes
        # For simplicity in this demo, we'll just update the internal synapse weights 
        # based on the hidden state correlation (self-organization)
        
        # Hebbian update: correlated firing strengthens connections
        # We use the hidden state to update the recurrent synapses (if we had them)
        # or the input-to-hidden weights.
        
        # Let's simulate updating the 'synapses' layer (hidden-to-hidden or similar)
        # Here we assume 'synapses' represents a recurrent connection or internal memory
        
        # Create a pseudo-pre-synaptic and post-synaptic activity from the hidden state
        # In a real RNN/SNN, this would be h_t-1 and h_t
        
        # For demonstration, we'll just use the current hidden state as both for self-reinforcement
        self.synapses.update(hidden_state, hidden_state)
        
        return self.synapses.weights.data
