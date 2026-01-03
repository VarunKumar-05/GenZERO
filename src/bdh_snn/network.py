import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianSynapse(nn.Module):
    def __init__(self, size, alpha=0.99, eta=0.01):
        super().__init__()
        # Initialize with small random weights
        self.weights = nn.Parameter(torch.randn(size, size) * 0.1)
        
        # Hyperparameters
        self.alpha = alpha  # Decay (Forgetting factor)
        self.eta = eta      # Learning rate
        
        # Register buffer for stability checking if needed, or just keep as params
        # We don't want backprop to optimize alpha/eta in this unsupervised setting typically
        # but making them parameters allows for meta-learning if we wanted. 
        # For this fix, let's keep them fixed to enforce stability.
        
    def forward(self, x):
        # Apply the synaptic weights: y = x @ W
        return torch.matmul(x, self.weights)

    def update(self, pre_synaptic, post_synaptic):
        """
        Oja's Rule-style Hebbian Update or simple Decay + Hebbian.
        
        Standard Hebb: dW = eta * pre * post
        Decay Hebb:  W = alpha * W + eta * pre * post
        Oja's:       dW = eta * (pre * post - post^2 * W) [Enforces normalization]
        
        We will use Decay Hebb + Soft Clamp for stability.
        """
        # Ensure dimensions align
        if pre_synaptic.dim() == 2:
            # Batch mode: Average update over batch
            # pre: (B, N), post: (B, N) -> outer: (B, N, N) -> mean -> (N, N)
            # We want outer product correlation
            outer = torch.matmul(pre_synaptic.t(), post_synaptic) / pre_synaptic.size(0)
        else:
            outer = torch.matmul(pre_synaptic.unsqueeze(1), post_synaptic.unsqueeze(0))

        # Update Rule
        # 1. Decay
        self.weights.data.mul_(self.alpha)
        
        # 2. Hebbian Growth
        self.weights.data.add_(self.eta * outer)
        
        # 3. Soft Clamp / Normalization to prevent explosion
        # If any weight > 5.0, scale it down
        max_val = self.weights.data.abs().max()
        if max_val > 5.0:
            self.weights.data.div_(max_val / 5.0)

class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        # Simple Linear Attention approximation (feature map based)
        # For simplicity in this dummy/demo, we'll use a scaled dot product 
        # but efficiently implemented or just a self-attention block.
        # Given "Linear Attention" usually implies kernel trick methods (Katharopoulos et al.),
        # we will implement a lightweight version: Q(K^T V) / Q K^T
        
        q, k, v = self.q(x), self.k(x), self.v(x)
        
        # Normalization (Re-centering) for stability
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Attention Output (simplified for 1D/Batch sequence)
        # If x is (Batch, Dim), this is just self-projection. 
        # Assuming x is (Batch, Dim) representing a timestep frame.
        # We model "Context" as an aggregation. 
        
        # Let's keep it simple: Gating mechanism
        # Gated Linear Unit style
        gate = torch.sigmoid(q)
        out = gate * v
        return out

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        print("Initialized BDH Spiking Neural Network (Refactored)")
        
        # 1. Input Normalization (Crucial fix for unscaled inputs)
        self.norm = nn.LayerNorm(input_dim)
        
        # 2. Input Projection
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # 3. Recurrent/Hebbian Synapses
        self.synapses = HebbianSynapse(hidden_dim, alpha=0.99, eta=0.01)
        
        # 4. Attention mechanism
        self.attention = LinearAttention(hidden_dim)
        
        # 5. Output
        self.output_layer = nn.Linear(hidden_dim, 10) 
        
        # Activity Regularization
        self.activity_decay = 0.5

    def forward(self, input_features):
        # Convert to tensor
        if not isinstance(input_features, torch.Tensor):
            import numpy as np
            if isinstance(input_features, list):
                input_features = np.array(input_features)
            input_features = torch.tensor(input_features, dtype=torch.float32)
        
        # Step 1: Normalize Input
        x = self.norm(input_features)
        
        # Step 2: Input -> Hidden
        ff_drive = self.input_layer(x)
        
        # Step 3: Apply Recurrent/Hebbian Weights
        # Initial hidden state approximation or previous state?
        # For feedforward-only Hebbian (Input->Hidden), we just use ff_drive.
        # But if we want 'synapses' to matter during forward, they should act on x or hidden.
        # Let's assume 'synapses' are lateral inhibition/excitation within hidden layer.
        # lateral = self.synapses(prev_hidden) ... but we are stateless in this API call structure
        # (train_daic loop doesn't pass state back in explicitly except for updating weights).
        
        # We will apply synapses as a transformation of the feedforward drive itself 
        # (self-associative memory effect).
        lateral = self.synapses(ff_drive)
        
        # Step 4: Spiking Nonlinearity (ReLU approximation)
        hidden = torch.relu(ff_drive + 0.1 * lateral) 
        
        # Step 5: Attention
        context = self.attention(hidden)
        
        # Step 6: Output
        output = self.output_layer(context)
        
        return {"output": output, "hidden_state": hidden}

    def update_synapses(self, input_features, hidden_state):
        """
        Updates synaptic weights.
        """
        # In this architecture, we update the lateral synapses based on hidden state co-activity.
        # "Fire together, wire together"
        self.synapses.update(hidden_state, hidden_state)
        return self.synapses.weights.data
