import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadBDHLayer(nn.Module):
    def __init__(self, num_nodes, num_heads=4):
        """
        Args:
            num_nodes: Number of ICN channels (105).
            num_heads: Number of plasticity heads.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        
        # Learnable plasticity parameters per head
        # Alpha (decay): Initialized to diverse values to encourage multi-scale memory
        # e.g., Head 0: 0.9 (Long-term), Head 1: 0.5 (Short-term), ...
        self.alpha = nn.Parameter(torch.tensor([0.9, 0.7, 0.5, 0.1][:num_heads] + [0.5]*(num_heads-4)))
        self.eta = nn.Parameter(torch.ones(num_heads) * 0.01)
        
    def forward(self, x, w_init, mask=None):
        """
        Args:
            x: Spike inputs (Batch, Time, Nodes)
            w_init: Initial weights (Batch, Nodes, Nodes) from FNC
            mask: (Batch, Time) mask for valid time steps.
        Returns:
            w_final: Final weights (Batch, Heads, Nodes, Nodes)
            y_seq: Output sequence (Batch, Time, Heads, Nodes)
        """
        batch_size, T, nodes = x.shape
        
        # Expand w_init for each head: (B, H, N, N)
        w = w_init.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        y_seq = []
        
        # Pre-compute alphas/etas shapes for broadcasting
        # alpha: (1, H, 1, 1)
        alpha_expanded = self.alpha.view(1, self.num_heads, 1, 1)
        eta_expanded = self.eta.view(1, self.num_heads, 1, 1)

        # Recurrent loop
        for t in range(T):
            xt = x[:, t, :] # (B, N)
            
            # Mask handling
            if mask is not None:
                m_t = mask[:, t].view(batch_size, 1, 1, 1).float() # (B, 1, 1, 1)
            else:
                m_t = 1.0
            
            # Forward pass: y_t = W_t * x_t
            # w: (B, H, N, N), xt: (B, N) -> xt expanded: (B, 1, N, 1)
            # Result: (B, H, N, 1) -> squeeze -> (B, H, N)
            xt_expanded = xt.view(batch_size, 1, nodes, 1)
            yt = torch.matmul(w, xt_expanded).squeeze(-1) # (B, H, N)
            y_seq.append(yt)
            
            # Hebbian Update: delta_W = eta * (xt * xt.T)
            # Outer product: (B, 1, N, 1) @ (B, 1, 1, N) -> (B, 1, N, N)
            outer = torch.matmul(xt_expanded, xt.view(batch_size, 1, 1, nodes))
            
            # Update W
            w_new = alpha_expanded * w + eta_expanded * outer
            
            # Apply mask
            w = m_t * w_new + (1.0 - m_t) * w
                
        y_seq = torch.stack(y_seq, dim=1) # (B, T, H, N)
        return w, y_seq

class TrajectoryPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Attention Mechanism
        # Keys/Values are the trajectory states
        # Query is a learnable "summary" vector
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.val_proj = nn.Linear(input_dim, hidden_dim)
        self.scale = 1.0 / math.sqrt(hidden_dim)
        
    def forward(self, y_seq, mask=None):
        """
        y_seq: (B, T, InputDim)
        """
        keys = self.key_proj(y_seq)   # (B, T, Hidden)
        values = self.val_proj(y_seq) # (B, T, Hidden)
        
        # Attention scores
        # (B, 1, Hidden) @ (B, Hidden, T) -> (B, 1, T)
        scores = torch.matmul(self.query, keys.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Mask is (B, T). Fill 0s with -inf
            # Expand mask to (B, 1, T)
            mask_expanded = mask.unsqueeze(1)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # Aggregate
        # (B, 1, T) @ (B, T, Hidden) -> (B, 1, Hidden)
        context = torch.matmul(attn_weights, values).squeeze(1)
        return context

class BDHNet(nn.Module):
    def __init__(self, num_nodes=105, num_classes=2, num_heads=4, pool_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        
        self.bdh_layer = MultiHeadBDHLayer(num_nodes, num_heads=num_heads)
        
        # Trajectory Head (predict next input)
        # We predict based on the mean of heads for simplicity
        self.traj_head = nn.Linear(num_nodes, num_nodes)
        
        # Pooling
        self.input_dim = num_heads * num_nodes
        self.pooling = TrajectoryPooling(self.input_dim, pool_dim)
        
        # Fused Classifier
        # Input: Flattened Weights (H*N*N) + Pooled Trajectory (PoolDim)
        self.flat_size = num_heads * num_nodes * num_nodes
        self.classifier_input = self.flat_size + pool_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.classifier_input, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, adj, mask=None):
        """
        x: (B, T, N)
        adj: (B, N, N)
        mask: (B, T)
        """
        # Run BDH
        # w_final: (B, H, N, N)
        # y_seq: (B, T, H, N)
        w_final, y_seq = self.bdh_layer(x, adj, mask)
        
        # 1. Structural Feature: Flattened weights
        w_flat = w_final.view(w_final.size(0), -1)
        
        # 2. Dynamic Feature: Pooled trajectory
        # Flatten heads in y_seq: (B, T, H*N)
        y_seq_flat = y_seq.view(y_seq.size(0), y_seq.size(1), -1)
        y_pooled = self.pooling(y_seq_flat, mask) # (B, PoolDim)
        
        # Fused Feature
        fused = torch.cat([w_flat, y_pooled], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
        # Trajectory Prediction (Aux Loss)
        # Average heads for prediction: (B, T, N)
        y_mean = y_seq.mean(dim=2)
        x_next_pred = self.traj_head(y_mean)
        
        return logits, x_next_pred, w_final
