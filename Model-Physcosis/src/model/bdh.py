import torch
import torch.nn as nn
import torch.nn.functional as F

class BDHLayer(nn.Module):
    def __init__(self, num_nodes, alpha=0.9, eta=0.01):
        """
        Args:
            num_nodes: Number of ICN channels (105).
            alpha: Decay factor for weights (forgetting). 1.0 = no forgetting.
            eta: Learning rate for Hebbian update.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.eta = nn.Parameter(torch.tensor(eta))
        
        # Trainable parameters for the update rule?
        # Maybe a gating mechanism?
        # The user specified "Linear Attention" which is similar to Fast Weights.
        # We'll stick to the simple Hebbian rule for now as requested.
        
    def forward(self, x, w_init, mask=None):
        """
        Args:
            x: Spike inputs (Batch, Time, Nodes)
            w_init: Initial weights (Batch, Nodes, Nodes) from FNC
            mask: (Batch, Time) mask for valid time steps.
        Returns:
            w_final: Final weights (Batch, Nodes, Nodes)
            y_seq: Output sequence (Batch, Time, Nodes)
        """
        batch_size, T, nodes = x.shape
        
        w = w_init
        y_seq = []
        
        # Recurrent loop
        for t in range(T):
            xt = x[:, t, :] # (B, N)
            
            if mask is not None:
                m_t = mask[:, t].unsqueeze(1).unsqueeze(2) # (B, 1, 1)
            else:
                m_t = 1.0
            
            # Forward pass through current graph
            # y_t = W_t * x_t (Linear activation)
            # (B, N, N) @ (B, N, 1) -> (B, N, 1)
            yt = torch.bmm(w, xt.unsqueeze(2)).squeeze(2) # (B, N)
            
            y_seq.append(yt)
            
            # Hebbian Update
            # delta_W = eta * (xt * xt.T)
            # Outer product
            outer = torch.bmm(xt.unsqueeze(2), xt.unsqueeze(1)) # (B, N, N)
            
            # Update W
            # Apply decay and update only for valid time steps
            w_new = self.alpha * w + self.eta * outer
            
            # If mask is 0, keep old w
            if isinstance(m_t, torch.Tensor):
                m_t = m_t.float()
                w = m_t * w_new + (1.0 - m_t) * w
            else:
                w = w_new
                
        y_seq = torch.stack(y_seq, dim=1) # (B, T, N)
        return w, y_seq

class BDHNet(nn.Module):
    def __init__(self, num_nodes=105, num_classes=2):
        super().__init__()
        self.bdh_layer = BDHLayer(num_nodes)
        
        # Classification Head (takes flattened W_final)
        # Input size: N * N
        # We can perform dimensionality reduction or just use a linear layer
        self.flat_size = num_nodes * num_nodes
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Trajectory Head (takes y_seq)
        # Predicts next time step input x_{t+1} (or its value)
        # Input: N (y_t) -> Output: N (x_hat_t+1)
        self.traj_head = nn.Linear(num_nodes, num_nodes)
        
    def forward(self, x, adj, mask=None):
        """
        x: (B, T, N)
        adj: (B, N, N)
        mask: (B, T)
        """
        w_final, y_seq = self.bdh_layer(x, adj, mask)
        
        # Classification
        w_flat = w_final.view(w_final.size(0), -1)
        logits = self.classifier(w_flat)
        
        # Trajectory Prediction
        # Predict x_{t+1} from y_t
        # Note: We are predicting the *input* of the next step (ICN activity)
        # y_seq is the "state" representation at t
        x_next_pred = self.traj_head(y_seq)
        
        return logits, x_next_pred, w_final
