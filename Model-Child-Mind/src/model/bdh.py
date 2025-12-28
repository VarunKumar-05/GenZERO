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
