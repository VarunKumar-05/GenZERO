import torch
import torch.nn as nn
from src.model.bdh import BDHLayer

class BDHFeatureExtractor(nn.Module):
    def __init__(self, input_channels=5, virtual_nodes=64, num_age_groups=3):
        super().__init__()
        # Project 5 physical channels to 64 virtual graph nodes
        self.input_proj = nn.Linear(input_channels, virtual_nodes)
        
        # BDH Core
        self.bdh = BDHLayer(num_nodes=virtual_nodes)
        
        # Initial weights (learnable since no FNC for child mind)
        # Age Gating: Multiple prototypes
        self.w_init_bank = nn.Parameter(torch.randn(num_age_groups, virtual_nodes, virtual_nodes) * 0.01)
        
    def forward(self, x, mask=None, age_group=None):
        """
        x: (B, T, 5)
        age_group: (B,) indices
        """
        B, T, C = x.shape
        
        # Project to virtual nodes
        x_virt = self.input_proj(x) # (B, T, 64)
        
        # Spike encoding? We can do rudimentary rate coding by Sigmoid
        x_spikes = torch.sigmoid(x_virt)
        
        # Select w_init based on age
        if age_group is not None:
             w = self.w_init_bank[age_group] # (B, Nodes, Nodes)
        else:
             # Default to mean or index 1
             w = self.w_init_bank.mean(dim=0).unsqueeze(0).expand(B, -1, -1)
        
        # Run BDH
        w_final, y_seq = self.bdh(x_spikes, w, mask)
        
        # Return features
        # 1. Flattened Memory Graph (B, 64*64)
        w_flat = w_final.view(B, -1)
        
        # 2. Mean Pooled Activity (B, 64)
        activity = y_seq.mean(dim=1)
        
        return torch.cat([w_flat, activity], dim=1) 
