import torch
import torch.nn as nn
import torch.optim as optim

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=32, hidden_dim=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU() # Functionally ensures non-negative sparse activations? Or just Latent code.
            # Usually strict sparsity is enforced by L1 loss on activations
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def train_sae(features, input_dim=4096, latent_dim=32, epochs=20, batch_size=32, device='cpu', l1_lambda=1e-5):
    """
    features: Numpy array (N, 4096)
    """
    model = SparseAutoencoder(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Convert to Tensor
    dataset = torch.tensor(features, dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    print("Training Sparse Autoencoder...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, latent = model(batch)
            mse_loss = criterion(recon, batch)
            
            # Sparsity Loss (L1 on latent)
            l1_loss = torch.mean(torch.abs(latent))
            
            loss = mse_loss + l1_lambda * l1_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"SAE Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        
    return model
