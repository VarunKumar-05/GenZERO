import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data_loader.loader import RsFmriDataset
from src.model.bdh import BDHNet
import numpy as np

# Config
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001
LAMBDA_SPARSE = 1e-4
LAMBDA_TRAJ = 0.5
SEQ_LEN = 230
DATA_ROOT = r'd:\Hackathons  & Competitions\Synaptix\Model-Physcosis\psychosis-classification-with-rsfmri\data\train'

def batch_smote(latents, labels):
    """
    Applies simplified SMOTE-like augmentation on the latent batch.
    Focuses on the minority class (SZ=1, BP=0).
    Count: SZ~288, BP~183. So BP is minority?
    Let's check counts dynamically or assume BP (0) is minority.
    """
    # Identify counts
    # classes: 0, 1
    
    # We want to balance the batch.
    # Find minority in *current batch*
    counts = torch.bincount(labels, minlength=2)
    if counts[0] == counts[1]:
        return latents, labels
    
    minority_class = torch.argmin(counts).item()
    maj_count = torch.max(counts).item()
    min_indices = (labels == minority_class).nonzero(as_tuple=True)[0]
    
    if len(min_indices) < 2:
        return latents, labels
        
    # How many to generate?
    diff = maj_count - len(min_indices)
    if diff <= 0:
        return latents, labels
        
    # Generate synthetic samples
    syn_latents = []
    syn_labels = []
    
    for _ in range(diff):
        # Pick two rand
        idx1, idx2 = torch.randint(0, len(min_indices), (2,))
        idx1, idx2 = min_indices[idx1], min_indices[idx2]
        
        alpha = torch.rand(1).item()
        # Interpolate
        # latents shape: (B, N*N)
        new_latent = alpha * latents[idx1] + (1 - alpha) * latents[idx2]
        syn_latents.append(new_latent)
        syn_labels.append(minority_class)
        
    if len(syn_latents) > 0:
        syn_latents = torch.stack(syn_latents)
        syn_labels = torch.tensor(syn_labels, device=latents.device)
        
        return torch.cat([latents, syn_latents]), torch.cat([labels, syn_labels])
    
    return latents, labels

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading data...")
    dataset = RsFmriDataset(DATA_ROOT, sequence_length=SEQ_LEN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # num_workers=0 is crucial for Windows to avoid spawning issues in some envs
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0) 
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = BDHNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_traj = nn.MSELoss()
    
    print("Starting training loop...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            adj = batch['adj'].to(device) # (B, N, N)
            spikes = batch['spikes'].to(device) # (B, T, N)
            icn_raw = batch['icn_raw'].to(device) # (B, T, N) target
            mask = batch['mask'].to(device) # (B, T)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            w_final, y_seq = model.bdh_layer(spikes, adj, mask)
            
            # Trajectory Loss
            x_next_pred = model.traj_head(y_seq) # (B, T, N)
            pred_slice = x_next_pred[:, :-1, :]
            target_slice = icn_raw[:, 1:, :] 
            loss_traj = criterion_traj(pred_slice, target_slice)
            
            # Classification with SMOTE
            w_flat = w_final.view(w_final.size(0), -1)
            
            if epoch > 0: # Start SMOTE earlier for testing
                w_smote, labels_smote = batch_smote(w_flat, labels)
            else:
                w_smote, labels_smote = w_flat, labels
                
            logits = model.classifier(w_smote)
            loss_cls = criterion_cls(logits, labels_smote)
            
            # Sparse Regularization (L1 on w_final)
            loss_sparse = torch.mean(torch.abs(w_final))
            
            # Total Loss
            loss = loss_cls + LAMBDA_TRAJ * loss_traj + LAMBDA_SPARSE * loss_sparse
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Acc
            orig_logits = logits[:labels.size(0)] # Only original batch
            _, predicted = torch.max(orig_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item():.4f}")
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {total_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%")
        
        # Validation
        val_acc = validate(model, val_loader, device)
        print(f"Val Acc: {val_acc:.2f}%")
        
    # Save Model
    torch.save(model.state_dict(), "bdh_model.pth")
    print("Model saved to bdh_model.pth")


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            adj = batch['adj'].to(device)
            spikes = batch['spikes'].to(device)
            labels = batch['label'].to(device)
            mask = batch['mask'].to(device)
            
            logits, _, _ = model(spikes, adj, mask)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

if __name__ == "__main__":
    train()
