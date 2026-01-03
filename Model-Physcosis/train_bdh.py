import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from src.data_loader.loader import RsFmriDataset
from src.model.bdh import BDHNet
import numpy as np

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Config
BATCH_SIZE = 16
EPOCHS = 5 # Reduced for quick verification, user can increase later or I can keep 20? User said "rerun model", implies full run. Keeping 20 might be slow.
# User asked to "rerun", usually implies full retraining. I will keep 20 or reduce slightly if I want speed? 
# The user didn't say "test run", they said "rerun". I'll keep 20.
EPOCHS = 50
LR = 0.001
LAMBDA_SPARSE = 1e-3 # Increased from 1e-4
LAMBDA_TRAJ = 0.5
PATIENCE = 5 # Early stopping patience
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
    set_seed(42) # Set global seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading data...")
    dataset = RsFmriDataset(DATA_ROOT, sequence_length=SEQ_LEN)
    # Manual Stratified Split
    # Get labels from dataset
    labels = np.array(dataset.labels)
    class_indices = [np.where(labels == i)[0] for i in range(2)] # [Indices for 0, Indices for 1]
    
    train_indices = []
    val_indices = []
    
    print(f"Class distribution: BP(0)={len(class_indices[0])}, SZ(1)={len(class_indices[1])}")
    
    for indices in class_indices:
        np.random.shuffle(indices) # Seeded by set_seed(42) above
        split = int(0.8 * len(indices))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])
        
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
    
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    
    # Weighted Random Sampler for Balanced Batches
    train_targets = [dataset.labels[i] for i in train_indices]
    class_sample_counts = np.bincount(train_targets)
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    # num_workers=0 is crucial for Windows to avoid spawning issues in some envs
    # shuffle must be False when using sampler
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True, num_workers=0) 
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    
    # Model
    model = BDHNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3) # Added L2 Regularization
    
    # Aggressive Bias Fix: Sampler + Weighted Loss
    # Even with balanced batches, we penalize BP errors more because they are harder to learn
    class_weights = torch.tensor([1.5, 0.8]).to(device) # Heavy weight on BP(0)
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    
    criterion_traj = nn.MSELoss()
    
    print("Starting training loop...")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            adj = batch['adj'].to(device) # (B, N, N)
            
            # Input Noise Augmentation
            # Add Gaussian noise to the initial weights (adj)
            noise = torch.randn_like(adj) * 0.05
            adj = adj + noise
            
            spikes = batch['spikes'].to(device) # (B, T, N)
            icn_raw = batch['icn_raw'].to(device) # (B, T, N) target
            mask = batch['mask'].to(device) # (B, T)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            w_final, y_seq = model.bdh_layer(spikes, adj, mask)
            
            # Trajectory Loss (Mean over heads)
            y_mean = y_seq.mean(dim=2) # (B, T, N)
            x_next_pred = model.traj_head(y_mean) # (B, T, N)
            pred_slice = x_next_pred[:, :-1, :]
            target_slice = icn_raw[:, 1:, :] 
            loss_traj = criterion_traj(pred_slice, target_slice)
            
            # Feature Fusion (Structure + Dynamics)
            w_flat = w_final.view(w_final.size(0), -1)
            y_seq_flat = y_seq.view(y_seq.size(0), y_seq.size(1), -1)
            y_pooled = model.pooling(y_seq_flat, mask)
            fused = torch.cat([w_flat, y_pooled], dim=1)
            
            # Classification with SMOTE on Fused Features
            if epoch > 0: # Start SMOTE earlier for testing
                fused_smote, labels_smote = batch_smote(fused, labels)
            else:
                fused_smote, labels_smote = fused, labels
                
            logits = model.classifier(fused_smote)
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
        
        # Early Stopping & Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "bdh_model.pth")
            print("New best model saved!")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
            
    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    # Per-class counters
    class_correct = [0, 0]
    class_total = [0, 0]
    classes = ['BP', 'SZ']
    
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
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    for i in range(2):
        if class_total[i] > 0:
            print(f"Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
            
    # Return Macro-Average Accuracy to force balance
    # (Acc_BP + Acc_SZ) / 2
    if class_total[0] > 0 and class_total[1] > 0:
        macro_acc = 100 * ((class_correct[0] / class_total[0]) + (class_correct[1] / class_total[1])) / 2
    else:
        macro_acc = 100 * correct / total
        
    print(f"Macro Acc: {macro_acc:.2f}%")
    return macro_acc

if __name__ == "__main__":
    train()
