
import torch
import numpy as np
import os
import torch.nn.functional as F
from src.model.bdh import BDHNet
from src.data_loader.spike_encoder import SpikeEncoder

# Config
SEQ_LEN = 230
MODEL_PATH = "bdh_model.pth"
SUB_ID = "sub777"
DATA_DIR = r"D:\Hackathons  & Competitions\Synaptix\Model-Physcosis\psychosis-classification-with-rsfmri\data\train\BP\sub777"

def load_and_preprocess(data_dir, sub_id):
    print(f"Loading data for {sub_id} from {data_dir}...")
    fnc_path = os.path.join(data_dir, "fnc.npy")
    icn_path = os.path.join(data_dir, "icn_tc.npy")
    
    if not os.path.exists(fnc_path) or not os.path.exists(icn_path):
        raise FileNotFoundError("Data files not found")

    # Load raw
    fnc = np.load(fnc_path).squeeze()
    icn_tc = np.load(icn_path)

    # 1. Reconstruct Adjacency
    adj = np.zeros((105, 105))
    upper_tri_indices = np.triu_indices(105, k=1)
    adj[upper_tri_indices] = fnc
    adj = adj + adj.T
    np.fill_diagonal(adj, 1.0)
    
    # 2. Encode Spikes
    icn_tc = (icn_tc - np.mean(icn_tc)) / (np.std(icn_tc) + 1e-6)
    encoder = SpikeEncoder(method='rate')
    
    # Deterministic Seed based on subject ID
    seed = int(hash(sub_id) % 10**8)
    spikes = encoder(icn_tc, seed=seed)
    
    # 3. Pad/Truncate
    T = icn_tc.shape[0]
    if T < SEQ_LEN:
        padding_len = SEQ_LEN - T
        spikes_padded = torch.cat([spikes, torch.zeros((padding_len, 105), dtype=torch.float32)], dim=0)
        mask = torch.cat([torch.ones(T, dtype=torch.bool), torch.zeros(padding_len, dtype=torch.bool)], dim=0)
    else:
        spikes_padded = spikes[:SEQ_LEN]
        mask = torch.ones(SEQ_LEN, dtype=torch.bool)
        
    # Batch dimension
    return (
        torch.tensor(adj, dtype=torch.float32).unsqueeze(0),
        spikes_padded.unsqueeze(0),
        mask.unsqueeze(0)
    )

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = BDHNet(num_nodes=105, num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.")
    
    # Process Data
    adj, spikes, mask = load_and_preprocess(DATA_DIR, SUB_ID)
    adj = adj.to(device)
    spikes = spikes.to(device)
    mask = mask.to(device)
    
    # Inference
    with torch.no_grad():
        logits, _, _ = model(spikes, adj, mask)
        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        
    classes = {0: 'BP (Bipolar)', 1: 'SZ (Schizophrenia)'}
    print("-" * 30)
    print(f"Prediction for {SUB_ID} (True Label: BP):")
    print(f"Logits: {logits.cpu().numpy()}")
    print(f"Probabilities: BP={probs[0][0]:.4f}, SZ={probs[0][1]:.4f}")
    print(f"Predicted Class: {classes[pred_label]}")
    print("-" * 30)

    if pred_label == 0:
        print("RESULT: CORRECT")
    else:
        print("RESULT: INCORRECT")

if __name__ == "__main__":
    predict()
