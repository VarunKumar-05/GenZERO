import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from src.model.bdh_features import BDHFeatureExtractor
from src.data_loader.child_mind import ChildMindDataset

def visualize_synaptic_graph(feature_extractor, dataset, device, sample_idx=0, threshold=0.1):
    feature_extractor.eval()
    feature_extractor.to(device)
    
    # Get a sample
    sample = dataset[sample_idx]
    ts = sample['ts'].unsqueeze(0).to(device) # (1, T, 5)
    mask = sample['mask'].unsqueeze(0).to(device)
    # Handle age group if model expects it
    if 'age_bucket' in sample: # check key name in dataset
         age_group = sample['age_bucket'].unsqueeze(0).to(device)
    elif 'age_group' in sample:
         age_group = sample['age_group'].unsqueeze(0).to(device)
    else:
         age_group = None

    # Run forward pass partially to get w_final
    # We need to access internal BDH layer or modify forward to return w
    # BDHFeatureExtractor.forward returns concatenated features.
    # Let's peek into the model:
    with torch.no_grad():
        x_virt = feature_extractor.input_proj(ts)
        x_spikes = torch.sigmoid(x_virt)
        
        if age_group is not None:
             w_init = feature_extractor.w_init_bank[age_group]
        else:
             # Default
             w_init = feature_extractor.w_init_bank.mean(dim=0).unsqueeze(0).expand(1, -1, -1)
             
        w_final, _ = feature_extractor.bdh(x_spikes, w_init, mask)
        
    w_matrix = w_final.squeeze().cpu().numpy() # (64, 64)
    
    # Graph Construction
    G = nx.Graph()
    num_nodes = w_matrix.shape[0]
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)
        
    # Add edges
    # w is usually non-symmetric in Hebbian, but for viz we can average
    adj = (w_matrix + w_matrix.T) / 2
    # Threshold
    max_w = np.max(np.abs(adj))
    if max_w > 0:
        adj = adj / max_w
        
    count = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            weight = adj[i, j]
            if abs(weight) > threshold:
                G.add_edge(i, j, weight=weight)
                count += 1
                
    print(f"Graph has {num_nodes} nodes and {count} edges (threshold={threshold})")
    
    # Layout (3D Spring)
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Nodes
    xs = [pos[i][0] for i in G.nodes()]
    ys = [pos[i][1] for i in G.nodes()]
    zs = [pos[i][2] for i in G.nodes()]
    ax.scatter(xs, ys, zs, c='cyan', s=20, alpha=0.8)
    
    # Edges
    for u, v, d in G.edges(data=True):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        w = d['weight']
        alpha = min(1.0, abs(w) + 0.1)
        color = 'lime' if w > 0 else 'red'
        ax.plot(x, y, z, c=color, alpha=alpha, linewidth=0.5)
        
    ax.set_title(f"Synaptic Memory Graph (Sample {sample_idx})", color='white')
    ax.axis('off')
    
    plt.savefig("child_mind_viz.png", dpi=150, bbox_inches='tight', facecolor='black')
    print("Saved visualization to child_mind_viz.png")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init Model
    model = BDHFeatureExtractor(input_channels=5, virtual_nodes=64, num_age_groups=3)
    # Load weights
    try:
        model.load_state_dict(torch.load("bdh_child_mind.pth", map_location=device))
        print("Loaded trained metrics.")
    except:
        print("Warning: Could not load trained weights. Using random initialization.")
    
    # Dataset
    dataset = ChildMindDataset(split='train', sequence_length=230)
    
    visualize_synaptic_graph(model, dataset, device, sample_idx=10, threshold=0.15)
