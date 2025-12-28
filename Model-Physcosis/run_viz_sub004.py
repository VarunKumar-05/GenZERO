import numpy as np
from src.analysis.viz import visualize_brain_graph

# Load real FNC
fnc_path = r'd:\Hackathons  & Competitions\Synaptix\Model-Physcosis\psychosis-classification-with-rsfmri\data\train\SZ\sub004\fnc.npy'
fnc = np.load(fnc_path).squeeze()

# Reconstruct
adj = np.zeros((105, 105))
upper_tri_indices = np.triu_indices(105, k=1)
adj[upper_tri_indices] = fnc
adj = adj + adj.T
np.fill_diagonal(adj, 1.0)

print(f"Adj range: {adj.min()} to {adj.max()}")

# Visualize
fig = visualize_brain_graph(adj, threshold=0.5, title="Subject 004 (SZ) - FNC Connectivity")
fig.write_html("sub004_fnc.html")
print("Saved sub004_fnc.html")
