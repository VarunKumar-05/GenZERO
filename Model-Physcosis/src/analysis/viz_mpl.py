import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_brain_graph_mpl(adj_matrix, threshold=0.0, title="Brain Connectivity"):
    N = adj_matrix.shape[0]
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
        
    if threshold == 0.0:
        threshold = np.percentile(np.abs(adj_matrix), 95)
        
    rows, cols = np.where(np.triu(np.abs(adj_matrix), 1) > threshold)
    for r, c in zip(rows, cols):
        weight = adj_matrix[r, c]
        G.add_edge(r, c, weight=weight)
        
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    
    # Nodes
    xs, ys, zs = [], [], []
    for i in range(N):
        xs.append(pos[i][0])
        ys.append(pos[i][1])
        zs.append(pos[i][2])
    ax.scatter(xs, ys, zs, c='b', marker='o', s=20, alpha=0.6)
    
    # Edges
    for u, v, d in G.edges(data=True):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        ax.plot(x, y, z, c='gray', alpha=0.3, linewidth=0.5)
        
    plt.tight_layout()
    plt.savefig('brain_viz.png')
    print("Saved brain_viz.png")

if __name__ == "__main__":
    adj = np.random.randn(105, 105)
    adj = (adj + adj.T) / 2
    visualize_brain_graph_mpl(adj, threshold=1.5)
