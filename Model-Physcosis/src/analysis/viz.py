import numpy as np
import networkx as nx
import plotly.graph_objects as go
import torch

def visualize_brain_graph(adj_matrix, threshold=0.0, title="Brain Connectivity"):
    """
    Visualizes the brain graph in 3D using Plotly.
    args:
        adj_matrix: (N, N) numpy array (symmetric).
        threshold: Absolute threshold for edges to show.
    """
    N = adj_matrix.shape[0]
    
    # Create graph
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
        
    # Add edges
    # We only add strong edges to avoid clutter
    # If threshold is 0.0, we might want to use percentile
    if threshold == 0.0:
        threshold = np.percentile(np.abs(adj_matrix), 95) # Top 5%
        print(f"Auto-thresholding at {threshold:.4f}")
        
    rows, cols = np.where(np.triu(np.abs(adj_matrix), 1) > threshold)
    
    for r, c in zip(rows, cols):
        weight = adj_matrix[r, c]
        G.add_edge(r, c, weight=weight)
        
    # Layout
    # Use spring layout in 3D
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Extract coordinates
    node_x = []
    node_y = []
    node_z = []
    for i in range(N):
        x, y, z = pos[i]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
    # Nodes trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.8
        ),
        text=[f"ICN {i}" for i in range(N)],
        hoverinfo='text'
    )
    
    # Edges trace
    # Plotly doesn't support varying color lines in a single trace well for 3D? 
    # We can use lines with None separate
    edge_x = []
    edge_y = []
    edge_z = []
    colors = []
    
    for u, v, d in G.edges(data=True):
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        
        # Color based on weight sign?
        # Ideally we want a color scale.
        # But single trace = single color scale. 
        # We'll map weight to a dummy coord or just simple color.
        
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(
            color='gray',
            width=1
        ),
        opacity=0.3
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        )
    )
    
    return fig

if __name__ == "__main__":
    # Mock data
    adj = np.random.randn(105, 105)
    adj = (adj + adj.T) / 2
    fig = visualize_brain_graph(adj, threshold=1.5)
    fig.write_html("brain_viz_demo.html")
    print("Saved brain_viz_demo.html")
