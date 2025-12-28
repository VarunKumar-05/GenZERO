import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class ClinicalDashboard:
    def __init__(self):
        print("Initialized Clinical Output & Interpretability")

    def generate_report(self, analysis_results):
        print("Generating clinical report...")
        anomalies = analysis_results.get("anomalies", [])
        probes = analysis_results.get("concept_probes", {})
        
        report = "## Clinical Assessment Report\n"
        if anomalies:
            report += f"**Alerts:** {', '.join(anomalies)}\n\n"
        else:
            report += "**Status:** Patient Stable\n\n"
            
        report += "### Neuromorphic Concept Activation:\n"
        for concept, data in probes.items():
            status = "Active" if data["active"] else "Inactive"
            report += f"- **{concept.capitalize()}**: {status} (Activation: {data['activation_level']:.2f}, Synaptic Strength: {data['synaptic_strength']:.2f})\n"
            
        return report

    def plot_sparsity(self, hidden_state):
        """
        Plots the sparsity of the neural activations.
        """
        if hasattr(hidden_state, 'detach'):
            activations = hidden_state.detach().numpy().flatten()
        else:
            activations = np.array(hidden_state).flatten()
            
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(activations)), activations, color='teal')
        ax.set_title("Neural Activation Sparsity (BDH)")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Firing Rate")
        ax.axhline(y=0.1, color='r', linestyle='--', label='Sparsity Threshold')
        ax.legend()
        return fig

    def plot_topology(self, weights):
        """
        Plots the force-directed graph of the synaptic weights.
        """
        if hasattr(weights, 'detach'):
            w = weights.detach().numpy()
        else:
            w = np.array(weights)
            
        # Create graph from weight matrix
        G = nx.Graph()
        rows, cols = w.shape
        for i in range(rows):
            for j in range(cols):
                if abs(w[i, j]) > 0.5: # Threshold for visualization
                    G.add_edge(i, j, weight=w[i, j])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.6, ax=ax)
        
        # Draw edges with varying thickness
        edges = G.edges(data=True)
        weights_list = [abs(d['weight']) for u, v, d in edges]
        nx.draw_networkx_edges(G, pos, width=weights_list, alpha=0.4, edge_color='gray', ax=ax)
        
        ax.set_title("Emergent Synaptic Topology")
        ax.axis('off')
        return fig

    def visualize(self, audio_data, snn_output):
        # Legacy method
        pass
        ax1.set_title("Input: Micro-prosodic Audio Stream")
        # Simulate a waveform
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(100)
        ax1.plot(t, signal, color='blue', alpha=0.7)
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        # 2. SNN Spike Raster Plot (BDH Architecture)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("BDH SNN: Spike Raster Plot")
        # Simulate spikes for 20 neurons over 50 timesteps
        spikes = np.random.rand(20, 50) > 0.9
        rows, cols = np.where(spikes)
        ax2.scatter(cols, rows, s=10, c='purple', marker='|')
        ax2.set_ylabel("Neuron ID")
        ax2.set_xlabel("Time Step")
        ax2.set_xlim(0, 50)

        # 3. Feature Space (Clusters)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title("Semantic Mapping: Symptom Clusters")
        # Simulate clusters
        c1 = np.random.normal(loc=[2, 2], scale=0.5, size=(20, 2))
        c2 = np.random.normal(loc=[-2, -1], scale=0.5, size=(20, 2))
        ax3.scatter(c1[:, 0], c1[:, 1], c='green', label='Healthy')
        ax3.scatter(c2[:, 0], c2[:, 1], c='red', label='Dysarthria (Detected)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Risk Gauge (Simple Bar)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title("Clinical Risk Score")
        risk_score = 0.75
        ax4.bar(["Risk"], [risk_score], color='orange', width=0.3)
        ax4.set_ylim(0, 1)
        ax4.text(0, risk_score + 0.05, f"{risk_score*100}%", ha='center')

        plt.tight_layout()
        plt.show()
