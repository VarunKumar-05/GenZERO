import torch
import numpy as np
import sys
import os

# Ensure src is in path (Parent directory)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from src.bdh_snn.network import SpikingNeuralNetwork
from src.analysis.detector import AnalysisLayer

def test_pipeline():
    print("=== Starting Model-Refactor Verification ===")
    
    # 1. Setup
    input_dim = 128
    hidden_dim = 256
    model = SpikingNeuralNetwork(input_dim, hidden_dim)
    analyzer = AnalysisLayer()
    
    # 2. Calibration Phase (First 200 frames)
    print("\n[Phase 1] Calibration (feeding random noise)...")
    # Generate random data: N(0, 1)
    # The new LayerNorm should handle this fine.
    baseline_data = torch.randn(210, input_dim) 
    
    anomalies_found = 0
    
    for i in range(210):
        x = baseline_data[i].unsqueeze(0) # (1, 128)
        out = model.forward(x)
        hidden = out["hidden_state"]
        
        # Determine last weights (dummy update)
        last_weights = model.update_synapses(x, hidden)
        
        analysis = analyzer.analyze(out, last_weights)
        
        if analysis.get("anomalies"):
            print(f"Frame {i}: Unexpected Anomaly! {analysis['anomalies']}")
            anomalies_found += 1
            
        if i == 199 or i == 200:
             # Check if calibration msg printed in console (by eye)
             pass
             
    if anomalies_found == 0:
        print("PASS: No anomalies detected during calibration.")
    else:
        print(f"FAIL: {anomalies_found} anomalies during/just after calibration.")

    # 3. Injection Phase (Force Tremor)
    # Tremor = Neurons [0, 1, 2] High
    # Since we can't easily force specific neurons in the hidden layer without backprop,
    # We will verify the *Detector Logic* directly by passing a hacked hidden state.
    
    print("\n[Phase 2] Testing Sensitivity (Injecting 'Tremor' Hidden State)...")
    
    # Create a hidden state where [0,1,2] are 5 sigma above mean (which is ~0 ish)
    # Since ReLU activations are positive, mean is > 0.
    # We'll just put a huge value.
    
    fake_hidden = torch.zeros(1, hidden_dim)
    # From previous runs we saw hidden norms ~20-30.
    # Let's put 100.0 into the target indices.
    fake_hidden[0, 0] = 100.0
    fake_hidden[0, 1] = 100.0
    fake_hidden[0, 2] = 100.0
    
    fake_out = {"output": torch.zeros(1, 10), "hidden_state": fake_hidden}
    
    analysis = analyzer.analyze(fake_out, last_weights)
    print(f"Injection Result: {analysis['anomalies']}")
    
    if "Tremor Detected" in analysis.get("anomalies", []):
        print("PASS: Tremor correctly detected.")
    else:
        print("FAIL: Tremor missed despite massive injection.")

    # 4. Injection Phase (Force Stutter)
    # Stutter = Neurons [3, 4] High
    fake_hidden = torch.zeros(1, hidden_dim)
    fake_hidden[0, 3] = 100.0
    fake_hidden[0, 4] = 100.0
    
    fake_out = {"output": torch.zeros(1, 10), "hidden_state": fake_hidden}
    analysis = analyzer.analyze(fake_out, last_weights)
    print(f"Injection Result: {analysis['anomalies']}")
    
    if "Speech Disfluency Detected" in analysis.get("anomalies", []):
        print("PASS: Stutter correctly detected.")
    else:
        print("FAIL: Stutter missed.")
        
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    test_pipeline()
