
import torch
import numpy as np
from src.data_loader.loader import RsFmriDataset
from src.data_loader.spike_encoder import SpikeEncoder

def check_determinism():
    # Use a dummy data root - we just need to instantiate the class or manually check encoder
    # Just check the encoder directly since that's where the flaw is
    
    encoder = SpikeEncoder(method='rate')
    
    # Create dummy input: 10 time steps, 5 channels
    dummy_input = np.random.rand(10, 5)
    
    # Run 1 with seed
    seed = 42
    spikes1 = encoder(dummy_input, seed=seed)
    
    # Run 2 with same seed
    spikes2 = encoder(dummy_input, seed=seed)
    
    print("Run 1 Spikes (first 5):")
    print(spikes1[:5, 0])
    
    print("Run 2 Spikes (first 5):")
    print(spikes2[:5, 0])
    
    if torch.equal(spikes1, spikes2):
        print("PASS: Output is deterministic with seed.")
    else:
        print("FAIL: Output is STILL non-deterministic.")

if __name__ == "__main__":
    check_determinism()
