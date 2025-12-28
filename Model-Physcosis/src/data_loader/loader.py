import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .spike_encoder import SpikeEncoder

class RsFmriDataset(Dataset):
    def __init__(self, data_root, transform=None, sequence_length=230):
        """
        Args:
            data_root (str): Path to the train directory containing 'SZ' and 'BP' folders.
            transform (callable, optional): Optional transform to be applied on a sample.
            sequence_length (int): Length of the time series to load.
        """
        self.data_root = data_root
        self.transform = transform
        self.sequence_length = sequence_length
        self.samples = []
        self.labels = []
        
        self._load_data()
        self.spike_encoder = SpikeEncoder(method='rate')

    def _load_data(self):
        # SZ = 1, BP = 0
        classes = {'BP': 0, 'SZ': 1}
        
        for class_name, label in classes.items():
            class_path = os.path.join(self.data_root, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Path {class_path} does not exist.")
                continue
                
            subjects = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
            
            for sub in subjects:
                sub_path = os.path.join(class_path, sub)
                fnc_path = os.path.join(sub_path, 'fnc.npy')
                icn_path = os.path.join(sub_path, 'icn_tc.npy')
                
                if os.path.exists(fnc_path) and os.path.exists(icn_path):
                    self.samples.append({
                        'fnc_path': fnc_path,
                        'icn_path': icn_path,
                        'subject_id': sub
                    })
                    self.labels.append(label)
        
        print(f"Loaded {len(self.samples)} samples from {self.data_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        label = self.labels[idx]
        
        # Load data
        fnc = np.load(sample_info['fnc_path']).squeeze() # (5460,)
        icn_tc = np.load(sample_info['icn_path']) # (Time, 105)
        
        # Preprocessing
        # 1. Reconstruct Adjacency Matrix from FNC (Upper Triangle)
        # FNC is usually the upper triangle of the correlation matrix.
        # 105 regions -> (105*104)/2 = 5460 elements. PERFECT.
        
        adj = np.zeros((105, 105))
        upper_tri_indices = np.triu_indices(105, k=1)
        adj[upper_tri_indices] = fnc
        adj = adj + adj.T # Make symmetric
        np.fill_diagonal(adj, 1.0) # Self-correlation is 1
        
        # 2. Encode Timecourses to Spikes
        # Normalize ICN first
        icn_tc = (icn_tc - np.mean(icn_tc)) / (np.std(icn_tc) + 1e-6)
        
        # Convert to spikes
        spikes = self.spike_encoder(icn_tc) # (Time, 105)
        
        # 3. Pad or Truncate
        T = icn_tc.shape[0]
        if T < self.sequence_length:
            # Pad
            padding_len = self.sequence_length - T
            # Pad spikes with 0 (False)
            spikes_padded = torch.cat([spikes, torch.zeros((padding_len, 105), dtype=torch.float32)], dim=0)
            # Pad icn_tc with 0
            icn_padded = torch.cat([torch.tensor(icn_tc, dtype=torch.float32), torch.zeros((padding_len, 105), dtype=torch.float32)], dim=0)
            # Mask: 1 for real, 0 for padded
            mask = torch.cat([torch.ones(T, dtype=torch.bool), torch.zeros(padding_len, dtype=torch.bool)], dim=0)
        else:
            # Truncate
            spikes_padded = spikes[:self.sequence_length]
            icn_padded = torch.tensor(icn_tc[:self.sequence_length], dtype=torch.float32)
            mask = torch.ones(self.sequence_length, dtype=torch.bool)
            
        return {
            'adj': torch.tensor(adj, dtype=torch.float32),
            'spikes': spikes_padded,
            'icn_raw': icn_padded,
            'mask': mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # Test loading
    dataset = RsFmriDataset(r'd:\Hackathons  & Competitions\Synaptix\Model-Physcosis\psychosis-classification-with-rsfmri\data\train')
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Adj shape:", sample['adj'].shape)
    print("Spikes shape:", sample['spikes'].shape)
    print("Label:", sample['label'])
