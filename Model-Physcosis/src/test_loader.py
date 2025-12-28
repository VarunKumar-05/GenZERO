import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader.loader import RsFmriDataset

def test_loader():
    data_root = r'd:\Hackathons  & Competitions\Synaptix\Model-Physcosis\psychosis-classification-with-rsfmri\data\train'
    dataset = RsFmriDataset(data_root)
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Adj shape:", sample['adj'].shape)
        print("Spikes shape:", sample['spikes'].shape)
        print("Label:", sample['label'])
    else:
        print("Dataset is empty!")

if __name__ == "__main__":
    test_loader()
