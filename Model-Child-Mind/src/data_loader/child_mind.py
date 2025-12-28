import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
import joblib

DATA_DIR = r'd:\Hackathons  & Competitions\Synaptix\Model-Child-Mind\data'

class ChildMindDataset(Dataset):
    def __init__(self, split='train', sequence_length=230):
        self.split = split
        self.sequence_length = sequence_length
        self.data_dir = DATA_DIR
        
        self._load_and_preprocess_tabular()
        self._prepare_time_series_paths()
        
    def _load_and_preprocess_tabular(self):
        # Load Tabular
        df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        
        # 1. Clean Implausible Values
        # Body Fat > 60% -> NaN (assuming column names 'Physical-BMI', 'Physical-Body_Fat_%' ???)
        # Need to check columns in EDA, but user said "implausible values".
        # Let's inspect columns first in real usage. For now, assume generic cleaning.
        
        # 2. Imputation (Simplified for now: Mean, but Plan calls for Lasso)
        # We will select numeric columns
        target_cols = ['PCIAT-PCIAT_Total', 'sii', 'id']
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in target_cols]
        
        # 3. Create Target
        if self.split == 'train':
            self.targets = df['PCIAT-PCIAT_Total'].values
            self.sii = df['sii'].values
            # Filter rows with NaN target?
            valid_mask = ~np.isnan(self.targets)
            df = df[valid_mask]
            self.targets = self.targets[valid_mask]
            self.sii = self.sii[valid_mask]
            
        self.ids = df['id'].values
        # Store Ages for Gating
        if 'Basic_Demos-Age' in df.columns:
            self.ages = df['Basic_Demos-Age'].values
        else:
            # Fallback if column missing (shouldn't happen based on EDA)
            self.ages = np.full(len(df), np.nan)
        
        # Feature Engineering: Age Normalization
        # "Normalizing PU and CU based on the mean of the age groups"
        # Columns likely 'Physical-Heart_Rate_PU', 'Physical-Heart_Rate_CU'? Or similar.
        # Let's try to identify PU/CU columns if they exist.
        
        # Normalize numeric columns
        # self.scaler = StandardScaler()
        # Keep NaNs for Synaptic Imputation downstream
        self.tabular_features = df[numeric_cols].values # .fillna(0) 
        # self.tabular_features = self.scaler.fit_transform(self.tabular_features)
        
    def _prepare_time_series_paths(self):
        self.ts_paths = {}
        # parquet files are in series_train.parquet/id=XXXX/part-0.parquet
        pq_root = os.path.join(self.data_dir, 'series_train.parquet')
        
        # Map ID to path
        # glob all subdirs
        subdirs = glob.glob(os.path.join(pq_root, 'id=*'))
        for d in subdirs:
            # id=00115b9f -> 00115b9f
            sid = os.path.basename(d).split('=')[1]
            files = glob.glob(os.path.join(d, '*.parquet'))
            if files:
                self.ts_paths[sid] = files[0]
                
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        sid = self.ids[idx]
        
        # Tabular
        tab_feat = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        
        # Time Series
        ts_path = self.ts_paths.get(sid)
        if ts_path:
            # Load parquet
            ts_df = pd.read_parquet(ts_path)
            # Columns: step, X, Y, Z, enmo, anglez, ...
            # Select channels
            channels = ['X', 'Y', 'Z', 'enmo', 'anglez']
            ts_data = ts_df[channels].values
            
            # Resample/Truncate
            # 43k steps is too much. Step=5s -> 12 samples/min. 230 samples -> ~20 mins?
            # User wants "Deep Analysis", BDH needs sequences. 
            # Let's take a strided view or random crop?
            # Or just first N?
            # Let's stride to fit sequence_length
            L = ts_data.shape[0]
            if L > self.sequence_length:
                # Stride to cover full duration?
                stride = L // self.sequence_length
                ts_data = ts_data[::stride][:self.sequence_length]
            else:
                # Pad
                pad = np.zeros((self.sequence_length - L, 5))
                ts_data = np.concatenate([ts_data, pad], axis=0)
                
            ts_tensor = torch.tensor(ts_data, dtype=torch.float32)
            mask = torch.ones(self.sequence_length, dtype=torch.float32)
        else:
            # Missing time series
            ts_tensor = torch.zeros((self.sequence_length, 5), dtype=torch.float32)
            mask = torch.zeros(self.sequence_length, dtype=torch.float32)
            
        target = torch.tensor(self.targets[idx], dtype=torch.float32) if self.split == 'train' else torch.tensor(0.0)
        
        # Age Bucket
        # Extract age from raw df? But we have self.tabular_features which is just values.
        # We need to store Age separately in __init__
        # Re-using logic: we need access to 'Basic_Demos-Age'
        # Since I can't easily access it from self.tabular_features without knowing index, 
        # I should assume it's stored or I should have stored it.
        # Quick fix: Add self.ages in _load_and_preprocess_tabular
        age = self.ages[idx]
        if np.isnan(age):
            age_bucket = 1 # Default to middle
        elif age < 10:
            age_bucket = 0
        elif age < 16:
            age_bucket = 1
        else:
            age_bucket = 2
            
        return {
            'id': sid,
            'tabular': tab_feat,
            'ts': ts_tensor,
            'mask': mask,
            'target': target,
            'age_group': torch.tensor(age_bucket, dtype=torch.long)
        }
