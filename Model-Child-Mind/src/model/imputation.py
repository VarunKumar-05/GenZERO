import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class SynapticKNNImputer:
    def __init__(self, k=5):
        self.k = k
        self.synaptic_features = None
        self.tabular_data = None
        
    def fit(self, synaptic_features, tabular_data):
        """
        synaptic_features: (N, D) latent vectors from SAE
        tabular_data: (N, M) tabular features (potentially with NaNs)
        """
        self.synaptic_features = synaptic_features
        # Store df to keep column names if possible, else numpy
        self.tabular_data = tabular_data if isinstance(tabular_data, pd.DataFrame) else pd.DataFrame(tabular_data)
        
    def transform(self, synaptic_features, tabular_data):
        """
        Impute missing values in tabular_data using nearest neighbors in synaptic_features
        """
        # If we are transforming the training set itself, we can use stored data?
        # But generally, for each sample i in tabular_data:
        # 1. if has NaNs:
        # 2. Find k nearest neighbors in self.synaptic_features
        # 3. Fill NaNs with weighted mean of neighbors
        
        X_tab = tabular_data.copy()
        if isinstance(X_tab, np.ndarray):
            X_tab = pd.DataFrame(X_tab)
            
        # Identify rows with NaNs
        # Actually, sklearn Imputer style
        # But we need to use 'synaptic_features' to find distances, not 'tabular_data'
        
        # Compute Cosine Similarity between input synaptic_features and stored self.synaptic_features
        # (N_query, N_train)
        sim_matrix = cosine_similarity(synaptic_features, self.synaptic_features)
        
        # For each sample
        imputed_data = X_tab.values
        
        for i in range(len(X_tab)):
            row = X_tab.iloc[i]
            if row.isnull().any():
                # Find indices of k nearest neighbors
                # Argsort returns ascending, so take last k
                # Exclude self if in training set? (sim[i,i] == 1.0)
                # But here we might be strictly imputing train set.
                
                # Get similarities for this row
                sims = sim_matrix[i]
                
                # Sort indices descending
                sorted_indices = np.argsort(sims)[::-1]
                
                # Take top k (skipping self if distance is 0/sim is 1 and it's exact match?)
                # Simple approach: just take top k+1 and ignore self if index matches
                
                neighbors_indices = sorted_indices[:self.k+1] # Take k+1 just in case
                
                # Get neighbor tabular data
                neighbor_tab = self.tabular_data.iloc[neighbors_indices]
                
                # Weighted average?
                # weights = sims[neighbors_indices]
                # Filter out neighbors that also have NaN for the specific column?
                
                # For each missing col
                missing_cols = row.index[row.isnull()]
                for col in missing_cols:
                    # Get values of neighbors for this col
                    vals = neighbor_tab[col].values
                    # Weights
                    w = sims[neighbors_indices]
                    
                    # Filter valid
                    valid_mask = ~pd.isna(vals)
                    if valid_mask.sum() > 0:
                        vals_valid = vals[valid_mask]
                        w_valid = w[valid_mask]
                        # Normalize weights
                        if w_valid.sum() > 0:
                            w_valid /= w_valid.sum()
                            imputed_val = np.dot(vals_valid, w_valid)
                        else:
                            imputed_val = np.nanmean(vals_valid)
                            
                        imputed_data[i, X_tab.columns.get_loc(col)] = imputed_val
                    else:
                        # Fallback to column mean?
                        pass 
                        
        return imputed_data
