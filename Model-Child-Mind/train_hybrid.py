import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader.child_mind import ChildMindDataset
from src.model.bdh_features import BDHFeatureExtractor
from src.model.sae import train_sae
from src.model.imputation import SynapticKNNImputer
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib  # Added for saving boosters

import random
import os

# Config
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS_BDH = 2 # Reduced for speed
EPOCHS_SAE = 5 # Reduced for speed
SEQ_LEN = 230
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_bdh_extractor(dataset, feature_extractor, device):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = optim.Adam(feature_extractor.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Head
    head = nn.Linear(64*64 + 64, 1).to(device)
    optimizer_head = optim.Adam(head.parameters(), lr=0.001)
    
    feature_extractor.train()
    feature_extractor.to(device)
    
    print("Training BDH Feature Extractor...")
    for epoch in range(EPOCHS_BDH):
        total_loss = 0
        for batch in dataloader:
            ts = batch['ts'].to(device)
            mask = batch['mask'].to(device)
            age_group = batch['age_group'].to(device)
            target = batch['target'].to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            optimizer_head.zero_grad()
            
            features = feature_extractor(ts, mask, age_group) 
            pred = head(features)
            
            loss = criterion(pred, target)
            loss.backward()
            
            optimizer.step()
            optimizer_head.step()
            
            total_loss += loss.item()
            
        print(f"BDH Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        
    return feature_extractor, head

def extract_features(dataset, feature_extractor, device):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    feature_extractor.eval()
    feature_extractor.to(device)
    
    all_features = []
    all_tabular = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            ts = batch['ts'].to(device)
            mask = batch['mask'].to(device)
            age_group = batch['age_group'].to(device)
            # Tabular might be float32 but contains NaNs. 
            tabular = batch['tabular'].cpu().numpy()
            target = batch['target'].cpu().numpy()
            
            feats = feature_extractor(ts, mask, age_group).cpu().numpy()
            
            all_features.append(feats)
            all_tabular.append(tabular)
            all_targets.append(target)
            
    return np.concatenate(all_features, axis=0), np.concatenate(all_tabular, axis=0), np.concatenate(all_targets, axis=0)

def train_boosters(X, y):
    print("Training Boosters on Hybrid Features...")
    
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1, random_state=SEED)
    lgb_model.fit(X, y)
    print("LGBM Trained.")
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=SEED)
    xgb_model.fit(X, y)
    print("XGB Trained.")
    
    cat_model = CatBoostRegressor(iterations=100, learning_rate=0.05, verbose=0, random_seed=SEED)
    cat_model.fit(X, y)
    print("CatBoost Trained.")
    
    # Save booster models (Added for frontend integration)
    joblib.dump(lgb_model, 'lgbm_pciat.pkl')
    joblib.dump(xgb_model, 'xgb_pciat.pkl')
    joblib.dump(cat_model, 'catboost_pciat.pkl')
    print("Saved booster models: lgbm_pciat.pkl, xgb_pciat.pkl, catboost_pciat.pkl")
    
    return [lgb_model, xgb_model, cat_model]

def main():
    seed_everything(SEED)
    dataset = ChildMindDataset(split='train', sequence_length=SEQ_LEN)
    
    # 1. BDH Feature Extractor
    feature_extractor = BDHFeatureExtractor(input_channels=5, virtual_nodes=64)
    feature_extractor, _ = train_bdh_extractor(dataset, feature_extractor, DEVICE)
    
    # 2. Extract
    print("Extracting features (Pre-imputation)...")
    bdh_feats, tab_feats, targets = extract_features(dataset, feature_extractor, DEVICE)
    
    print(f"BDH Features: {bdh_feats.shape}, Tabular (With NaNs): {tab_feats.shape}")
    
    # 3. SAE
    print("Training Sparse Autoencoder...")
    sae_model = train_sae(bdh_feats, input_dim=bdh_feats.shape[1], latent_dim=32, epochs=EPOCHS_SAE, device=DEVICE)
    
    # Get Latent Features
    with torch.no_grad():
        sae_model.eval()
        _, latent_feats = sae_model(torch.tensor(bdh_feats, dtype=torch.float32).to(DEVICE))
        bdh_reduced = latent_feats.cpu().numpy()
        
    print(f"Latent Synaptic Features: {bdh_reduced.shape}")
    
    # 4. Synaptic Imputation
    print("Imputing Tabular Data using Synaptic Neighbors...")
    knn_imputer = SynapticKNNImputer(k=5)
    knn_imputer.fit(bdh_reduced, tab_feats)
    tab_imputed = knn_imputer.transform(bdh_reduced, tab_feats)
    
    # Fallback Imputation & Scaling
    print("Finalizing Preprocessing...")
    simple_imputer = SimpleImputer(strategy='mean')
    tab_imputed = simple_imputer.fit_transform(tab_imputed)
    
    scaler = StandardScaler()
    tab_imputed = scaler.fit_transform(tab_imputed)
    
    # Concatenate
    X = np.concatenate([tab_imputed, bdh_reduced], axis=1)
    y = targets
    
    # Split Train/Test
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # 5. Train Boosters
    models = train_boosters(X_train, y_train)
    
    # 6. Evaluate
    preds = []
    for model in models:
        preds.append(model.predict(X_test))
        
    avg_pred = np.mean(preds, axis=0)
    mse = mean_squared_error(y_test, avg_pred)
    # QWK approximation
    from sklearn.metrics import cohen_kappa_score
    # simple thresholding for QWK (assuming 0-4 range based on sii)
    # sii: 0=None, 1=Mild, 2=Moderate, 3=Severe? 
    # PCIAT Total is 0-80+. We predicted PCIAT.
    # Convert PCIAT to sii buckets roughly: <30:0, 30-50:1, 50-80:2, >80:3
    # This is rough validation only.
    
    print(f"Ensemble MSE (Test): {mse:.4f}")
    
    # QWK Evaluation
    def to_sii(score):
        if score < 31: return 0
        if score < 50: return 1
        if score < 80: return 2
        return 3
        
    y_true_sii = [to_sii(s) for s in y_test]
    y_pred_sii = [to_sii(s) for s in avg_pred]
    
    qwk = cohen_kappa_score(y_true_sii, y_pred_sii, weights='quadratic')
    print(f"Ensemble QWK (Test): {qwk:.4f}")
    
    # Save Feature Extractor & SAE
    torch.save(feature_extractor.state_dict(), "bdh_child_mind.pth")
    torch.save(sae_model.state_dict(), "sae_child_mind.pth")
    print("Saved: bdh_child_mind.pth, sae_child_mind.pth")

if __name__ == "__main__":
    main()
