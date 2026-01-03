"""
Train PCIAT prediction boosters on synthetic data.
This creates realistic-looking models without needing the actual Child Mind dataset.
"""
import numpy as np
import joblib
import os

# Try to import boosters
try:
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    BOOSTERS_AVAILABLE = True
except ImportError:
    BOOSTERS_AVAILABLE = False
    print("Warning: Booster packages not installed")

def generate_synthetic_latent_data(n_samples=500, latent_dim=32):
    """Generate synthetic SAE latent features that mimic real patterns."""
    np.random.seed(42)  # Reproducibility
    
    # Generate latent features with some structure
    latent = np.random.randn(n_samples, latent_dim) * 0.5
    
    # Add correlations between features to simulate learned representations
    for i in range(0, latent_dim - 1, 2):
        latent[:, i+1] = latent[:, i] * 0.7 + np.random.randn(n_samples) * 0.3
    
    # Generate PCIAT scores based on latent features
    # Create a nonlinear relationship
    weights = np.random.randn(latent_dim) * 2
    base_score = np.dot(latent, weights)
    
    # Normalize to reasonable PCIAT range (0-100)
    base_score = (base_score - base_score.min()) / (base_score.max() - base_score.min())
    pciat_scores = 10 + base_score * 70  # Range: 10-80
    
    # Add some noise
    pciat_scores += np.random.randn(n_samples) * 5
    pciat_scores = np.clip(pciat_scores, 0, 100)
    
    return latent, pciat_scores

def train_and_save_boosters(save_dir="."):
    """Train booster models on synthetic data and save them."""
    if not BOOSTERS_AVAILABLE:
        print("Cannot train boosters - packages not installed")
        return False
    
    print("Generating synthetic training data...")
    X, y = generate_synthetic_latent_data(n_samples=1000, latent_dim=32)
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"PCIAT score range: {y.min():.1f} - {y.max():.1f}")
    
    # Train LightGBM
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
    lgb_model.fit(X, y)
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, verbosity=0)
    xgb_model.fit(X, y)
    
    # Train CatBoost
    print("Training CatBoost...")
    cat_model = CatBoostRegressor(iterations=100, learning_rate=0.05, verbose=0)
    cat_model.fit(X, y)
    
    # Save models
    lgb_path = os.path.join(save_dir, 'lgbm_pciat.pkl')
    xgb_path = os.path.join(save_dir, 'xgb_pciat.pkl')
    cat_path = os.path.join(save_dir, 'catboost_pciat.pkl')
    
    joblib.dump(lgb_model, lgb_path)
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(cat_model, cat_path)
    
    print(f"\n✅ Saved: {lgb_path}")
    print(f"✅ Saved: {xgb_path}")
    print(f"✅ Saved: {cat_path}")
    
    # Verify models work
    test_input = np.random.randn(1, 32)
    lgb_pred = lgb_model.predict(test_input)[0]
    xgb_pred = xgb_model.predict(test_input)[0]
    cat_pred = cat_model.predict(test_input)[0]
    avg_pred = (lgb_pred + xgb_pred + cat_pred) / 3
    
    print(f"\n🧪 Test prediction: LGBM={lgb_pred:.1f}, XGB={xgb_pred:.1f}, CatBoost={cat_pred:.1f}")
    print(f"📊 Ensemble average: {avg_pred:.1f}")
    
    return True

if __name__ == "__main__":
    # Train and save in Model-Child-Mind directory
    success = train_and_save_boosters()
    if success:
        print("\n🎉 Booster training complete!")
    else:
        print("\n❌ Training failed")
