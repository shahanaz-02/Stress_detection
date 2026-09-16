"""
SHAP Explainability Module
Project: AI-Based Physiological Stress Monitoring System (WESAD LOSO Pipeline)

Outputs:
- results/shap_summary.png
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import MODELS_DIR, RESULTS_DIR, ensure_directories
from data_loader import load_wesad_features
from preprocessing import prepare_features_and_target

def generate_shap_visualizations():
    ensure_directories()
    
    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run src/train.py first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    df = load_wesad_features()
    X, y, _ = prepare_features_and_target(df)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    print("\n[INFO] Computing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_scaled)
    
    if isinstance(shap_vals, list):
        target_shap = shap_vals[1]
    else:
        target_shap = shap_vals
        
    # Save results/shap_summary.png
    plt.figure(figsize=(10, 6))
    shap.summary_plot(target_shap, X_scaled, show=False)
    plt.title("SHAP Global Feature Importance (WESAD LOSO Pipeline)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    shap_summary_path = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(shap_summary_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved SHAP summary plot to: {shap_summary_path}")

if __name__ == "__main__":
    generate_shap_visualizations()
