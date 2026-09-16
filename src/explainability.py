"""
Explainable AI (XAI) Module using SHAP
Project: AI-Based Physiological Stress Monitoring System
Stage: 11 - SHAP Explainable AI
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from data_loader import load_and_inspect_dataset
from preprocessing import prepare_subject_aware_split

def generate_shap_explanations(models_dir: str = "models", results_dir: str = "results", data_path: str = None):
    """
    Computes SHAP values using TreeExplainer on the best trained model.
    Saves global summary plot and bar plot to results directory.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    best_model_path = os.path.join(models_dir, "best_stress_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    meta_path = os.path.join(models_dir, "model_metadata.pkl")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Trained model not found at {best_model_path}. Run src/train.py first.")
        
    model = joblib.load(best_model_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)
    
    feature_cols = meta["feature_cols"]
    model_name = meta["model_name"]
    
    if data_path is None:
        data_path = os.path.join("data", "processed", "physiological_stress_data.csv")
        
    df = load_and_inspect_dataset(data_path)
    _, X_test, _, _, _, _ = prepare_subject_aware_split(df)
    
    print(f"\n[INFO] Computing SHAP values for model: {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle single output vs binary output list format in shap
    if isinstance(shap_values, list):
        shap_vals_target = shap_values[1] # Class 1 (Stressed)
    else:
        shap_vals_target = shap_values

    # 1. SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_target, X_test, show=False)
    plt.title(f"SHAP Global Feature Attribution ({model_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    shap_beeswarm_path = os.path.join(results_dir, "shap_summary_beeswarm.png")
    plt.savefig(shap_beeswarm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved SHAP summary plot to: {shap_beeswarm_path}")

    # 2. SHAP Bar Importance Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_target, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Mean |Value| Feature Importance ({model_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    shap_bar_path = os.path.join(results_dir, "shap_feature_bar.png")
    plt.savefig(shap_bar_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved SHAP bar plot to: {shap_bar_path}")

    return explainer, shap_vals_target, X_test

def explain_single_sample(model, explainer, sample_row_df: pd.DataFrame):
    """
    Computes local SHAP explanation for an individual user prediction.
    Returns sorted list of feature contributions.
    """
    shap_vals = explainer.shap_values(sample_row_df)
    
    if isinstance(shap_vals, list):
        sample_shap = shap_vals[1][0]
    elif len(shap_vals.shape) == 2:
        sample_shap = shap_vals[0]
    else:
        sample_shap = shap_vals
        
    explanation = []
    for col, val, shap_val in zip(sample_row_df.columns, sample_row_df.values[0], sample_shap):
        direction = "Increases Stress Risk" if shap_val > 0 else "Decreases Stress Risk"
        explanation.append({
            "feature": col,
            "value": round(float(val), 3),
            "shap_value": round(float(shap_val), 4),
            "impact": direction
        })
        
    explanation_df = pd.DataFrame(explanation).sort_values(by="shap_value", key=abs, ascending=False)
    return explanation_df

if __name__ == "__main__":
    generate_shap_explanations()
