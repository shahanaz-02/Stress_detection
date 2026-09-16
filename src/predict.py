"""
Re-Usable Stress Prediction & Inference Engine
Project: AI-Based Physiological Stress Monitoring System
Stage: 12 - Stress Prediction Function & Inference Engine
"""

import os
import joblib
import pandas as pd
import numpy as np
import sys
import shap
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocessing import engineer_physiological_features

class StressPredictionEngine:
    def __init__(self, models_dir: str = "models", model_file: str = "best_stress_model.pkl"):
        self.best_model_path = os.path.join(models_dir, model_file)
        self.scaler_path = os.path.join(models_dir, "scaler.pkl")
        self.meta_path = os.path.join(models_dir, "model_metadata.pkl")
        
        if not os.path.exists(self.best_model_path):
            # Fallback to best_stress_model.pkl if requested file not found
            self.best_model_path = os.path.join(models_dir, "best_stress_model.pkl")
            
        self.model = joblib.load(self.best_model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.meta = joblib.load(self.meta_path)
        self.feature_cols = self.meta["feature_cols"]
        self.model_name = type(self.model).__name__
        
        # Initialize SHAP explainer once for fast inference
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, input_data: dict) -> dict:
        """
        Takes raw physiological inputs dict, performs feature engineering & scaling,
        and returns prediction label, probability confidence score, and SHAP drivers.
        
        Expected raw inputs in input_data:
        - mean_HR: Heart rate (bpm)
        - std_HR: Heart rate std dev
        - RMSSD: Root mean square of RR diffs (ms)
        - SDNN: Std dev of RR intervals (ms)
        - mean_EDA: Electrodermal activity level (µS)
        - std_EDA: EDA std dev
        - SCR_peaks: Skin conductance peak count
        - mean_Temp: Skin temperature (°C)
        - std_Temp: Temp std dev
        - mean_RESP: Respiration rate
        """
        # Create single row DataFrame
        raw_df = pd.DataFrame([input_data])
        
        # Apply feature engineering
        eng_df = engineer_physiological_features(raw_df)
        
        # Select required features in identical column order
        X_raw = eng_df[self.feature_cols]
        
        # Scale features
        X_scaled = pd.DataFrame(self.scaler.transform(X_raw), columns=self.feature_cols)
        
        # Predict class and confidence
        pred_class = int(self.model.predict(X_scaled)[0])
        pred_probs = self.model.predict_proba(X_scaled)[0]
        confidence = float(pred_probs[pred_class] * 100.0)
        
        stress_label = "STRESSED" if pred_class == 1 else "NON-STRESSED"
        
        # Compute SHAP explanation for instance
        shap_vals = self.explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            sample_shap = shap_vals[1]
        else:
            sample_shap = shap_vals
            
        sample_shap = np.array(sample_shap).flatten()
            
        contributions = []
        for col, raw_val, shap_val in zip(self.feature_cols, X_raw.values[0], sample_shap):
            shap_val_float = float(shap_val)
            impact = "Elevates Stress Risk" if shap_val_float > 0 else "Lowers Stress Risk"
            contributions.append({
                "feature": col,
                "value": round(float(raw_val), 2),
                "shap_impact": round(shap_val_float, 4),
                "effect": impact
            })
            
        # Sort by absolute impact magnitude
        contributions_df = pd.DataFrame(contributions).sort_values(by="shap_impact", key=abs, ascending=False)
        
        return {
            "prediction_class": pred_class,
            "prediction_label": stress_label,
            "confidence_percentage": round(confidence, 2),
            "stress_probability": round(float(pred_probs[1] * 100.0), 2),
            "model_used": self.model_name,
            "top_contributing_features": contributions_df.to_dict(orient="records")
        }

if __name__ == "__main__":
    engine = StressPredictionEngine()
    
    # Test sample 1: Normal / Low Stress
    sample_low = {
        'mean_HR': 68.5, 'std_HR': 3.2, 'RMSSD': 48.0, 'SDNN': 56.0,
        'mean_EDA': 0.8, 'std_EDA': 0.05, 'SCR_peaks': 1,
        'mean_Temp': 33.2, 'std_Temp': 0.04, 'mean_RESP': 15.5
    }
    
    # Test sample 2: High Stress
    sample_high = {
        'mean_HR': 92.0, 'std_HR': 7.5, 'RMSSD': 18.0, 'SDNN': 24.0,
        'mean_EDA': 5.2, 'std_EDA': 0.35, 'SCR_peaks': 7,
        'mean_Temp': 31.8, 'std_Temp': 0.12, 'mean_RESP': 24.0
    }
    
    print("\n--- Low Stress Inference Test ---")
    res_low = engine.predict(sample_low)
    print(f"Prediction: {res_low['prediction_label']} | Confidence: {res_low['confidence_percentage']}%")
    
    print("\n--- High Stress Inference Test ---")
    res_high = engine.predict(sample_high)
    print(f"Prediction: {res_high['prediction_label']} | Confidence: {res_high['confidence_percentage']}%")
