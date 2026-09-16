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

from feature_extraction import engineer_features_df

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
        eng_df = engineer_features_df(raw_df)
        
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
        
        # Generate Human-Readable Natural Language XAI Cause Explanation
        clinical_narrative = self._generate_clinical_explanation(
            stress_label, confidence, contributions_df, X_raw.iloc[0].to_dict()
        )
        
        return {
            "prediction_class": pred_class,
            "prediction_label": stress_label,
            "confidence_percentage": round(confidence, 2),
            "stress_probability": round(float(pred_probs[1] * 100.0), 2),
            "model_used": self.model_name,
            "top_contributing_features": contributions_df.to_dict(orient="records"),
            "clinical_explanation": clinical_narrative
        }

    def _generate_clinical_explanation(self, label: str, confidence: float, contrib_df: pd.DataFrame, raw_vals: dict) -> dict:
        """
        Translates SHAP mathematical attributions into plain English physiological explanations.
        Identifies the exact input values that triggered high stress or relaxed states.
        """
        # Separate features elevating risk vs lowering risk (any positive/negative SHAP value)
        positive_drivers = contrib_df[contrib_df['shap_impact'] > 0.0].sort_values(by='shap_impact', ascending=False)
        negative_drivers = contrib_df[contrib_df['shap_impact'] < 0.0].sort_values(by='shap_impact', ascending=True)
        
        feature_labels = {
            'mean_HR': ('Mean Heart Rate', 'bpm'),
            'std_HR': ('Heart Rate Fluctuation', 'bpm'),
            'RMSSD': ('Parasympathetic HRV (RMSSD)', 'ms'),
            'SDNN': ('Total HRV Variability (SDNN)', 'ms'),
            'mean_EDA': ('Skin Conductance (GSR/EDA)', 'µS'),
            'std_EDA': ('EDA Variation', 'µS'),
            'SCR_peaks': ('Skin Conductance Response Bursts', 'peaks'),
            'mean_Temp': ('Skin Temperature', '°C'),
            'std_Temp': ('Skin Temp Fluctuation', '°C'),
            'mean_RESP': ('Respiration Rate', 'rpm'),
            'HRV_ratio': ('Autonomic Balance Ratio (HRV Ratio)', 'index'),
            'EDA_activation': ('Electrodermal Activation Index', 'index'),
            'HR_CV': ('Normalized HR Coefficient of Variation', 'index')
        }

        feature_descriptions = {
            'EDA_activation': "High skin conductance combined with frequent SCR bursts indicated sympathetic nervous system fight-or-flight arousal.",
            'mean_EDA': "Elevated electrodermal activity level indicated increased sweat gland activation driven by stress.",
            'SCR_peaks': "Frequent electrodermal response bursts indicated active sympathetic stress triggers.",
            'RMSSD': "Reduced Heart Rate Variability (RMSSD) indicated parasympathetic withdrawal and high autonomic stress.",
            'HRV_ratio': "A low HRV ratio indicated sympathetic dominance over parasympathetic rest.",
            'mean_Temp': "Low skin temperature indicated stress-induced peripheral vasoconstriction (cold skin response).",
            'mean_HR': "Elevated heart rate indicated cardiac sympathetic stimulation.",
            'mean_RESP': "Elevated respiration rate indicated rapid stress breathing (tachypnea).",
            'std_HR': "Increased heart rate fluctuation indicated unstable cardiac rhythm under stress.",
            'std_Temp': "Temperature fluctuations indicated unstable peripheral blood flow during stress."
        }

        causes = []
        for _, row in positive_drivers.iterrows():
            f = row['feature']
            val = row['value']
            shap_score = row['shap_impact']
            fname, unit = feature_labels.get(f, (f, ''))
            desc = feature_descriptions.get(f, "Elevated risk contribution towards stress.")
            
            causes.append({
                'feature_key': f,
                'feature_name': fname,
                'value': f"{val} {unit}".strip(),
                'shap_impact': round(float(shap_score), 4),
                'explanation': f"**{fname} ({val} {unit})**: {desc} (SHAP impact: +{shap_score:.3f})"
            })
            
        protective = []
        for _, row in negative_drivers.iterrows():
            f = row['feature']
            val = row['value']
            shap_score = abs(row['shap_impact'])
            fname, unit = feature_labels.get(f, (f, ''))
            desc = "Helped lower overall stress risk score."
            if f == 'RMSSD':
                desc = "Healthy resting HRV parasympathetic activity."
            elif f == 'mean_RESP':
                desc = "Calm and steady respiration rate."
            elif f == 'mean_EDA':
                desc = "Baseline non-aroused electrodermal activity."
            elif f == 'HRV_ratio':
                desc = "Favorable autonomic balance supporting recovery."
                
            protective.append({
                'feature_key': f,
                'feature_name': fname,
                'value': f"{val} {unit}".strip(),
                'shap_impact': round(float(shap_score), 4),
                'explanation': f"**{fname} ({val} {unit})**: {desc} (Reduced stress risk by -{shap_score:.3f})"
            })

        if label == "STRESSED":
            if causes:
                top_causes_str = ", ".join([f"**{c['feature_name']}** ({c['value']})" for c in causes[:3]])
                summary = f"The AI model classified this state as **STRESSED** with **{confidence:.1f}% confidence**. High stress was primarily driven by: {top_causes_str}."
            else:
                summary = f"The AI model classified this state as **STRESSED** with **{confidence:.1f}% confidence** based on physiological marker shifts."
        else:
            if protective:
                top_prot_str = ", ".join([f"**{p['feature_name']}** ({p['value']})" for p in protective[:3]])
                summary = f"The AI model classified this state as **NON-STRESSED / RELAXED** with **{confidence:.1f}% confidence**. Calm state maintained by: {top_prot_str}."
            else:
                summary = f"The AI model classified this state as **NON-STRESSED / RELAXED** with **{confidence:.1f}% confidence**."

        return {
            "summary_sentence": summary,
            "primary_stress_causes": causes,
            "protective_relaxed_factors": protective
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
