"""
Model Training & Leave-One-Subject-Out (LOSO) Cross-Validation Module
Project: AI-Based Physiological Stress Monitoring System (WESAD LOSO Pipeline)

Key Rules Implemented:
1. Leave-One-Subject-Out (LOSO) Cross-Validation across all 15 WESAD subjects.
2. Strict Fold Preprocessing: Scaler is fit ONLY on training subjects inside each fold.
3. Evaluates Baseline (Decision Tree), Random Forest, and XGBoost models.
4. Saves final models to models/random_forest.pkl & models/xgboost.pkl.
5. Saves metrics to results/metrics.csv.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import ensure_directories, MODELS_DIR, RESULTS_DIR
from data_loader import load_wesad_features
from preprocessing import prepare_features_and_target, fit_scale_fold, FEATURE_COLS
from evaluate import calculate_metrics, plot_confusion_matrix, plot_roc_curve

def run_loso_cross_validation():
    """
    Executes 15-fold Leave-One-Subject-Out Cross-Validation across all 15 WESAD subjects.
    Guarantees strict fold-level scaler fitting and zero subject data leakage.
    """
    ensure_directories()
    
    # 1. Load Data
    df = load_wesad_features()
    X, y, groups = prepare_features_and_target(df)
    
    unique_subjects = sorted(groups.unique().tolist())
    n_subjects = len(unique_subjects)
    print("\n" + "="*65)
    print(f"STARTING LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print(f"Total Subjects ({n_subjects}): {unique_subjects}")
    print("="*65)
    
    logo = LeaveOneGroupOut()
    
    # Define model instances
    models = {
        "Baseline (Decision Tree)": lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": lambda: XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42)
    }
    
    model_predictions = {name: [] for name in models}
    model_probabilities = {name: [] for name in models}
    y_true_all = []
    
    fold_reports = []
    
    # Iterate over each LOSO Fold (1 to 15)
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        test_sub = groups.iloc[test_idx].iloc[0]
        train_subs = sorted(groups.iloc[train_idx].unique().tolist())
        
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # STRICT FOLD PREPROCESSING: Fit scaler ONLY on train subjects
        X_train_scaled, X_test_scaled, fold_scaler = fit_scale_fold(X_train_raw, X_test_raw)
        
        if fold == 1:
            y_true_all.extend(y_test.values)
        elif len(models) > 0 and fold > 1 and len(y_true_all) < len(y):
            y_true_all.extend(y_test.values)
            
        print(f"\n---> Fold {fold}/{n_subjects} | Test Subject: {test_sub} (Train: {len(train_subs)} subjects)")
        
        fold_row = {'Fold': fold, 'Test_Subject': test_sub}
        
        for name, model_fn in models.items():
            model = model_fn()
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            model_predictions[name].extend(y_pred)
            model_probabilities[name].extend(y_prob)
            
            fold_acc = float(np.mean(y_pred == y_test.values))
            fold_f1 = float(calculate_metrics(y_test, y_pred, y_prob)['F1-Score'])
            
            fold_row[f"{name}_Acc"] = round(fold_acc, 4)
            fold_row[f"{name}_F1"] = round(fold_f1, 4)
            
        fold_reports.append(fold_row)

    # 2. Aggregate Overall Subject-Independent LOSO Metrics
    y_true_array = np.array(y_true_all)
    overall_metrics = []
    
    print("\n" + "="*65)
    print("AGGREGATED LEAVE-ONE-SUBJECT-OUT (LOSO) SYSTEM PERFORMANCE")
    print("="*65)
    
    for name in models:
        preds = np.array(model_predictions[name])
        probs = np.array(model_probabilities[name])
        
        metrics = calculate_metrics(y_true_array, preds, probs)
        metrics['Model'] = name
        overall_metrics.append(metrics)
        
        print(f"\n[{name}] LOSO Performance:")
        print(f"   - Accuracy:  {metrics['Accuracy']*100:.2f}%")
        print(f"   - Precision: {metrics['Precision']*100:.2f}%")
        print(f"   - Recall:    {metrics['Recall']*100:.2f}%")
        print(f"   - F1-Score:  {metrics['F1-Score']*100:.2f}%")
        print(f"   - ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        
    metrics_df = pd.DataFrame(overall_metrics)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    
    # Save overall LOSO metrics CSV
    metrics_csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[SUCCESS] Saved LOSO evaluation report to: {metrics_csv_path}")

    # Plot Confusion Matrix and ROC Curve for top ensemble (Random Forest / XGBoost)
    best_name = metrics_df.sort_values(by='F1-Score', ascending=False).iloc[0]['Model']
    plot_confusion_matrix(y_true_array, np.array(model_predictions[best_name]), best_name)
    plot_roc_curve(y_true_array, np.array(model_probabilities[best_name]), best_name)

    # 3. Fit Final Models on Complete Dataset for Production Inference
    print("\n" + "="*65)
    print("TRAINING FINAL PRODUCTION MODELS & SERIALIZING (.PKL)")
    print("="*65)
    
    scaler_final = fit_scale_fold(X, X)[2] # Scaler on full dataset for serving
    X_scaled_full = pd.DataFrame(scaler_final.transform(X), columns=FEATURE_COLS)
    
    rf_final = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_final.fit(X_scaled_full, y)
    
    xgb_final = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42)
    xgb_final.fit(X_scaled_full, y)
    
    dt_final = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_final.fit(X_scaled_full, y)
    
    joblib.dump(rf_final, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(xgb_final, os.path.join(MODELS_DIR, "xgboost.pkl"))
    joblib.dump(dt_final, os.path.join(MODELS_DIR, "decision_tree.pkl"))
    joblib.dump(rf_final, os.path.join(MODELS_DIR, "best_stress_model.pkl"))
    joblib.dump(scaler_final, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump({
        "model_name": best_name,
        "feature_cols": FEATURE_COLS,
        "loso_metrics": metrics_df.to_dict(orient="records")
    }, os.path.join(MODELS_DIR, "model_metadata.pkl"))
    
    print(f"[SUCCESS] Saved final models to models/random_forest.pkl & models/xgboost.pkl")
    return metrics_df

if __name__ == "__main__":
    run_loso_cross_validation()
