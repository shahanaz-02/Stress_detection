"""
Model Training & Evaluation Module
Project: AI-Based Physiological Stress Monitoring System
Stage: 7, 8, 9, 10 & 12 - Baseline, Random Forest, XGBoost & Model Serialization
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

from data_loader import load_and_inspect_dataset
from preprocessing import prepare_subject_aware_split

def train_and_evaluate_models(data_path: str, models_dir: str = "models", results_dir: str = "results"):
    """
    Trains Baseline (Decision Tree), Random Forest, and XGBoost models.
    Evaluates metrics, saves results, and persists the best performing model.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load and Preprocess Data
    df = load_and_inspect_dataset(data_path)
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_subject_aware_split(df)
    
    # Define models dictionary
    models = {
        "Baseline (Decision Tree)": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42)
    }
    
    metrics_summary = []
    trained_models = {}
    confusion_matrices = {}
    
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING & SUBJECT-AWARE EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n---> Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        trained_models[name] = model
        confusion_matrices[name] = cm
        
        metrics_summary.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Non-Stressed", "Stressed"]))

    # Convert results to DataFrame
    comparison_df = pd.DataFrame(metrics_summary)
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Save comparison metrics CSV
    comparison_csv_path = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\n[INFO] Saved model comparison table to: {comparison_csv_path}")

    # Determine best model based on F1-Score
    best_model_name = comparison_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n[WINNER] Best Performing Model: {best_model_name}")

    # Save individual models for multi-model web app switching
    joblib.dump(trained_models["Baseline (Decision Tree)"], os.path.join(models_dir, "decision_tree.pkl"))
    joblib.dump(trained_models["Random Forest"], os.path.join(models_dir, "random_forest.pkl"))
    joblib.dump(trained_models["XGBoost"], os.path.join(models_dir, "xgboost.pkl"))

    # Save best model, scaler, and feature list via Joblib
    best_model_path = os.path.join(models_dir, "best_stress_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    meta_path = os.path.join(models_dir, "model_metadata.pkl")
    
    joblib.dump(best_model, best_model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        "model_name": best_model_name,
        "feature_cols": feature_cols,
        "train_shape": X_train.shape,
        "metrics": comparison_df.to_dict(orient="records")
    }, meta_path)
    
    print(f"[INFO] Saved individual model files: decision_tree.pkl, random_forest.pkl, xgboost.pkl")
    print(f"[INFO] Best model saved to: {best_model_path}")
    print(f"[INFO] Scaler saved to: {scaler_path}")

    # Generate & Save Evaluation Visualizations
    plot_model_comparison(comparison_df, results_dir)
    plot_confusion_matrices(confusion_matrices, results_dir)
    plot_feature_importances(best_model, feature_cols, best_model_name, results_dir)
    
    return comparison_df, best_model, feature_cols, X_train, X_test, y_train, y_test

def plot_model_comparison(df: pd.DataFrame, results_dir: str):
    """Plots metric comparison bar chart across models."""
    plt.figure(figsize=(10, 6))
    df_melted = df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                        var_name="Metric", value_name="Score")
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Physiological Stress Model Performance Comparison", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "model_comparison_chart.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved comparison chart to: {plot_path}")

def plot_confusion_matrices(cm_dict: dict, results_dir: str):
    """Plots confusion matrix heatmaps for each model."""
    fig, axes = plt.subplots(1, len(cm_dict), figsize=(5 * len(cm_dict), 4))
    if len(cm_dict) == 1:
        axes = [axes]
        
    for ax, (name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Non-Stressed", "Stressed"],
                    yticklabels=["Non-Stressed", "Stressed"])
        ax.set_title(f"{name}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrices heatmap to: {cm_path}")

def plot_feature_importances(model, feature_cols, model_name, results_dir):
    """Plots Gini/Split feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_cols)[indices], palette="magma")
    plt.title(f"Feature Importance ({model_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Importance")
    plt.ylabel("Physiological Feature")
    plt.tight_layout()
    fi_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved feature importance plot to: {fi_path}")

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "physiological_stress_data.csv")
    train_and_evaluate_models(data_path)
