"""
Evaluation & Visualization Module (LOSO CV Metrics, Confusion Matrix, ROC Curve)
Project: AI-Based Physiological Stress Monitoring System (WESAD LOSO Pipeline)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import RESULTS_DIR, ensure_directories

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculates evaluation metrics dictionary."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    
    return {
        'Accuracy': round(float(acc), 4),
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4),
        'F1-Score': round(float(f1), 4),
        'ROC-AUC': round(float(auc), 4)
    }

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    """Plots and saves confusion matrix heatmap."""
    ensure_directories()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Stressed', 'Stressed'],
                yticklabels=['Non-Stressed', 'Stressed'])
    plt.title(f"LOSO Confusion Matrix ({model_name})", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved LOSO confusion matrix figure to: {save_path}")

def plot_roc_curve(y_true, y_prob, model_name: str, save_path: str = None):
    """Plots and saves ROC curve figure."""
    ensure_directories()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "roc_curve.png")
        
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='#0f766e', lw=2.5, label=f'{model_name} (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Leave-One-Subject-Out ROC Curve ({model_name})', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved LOSO ROC curve figure to: {save_path}")
