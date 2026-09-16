"""
Preprocessing & Scaler Fitting Module
Project: AI-Based Physiological Stress Monitoring System (WESAD LOSO Pipeline)

STRICT RULE:
- Scaler must be fit ONLY on the training fold subjects.
- Test subject data must ONLY be transformed using the fitted training scaler.
- Prevents information leakage from the test subject into training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'mean_HR', 'std_HR', 'RMSSD', 'SDNN', 
    'mean_EDA', 'std_EDA', 'SCR_peaks', 
    'mean_Temp', 'std_Temp', 'mean_RESP',
    'HRV_ratio', 'EDA_activation', 'HR_CV'
]

def prepare_features_and_target(df: pd.DataFrame, target_col: str = 'stress_binary'):
    """Extracts feature matrix X, target y, and subject_id groups."""
    X = df[FEATURE_COLS].copy()
    y = df[target_col].copy()
    groups = df['subject_id'].copy()
    return X, y, groups

def fit_scale_fold(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame):
    """
    Fits StandardScaler exclusively on X_train_raw (training subjects),
    then transforms both X_train_raw and X_test_raw.
    Guarantees NO data leakage from test subject.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw), 
        columns=X_train_raw.columns, 
        index=X_train_raw.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw), 
        columns=X_test_raw.columns, 
        index=X_test_raw.index
    )
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    print("[INFO] Preprocessing module ready with feature columns:", FEATURE_COLS)
