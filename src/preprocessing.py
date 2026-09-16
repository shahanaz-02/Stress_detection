"""
Data Preprocessing & Feature Engineering Module
Project: AI-Based Physiological Stress Monitoring System
Stage: 5 & 6 - Preprocessing & Feature Engineering
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

def engineer_physiological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers physiological feature interactions supported by stress literature:
    1. HRV Ratio (RMSSD / SDNN): Low ratio indicates sympathetic dominance (stress).
    2. EDA Activation Index (mean_EDA * SCR_peaks): Measures electrodermal arousal.
    3. HR Variability Index (std_HR / mean_HR): Normalized heart rate fluctuation.
    """
    df = df.copy()
    
    # 1. HRV Ratio (small epsilon to prevent divide-by-zero)
    df['HRV_ratio'] = df['RMSSD'] / (df['SDNN'] + 1e-5)
    
    # 2. EDA Activation Index
    df['EDA_activation'] = df['mean_EDA'] * (df['SCR_peaks'] + 1.0)
    
    # 3. Normalized HR Fluctuation
    df['HR_CV'] = df['std_HR'] / (df['mean_HR'] + 1e-5)
    
    return df

def prepare_subject_aware_split(df: pd.DataFrame, target_col: str = 'stress_binary', test_size: float = 0.2, random_state: int = 42):
    """
    Performs Subject-Aware Train/Test Splitting using GroupShuffleSplit.
    Ensures data from the same participant does NOT appear in both training
    and testing sets, preventing data leakage and ensuring true generalization.
    """
    df_engineered = engineer_physiological_features(df)
    
    # Define feature set X (exclude non-predictive metadata & targets)
    ignore_cols = ['subject_id', 'label', 'stress_binary']
    feature_cols = [col for col in df_engineered.columns if col not in ignore_cols]
    
    X = df_engineered[feature_cols]
    y = df_engineered[target_col]
    groups = df_engineered['subject_id']
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    train_subjects = set(groups.iloc[train_idx])
    test_subjects = set(groups.iloc[test_idx])
    
    print(f"[INFO] Train Subjects ({len(train_subjects)}): {sorted(list(train_subjects))}")
    print(f"[INFO] Test Subjects ({len(test_subjects)}): {sorted(list(test_subjects))}")
    print(f"[INFO] Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")
    
    # Scale features using StandardScaler fitted ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_cols, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_cols, index=X_test_raw.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

if __name__ == "__main__":
    from data_loader import load_and_inspect_dataset
    import os
    
    csv_path = os.path.join("data", "processed", "physiological_stress_data.csv")
    df = load_and_inspect_dataset(csv_path)
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_subject_aware_split(df)
    print("\n[SUCCESS] Preprocessing completed without data leakage.")
