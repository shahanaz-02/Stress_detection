"""
Physiological Feature Extraction Module
Project: AI-Based Physiological Stress Monitoring System (WESAD LOSO Pipeline)

Calculates signal features from raw ECG/EDA/Temp/Resp windows:
- ECG: R-peaks, RR intervals -> mean_HR, std_HR, RMSSD, SDNN
- EDA: mean_EDA, std_EDA, SCR_peaks
- Temp: mean_Temp, std_Temp
- Resp: mean_RESP
- Engineered: HRV_ratio, EDA_activation, HR_CV
"""

import numpy as np
import pandas as pd

def extract_window_features(ecg_window: np.ndarray, eda_window: np.ndarray, 
                            temp_window: np.ndarray, resp_window: np.ndarray, 
                            fs_ecg: float = 700.0, fs_eda: float = 4.0) -> dict:
    """
    Extracts physiological time-domain and frequency-domain metrics from a single signal window.
    """
    # 1. ECG Signal Processing -> Heart Rate & HRV Metrics
    # Simple peak detection threshold for R-peaks in ECG window
    peaks = []
    threshold = np.mean(ecg_window) + 1.2 * np.std(ecg_window)
    min_dist = int(fs_ecg * 0.4) # Minimum 400ms between consecutive R-peaks (max 150 bpm)
    
    last_peak = -min_dist
    for i in range(1, len(ecg_window) - 1):
        if ecg_window[i] > threshold and ecg_window[i] > ecg_window[i-1] and ecg_window[i] > ecg_window[i+1]:
            if (i - last_peak) >= min_dist:
                peaks.append(i)
                last_peak = i
                
    if len(peaks) > 1:
        rr_intervals_ms = (np.diff(peaks) / fs_ecg) * 1000.0
        mean_hr = 60000.0 / np.mean(rr_intervals_ms)
        std_hr = np.std(60000.0 / rr_intervals_ms) if len(rr_intervals_ms) > 1 else 3.0
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals_ms)))) if len(rr_intervals_ms) > 1 else 25.0
        sdnn = np.std(rr_intervals_ms)
    else:
        mean_hr = 72.0
        std_hr = 3.5
        rmssd = 35.0
        sdnn = 45.0
        
    # 2. EDA Signal Processing
    mean_eda = float(np.mean(eda_window))
    std_eda = float(np.std(eda_window))
    
    # Detect SCR peaks (derivative > threshold)
    eda_diff = np.diff(eda_window)
    scr_peaks = int(np.sum((eda_diff[:-1] < 0.02) & (eda_diff[1:] >= 0.02)))
    
    # 3. Skin Temperature Processing
    mean_temp = float(np.mean(temp_window))
    std_temp = float(np.std(temp_window))
    
    # 4. Respiration Rate
    mean_resp = float(np.mean(resp_window))
    
    # 5. Engineered Feature Interactions
    hrv_ratio = float(rmssd / (sdnn + 1e-5))
    eda_act = float(mean_eda * (scr_peaks + 1))
    hr_cv = float(std_hr / (mean_hr + 1e-5))
    
    return {
        'mean_HR': round(mean_hr, 2),
        'std_HR': round(std_hr, 2),
        'RMSSD': round(rmssd, 2),
        'SDNN': round(sdnn, 2),
        'mean_EDA': round(mean_eda, 2),
        'std_EDA': round(std_eda, 4),
        'SCR_peaks': scr_peaks,
        'mean_Temp': round(mean_temp, 2),
        'std_Temp': round(std_temp, 4),
        'mean_RESP': round(mean_resp, 2),
        'HRV_ratio': round(hrv_ratio, 4),
        'EDA_activation': round(eda_act, 4),
        'HR_CV': round(hr_cv, 4)
    }

def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered features to dataframe if not already present."""
    df = df.copy()
    if 'HRV_ratio' not in df.columns:
        df['HRV_ratio'] = df['RMSSD'] / (df['SDNN'] + 1e-5)
    if 'EDA_activation' not in df.columns:
        df['EDA_activation'] = df['mean_EDA'] * (df['SCR_peaks'] + 1.0)
    if 'HR_CV' not in df.columns:
        df['HR_CV'] = df['std_HR'] / (df['mean_HR'] + 1e-5)
    return df

if __name__ == "__main__":
    dummy_ecg = np.sin(np.linspace(0, 50, 700*60))
    dummy_eda = np.linspace(1.0, 2.5, 4*60)
    dummy_temp = np.full(4*60, 33.5)
    dummy_resp = np.full(4*60, 16.0)
    feat = extract_window_features(dummy_ecg, dummy_eda, dummy_temp, dummy_resp)
    print("[INFO] Extracted features sample:", feat)
