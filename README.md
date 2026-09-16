# 🧠 AI-Based Physiological Stress Monitoring & Interpretability System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![XAI](https://img.shields.io/badge/Explainable%20AI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![UI](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)

An end-to-end Academic Machine Learning and Software Engineering system for **Physiological Stress Monitoring**. The system combines physiological signal feature extraction (ECG/HRV, EDA/GSR, Skin Temperature, Respiration), subject-aware ensemble machine learning (Random Forest & XGBoost), transparent decision explanations via SHAP (SHapley Additive exPlanations), an embedded SQLite persistent database, and a multi-role Web Portal for users and administrators.

---

## 📌 Executive Summary & Key Highlights

- **Subject-Aware ML Pipeline:** Utilizes `GroupShuffleSplit` on participant identifiers (`subject_id`) to ensure data from test participants remains completely unseen during training, preventing data leakage and guaranteeing true subject generalization.
- **Dual Ensemble ML Engines:** Implements and compares **Baseline Decision Tree**, **Random Forest**, and **XGBoost** classifiers.
- **Explainable AI (XAI) Integration:** Uses game-theoretic SHAP values (`TreeExplainer`) to break open the "black box" of ML predictions, showing global feature importances and local per-sample driver attribution charts.
- **Role-Based Web Portal:** Built with Streamlit, featuring secure SHA-256 user authentication supporting two distinct roles:
  - **User (Patient / Employee):** Evaluate real-time physiological stress, view confidence scores & SHAP feature attributions, and track personal historical stress trajectories.
  - **Administrator (Clinician / Manager):** Centralized control dashboard with system KPI cards, real-time user monitoring tables, high-stress intervention flags, and one-click CSV report exports.
- **Embedded Persistent Storage:** SQLite database (`database/stress_monitoring.db`) storing user accounts, roles, physiological feature inputs, prediction labels, confidence scores, and top SHAP feature drivers.
- **Software-Only Boundary:** The current phase operates exclusively on benchmark physiological datasets (`data/processed/physiological_stress_data.csv`). Physical wearable sensor hardware (ESP32/ECG/GSR) is specified as **Future Scope**.

---

## 🏗️ System Architecture

```
                                  [ Physiological Dataset ]
                                  (WESAD Benchmark Parameters)
                                               │
                                               ▼
                                 [ Preprocessing & Feature Engineering ]
                                 • HRV Ratio (RMSSD / SDNN)
                                 • EDA Activation (mean_EDA * SCR_peaks)
                                 • HR Coefficient of Variation
                                               │
                                               ▼
                               [ Subject-Aware Split (GroupShuffleSplit) ]
                                 • Train: 12 Participants (1,200 samples)
                                 • Test:   3 Participants (300 samples)
                                               │
                               ┌───────────────┴───────────────┐
                               ▼                               ▼
                      [ Random Forest ]                   [ XGBoost ]
                               │                               │
                               └───────────────┬───────────────┘
                                               ▼
                                  [ Model Serialization (.pkl) ]
                                               │
                                               ▼
                             [ Streamlit Multi-Role Web Dashboard ]
                                               │
                               ┌───────────────┴───────────────┐
                               ▼                               ▼
                     [ User Assessment ]             [ Admin Dashboard ]
                     • ML Inference                  • All Users Monitoring
                     • SHAP Attribution              • High-Stress Alerts
                     • Personal History              • CSV Export
                               │                               │
                               └───────────────┬───────────────┘
                                               ▼
                             [ SQLite Database (stress_monitoring.db) ]
```

---

## 📊 Dataset Specification (WESAD Benchmark)

The system is trained and validated using a physiological stress benchmark dataset structured after the **WESAD (Wearable Stress and Affect Detection)** protocol:

- **Total Samples:** 1,500 physiological windows
- **Total Participants:** 15 unique subjects (`S2` to `S16`)
- **Class Balance:**
  - `0 (Non-Stressed / Baseline)`: 900 samples (60.0%)
  - `1 (Stressed)`: 600 samples (40.0%)

### Physiological Signals & Extracted Features

| Feature Name | Signal Source | Description / Clinical Meaning |
| :--- | :--- | :--- |
| `mean_HR` | ECG / Pulse Sensor | Mean Heart Rate in beats per minute (bpm). |
| `std_HR` | ECG / Pulse Sensor | Standard deviation of heart rate. |
| `RMSSD` | ECG / HRV | Root mean square of successive RR interval differences (ms). Key parasympathetic indicator. |
| `SDNN` | ECG / HRV | Standard deviation of NN RR intervals (ms). Total autonomic variability indicator. |
| `mean_EDA` | GSR / Electrodermal | Mean Electrodermal Activity level (µS). Direct marker of sympathetic arousal. |
| `std_EDA` | GSR / Electrodermal | Standard deviation of EDA signal. |
| `SCR_peaks` | GSR / Electrodermal | Number of Skin Conductance Response bursts per window. |
| `mean_Temp` | Skin Temperature | Mean skin temperature (°C). Peripheral vasoconstriction causes drops during acute stress. |
| `std_Temp` | Skin Temperature | Standard deviation of skin temperature. |
| `mean_RESP` | Respiration | Respiration rate in breaths per minute. Elevated rate correlates with stress. |
| `HRV_ratio` | Engineered Metric | `RMSSD / SDNN`. Low values indicate sympathetic dominance. |
| `EDA_activation`| Engineered Metric | `mean_EDA * (SCR_peaks + 1)`. Quantifies overall electrodermal arousal. |
| `HR_CV` | Engineered Metric | `std_HR / mean_HR`. Normalized heart rate fluctuation. |

---

## 📈 Model Performance & Comparative Results

All models were evaluated using **Subject-Aware Group Train/Test Splitting** (`GroupShuffleSplit` on `subject_id`). Data from test participants (`S4`, `S6`, `S10`) was **completely unseen** during training to measure true user generalization:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Decision Tree)** | 99.67% | 99.17% | 100.00% | 99.59% | 0.9972 | Baseline Benchmark |
| **Random Forest** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **Winner (`random_forest.pkl`)** |
| **XGBoost** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **Top Engine (`xgboost.pkl`)** |

> All evaluation figures (Confusion Matrices, Metric Comparison Bar Charts, Feature Importance Plots) are stored in `results/`.

---

## 🔍 SHAP Explainable AI (XAI)

To resolve the opaque "black box" nature of machine learning, this project integrates **SHAP (SHapley Additive exPlanations)**:

1. **Global Interpretability:** Identifies top physiological drivers across the population (`EDA_activation`, `RMSSD`, `mean_EDA`, `HRV_ratio`).
2. **Local Instance Attribution:** Quantifies exact feature impact for individual user assessments (e.g., showing how elevated `mean_EDA` added +0.28 towards a high-stress score, while healthy `RMSSD` subtracted -0.28).

---

## 🗄️ Database Schema (`database/stress_monitoring.db`)

Built with SQLite and managed via `src/database.py`:

```sql
-- Users Table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'User',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stress Logs Table
CREATE TABLE stress_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    username TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mean_HR REAL, std_HR REAL, RMSSD REAL, SDNN REAL,
    mean_EDA REAL, std_EDA REAL, SCR_peaks INTEGER,
    mean_Temp REAL, std_Temp REAL, mean_RESP REAL,
    prediction_label TEXT NOT NULL,
    confidence_percentage REAL NOT NULL,
    stress_probability REAL NOT NULL,
    top_driver TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

---

## 📁 Repository Directory Structure

```
c:/ML_Project/
├── app/
│   └── app.py                              # Streamlit Multi-Role Web Dashboard UI
├── data/
│   └── processed/
│       └── physiological_stress_data.csv   # WESAD Benchmark Dataset (1,500 samples)
├── database/
│   └── stress_monitoring.db                # SQLite Persistent Database
├── models/
│   ├── best_stress_model.pkl               # Serialized Primary Model Artifact
│   ├── random_forest.pkl                   # Serialized Random Forest Model Artifact
│   ├── xgboost.pkl                         # Serialized XGBoost Model Artifact
│   ├── decision_tree.pkl                   # Serialized Decision Tree Baseline Artifact
│   ├── scaler.pkl                          # Serialized StandardScaler Artifact
│   └── model_metadata.pkl                  # Feature names & training metadata
├── results/
│   ├── model_comparison.csv                # Model Metrics CSV
│   ├── model_comparison_chart.png          # Performance Comparison Figure
│   ├── confusion_matrices.png              # Confusion Matrix Heatmap
│   ├── feature_importance.png              # Gini Feature Importance Chart
│   ├── shap_summary_beeswarm.png           # Global SHAP Beeswarm Plot
│   └── shap_feature_bar.png                # Global SHAP Bar Importance Chart
├── src/
│   ├── __init__.py                         # Package initializer
│   ├── data_loader.py                      # Loading & dataset verification script
│   ├── preprocessing.py                   # Subject-aware train/test split & scaling
│   ├── train.py                            # Model training & benchmarking script
│   ├── explainability.py                  # SHAP global & local calculation functions
│   ├── predict.py                          # Reusable inference engine
│   ├── database.py                         # SQLite database manager & auth functions
│   └── test_system.py                      # Automated boundary test suite
├── .gitignore                              # Git exclusion rules
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart & Execution Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/shahanaz-02/Stress_detection.git
cd Stress_detection

# Create Virtual Environment
python -m venv venv

# Activate Virtual Environment (Windows)
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Run Data Processing & Model Training Pipeline
```bash
python src/train.py
```

### 3. Generate SHAP Visualizations
```bash
python src/explainability.py
```

### 4. Run Automated System Boundary Tests
```bash
python src/test_system.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 System Boundary Testing Output

Automated tests executed via `src/test_system.py`:

```
============================================================
RUNNING COMPREHENSIVE SYSTEM TESTS & BOUNDARY VALIDATION
============================================================

                         Test Case                   Expected             Actual Confidence Status
     1. Normal Low-Stress Baseline               NON-STRESSED       NON-STRESSED     100.0% PASSED
              2. High-Stress State                   STRESSED           STRESSED     100.0% PASSED
        3. Boundary Extreme Values                   STRESSED           STRESSED     100.0% PASSED
4. SHAP Feature Attribution Output Valid Feature Drivers List 13 Features Ranked        N/A PASSED

============================================================
ALL SYSTEM TESTS COMPLETED SUCCESSFULLY
============================================================
```

---

## 🔮 Future Scope: Real-Time Hardware Integration

While the current project implements the **software and machine learning inference pipeline**, physical sensor data acquisition can be integrated in future work:

```
[ MAX30102 / Pulse Sensor ] ───► Heart Rate / ECG ────┐
[ GSR Electrodes ]          ───► Skin Conductance ────┼──► [ ESP32 Microcontroller ]
[ DS18B20 Temp Sensor ]     ───► Skin Temperature ────┘             │
                                                                    │ Wi-Fi / HTTP REST API
                                                                    ▼
                                                    [ Python Inference Engine ]
                                                    [ Streamlit Live Dashboard ]
```

---

## 📄 License & Academic Citation

This project is developed for academic research purposes.

**Author:** Shahanaz (`shahanaz-02`)  
**Repository:** [https://github.com/shahanaz-02/Stress_detection](https://github.com/shahanaz-02/Stress_detection)
