# 🧠 AI-Based Physiological Stress Monitoring & Interpretability System (WESAD LOSO Pipeline)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![XAI](https://img.shields.io/badge/Explainable%20AI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Validation](https://img.shields.io/badge/Validation-15--Fold%20LOSO%20CV-purple.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html)
[![UI](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)

An academic Machine Learning and Software Engineering system for **Physiological Stress Detection**. The system processes signal windows (ECG, EDA, Skin Temperature, Respiration), performs signal feature extraction, executes **15-Fold Leave-One-Subject-Out (LOSO) Cross-Validation**, generates **SHAP decision explanations**, and serves predictions via an interactive **Streamlit Multi-Role Web Application** backed by an embedded SQLite database.

---

## 📌 Key Architectural Highlights

- **15-Fold Leave-One-Subject-Out (LOSO) Cross-Validation:** Evaluated using `LeaveOneGroupOut` across all 15 authentic WESAD subjects (`S2` to `S17`, `S12` excluded). In each fold, 14 subjects train the model, and 1 isolated subject tests the model.
- **Strict Fold Preprocessing (Zero Data Leakage):** Scaler fitting occurs strictly inside each fold on training subjects (`scaler.fit(X_train)`). Test subject data is transformed exclusively using the training fold scaler (`X_test = scaler.transform(X_test)`).
- **Physiological Signal Extraction (`src/feature_extraction.py`):** Calculates time-domain and HRV metrics from ECG (R-peaks, RR-intervals $\rightarrow$ `mean_HR`, `std_HR`, `RMSSD`, `SDNN`), electrodermal response from EDA (`mean_EDA`, `std_EDA`, `SCR_peaks`), skin temperature dynamics (`mean_Temp`, `std_Temp`), respiration rate (`mean_RESP`), and feature interactions (`HRV_ratio`, `EDA_activation`, `HR_CV`).
- **Ensemble ML Benchmark:** Trains and evaluates **Baseline Decision Tree**, **Random Forest**, and **XGBoost** classifiers.
- **SHAP Explainable AI (XAI):** Game-theoretic feature attributions via `TreeExplainer` producing global summary plots (`results/shap_summary.png`) and local instance feature driver charts.
- **Multi-Role Web Application (`app/app.py`):** Streamlit Web UI featuring secure user authentication (`User` vs `Admin`), personal stress evaluations, historical trend line charts, central admin monitoring, and one-click CSV report downloads.

---

## 🏗️ System Architecture & Workflow

```
Stress_detection/
├── data/
│   ├── raw/WESAD/                        # Authentic WESAD recording windows
│   └── processed/wesad_features.csv      # Signal feature matrix (3,300 rows)
├── src/
│   ├── data_loader.py                    # Dataset loader & window processing trigger
│   ├── preprocessing.py                 # Strict fold-level scaler fitting (fit_scale_fold)
│   ├── feature_extraction.py             # Signal feature calculation (ECG, EDA, Temp, Resp)
│   ├── train.py                          # 15-Fold Leave-One-Subject-Out CV & serialization
│   ├── evaluate.py                       # Metric evaluation, confusion matrix & ROC curves
│   ├── explainability.py                 # SHAP global & local feature attributions
│   ├── database.py                       # SQLite database manager & auth functions
│   ├── test_system.py                    # Automated boundary test suite
│   └── utils.py                          # Path constants & directory helpers
├── models/
│   ├── random_forest.pkl                 # Serialized Random Forest model artifact
│   └── xgboost.pkl                       # Serialized XGBoost model artifact
├── results/
│   ├── metrics.csv                       # Aggregated LOSO evaluation metrics CSV
│   ├── confusion_matrix.png              # LOSO Confusion Matrix figure
│   ├── roc_curve.png                     # LOSO ROC Curve figure
│   └── shap_summary.png                  # Global SHAP summary plot
├── app/
│   └── app.py                            # Streamlit Web Application
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 📊 Dataset Specifications (Authentic WESAD)

Reference: *Schmidt et al. (2018), "Introducing WESAD: a multimodal dataset for wearable stress and affect detection," ACM ICMI.*

- **Total Samples:** 3,310 windowed signal records extracted from raw 700Hz continuous RespiBAN chest recordings
- **Authentic Subjects (15):** `S2`, `S3`, `S4`, `S5`, `S6`, `S7`, `S8`, `S9`, `S10`, `S11`, `S13`, `S14`, `S15`, `S16`, `S17`  
  *(Note: Subject `S12` is excluded in strict accordance with the official WESAD paper due to sensor failure).*
- **Class Balance:**
  - `0 (Baseline / Non-Stressed)`: 2,110 samples (63.75%)
  - `1 (Stressed - TSST Task)`: 1,200 samples (36.25%)

### Extracted Physiological Features

| Feature Name | Signal Source | Description / Clinical Significance |
| :--- | :--- | :--- |
| `mean_HR` | ECG / Pulse | Mean Heart Rate in beats per minute (bpm). |
| `std_HR` | ECG / Pulse | Standard deviation of heart rate. |
| `RMSSD` | ECG / HRV | Root mean square of successive RR interval differences (ms). Parasympathetic indicator. |
| `SDNN` | ECG / HRV | Standard deviation of NN RR intervals (ms). Total autonomic HRV variability. |
| `mean_EDA` | GSR / EDA | Mean Electrodermal Activity level (µS). Direct marker of sympathetic arousal. |
| `std_EDA` | GSR / EDA | Standard deviation of skin conductance. |
| `SCR_peaks` | GSR / EDA | Number of Skin Conductance Response bursts per window. |
| `mean_Temp` | Skin Temperature | Mean skin temperature (°C). Vasoconstriction causes drops during acute stress. |
| `std_Temp` | Skin Temperature | Standard deviation of skin temperature. |
| `mean_RESP` | Respiration | Respiration rate in breaths per minute. |
| `HRV_ratio` | Engineered Metric | `RMSSD / (SDNN + 1e-5)`. Low values indicate sympathetic dominance. |
| `EDA_activation`| Engineered Metric | `mean_EDA * (SCR_peaks + 1)`. Quantifies overall electrodermal arousal. |
| `HR_CV` | Engineered Metric | `std_HR / (mean_HR + 1e-5)`. Normalized heart rate fluctuation. |

---

## 📈 Aggregated Authentic LOSO Cross-Validation Results (`results/metrics.csv`)

Performance evaluated across **15 independent test folds** (where the test subject was completely unseen during training):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Decision Tree)** | 80.63% | 68.40% | 66.00% | 67.18% | 0.7886 | LOSO Baseline |
| **XGBoost** | 86.59% | 75.23% | 82.49% | 78.69% | 0.9402 | Top Gradient Engine |
| **Random Forest** | **88.52%** | **79.92%** | **82.49%** | **81.19%** | **0.9453** | **Winner (`random_forest.pkl`)** |

---

## 🔍 SHAP Explainable AI (XAI) Output

Global feature importances and local per-instance predictions are generated via `shap.TreeExplainer`:
- **Top Risk Drivers:** `EDA_activation`, `RMSSD`, `mean_EDA`, `HRV_ratio`, `mean_Temp`.
- **Saved Visualizations:**
  - Global Summary Plot: [`results/shap_summary.png`](file:///c:/ML_Project/results/shap_summary.png)
  - Confusion Matrix: [`results/confusion_matrix.png`](file:///c:/ML_Project/results/confusion_matrix.png)
  - ROC Curve: [`results/roc_curve.png`](file:///c:/ML_Project/results/roc_curve.png)

---

## 🚀 Installation & Execution Guide

### 1. Environment Setup
```bash
git clone https://github.com/shahanaz-02/Stress_detection.git
cd Stress_detection

# Create and Activate Virtual Environment
python -m venv venv
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Run 15-Fold LOSO Training & Evaluation Pipeline
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

### 5. Launch Interactive Web Portal
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

---

## 📄 License & Academic Citation

Developed for academic research and final year AIML presentation.

**Author:** Shahanaz (`shahanaz-02`)  
**Repository:** [https://github.com/shahanaz-02/Stress_detection](https://github.com/shahanaz-02/Stress_detection)
