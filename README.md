# AI-Based Physiological Stress Monitoring System

Academic ML project for physiological stress detection using benchmark datasets (WESAD / physiological signals), Random Forest, XGBoost, SHAP Explainable AI, and Streamlit.

## Project Structure
- `data/`: Contains raw and preprocessed datasets.
- `notebooks/`: Jupyter notebooks for data exploration and visualization.
- `src/`: Core Python modules for loading, preprocessing, training, explainability, and inference.
- `models/`: Serialized model artifacts (`.pkl`).
- `results/`: Evaluation figures, feature importance plots, and metric reports.
- `app/`: Interactive Streamlit Web UI.

## Getting Started
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Data Exploration & Preprocessing:
   ```bash
   python src/data_loader.py
   ```
