"""
Utility Functions & Directory Helper Module
Project: AI-Based Physiological Stress Monitoring System (WESAD LOSO Pipeline)
"""

import os
import sys

# Define Root Project Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw", "WESAD")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

def ensure_directories():
    """Ensures all project output directories exist."""
    for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("[INFO] Directories verified.")
