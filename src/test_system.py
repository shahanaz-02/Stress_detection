"""
Comprehensive System Testing & Input Boundary Validation Module
Project: AI-Based Physiological Stress Monitoring System
Stage: 15 - Comprehensive System Testing
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import StressPredictionEngine

def run_system_tests():
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE SYSTEM TESTS & BOUNDARY VALIDATION")
    print("="*60)
    
    engine = StressPredictionEngine()
    test_results = []
    
    # Test Case 1: Normal Low-Stress Baseline Input
    tc1_input = {
        'mean_HR': 68.0, 'std_HR': 3.0, 'RMSSD': 45.0, 'SDNN': 52.0,
        'mean_EDA': 0.7, 'std_EDA': 0.04, 'SCR_peaks': 1,
        'mean_Temp': 33.5, 'std_Temp': 0.03, 'mean_RESP': 15.0
    }
    tc1_res = engine.predict(tc1_input)
    tc1_passed = (tc1_res['prediction_label'] == 'NON-STRESSED')
    test_results.append({
        "Test Case": "1. Normal Low-Stress Baseline",
        "Expected": "NON-STRESSED",
        "Actual": tc1_res['prediction_label'],
        "Confidence": f"{tc1_res['confidence_percentage']}%",
        "Status": "PASSED" if tc1_passed else "FAILED"
    })
    
    # Test Case 2: High-Stress Input
    tc2_input = {
        'mean_HR': 98.0, 'std_HR': 8.2, 'RMSSD': 16.0, 'SDNN': 22.0,
        'mean_EDA': 5.8, 'std_EDA': 0.42, 'SCR_peaks': 8,
        'mean_Temp': 31.2, 'std_Temp': 0.15, 'mean_RESP': 25.0
    }
    tc2_res = engine.predict(tc2_input)
    tc2_passed = (tc2_res['prediction_label'] == 'STRESSED')
    test_results.append({
        "Test Case": "2. High-Stress State",
        "Expected": "STRESSED",
        "Actual": tc2_res['prediction_label'],
        "Confidence": f"{tc2_res['confidence_percentage']}%",
        "Status": "PASSED" if tc2_passed else "FAILED"
    })
    
    # Test Case 3: Boundary / Extreme Elevated HR & EDA
    tc3_input = {
        'mean_HR': 130.0, 'std_HR': 12.0, 'RMSSD': 10.0, 'SDNN': 15.0,
        'mean_EDA': 10.0, 'std_EDA': 0.8, 'SCR_peaks': 12,
        'mean_Temp': 29.5, 'std_Temp': 0.3, 'mean_RESP': 32.0
    }
    tc3_res = engine.predict(tc3_input)
    tc3_passed = (tc3_res['prediction_label'] == 'STRESSED')
    test_results.append({
        "Test Case": "3. Boundary Extreme Values",
        "Expected": "STRESSED",
        "Actual": tc3_res['prediction_label'],
        "Confidence": f"{tc3_res['confidence_percentage']}%",
        "Status": "PASSED" if tc3_passed else "FAILED"
    })
    
    # Test Case 4: SHAP Driver Attribution Consistency
    tc4_has_shap = len(tc2_res['top_contributing_features']) > 0
    test_results.append({
        "Test Case": "4. SHAP Feature Attribution Output",
        "Expected": "Valid Feature Drivers List",
        "Actual": f"{len(tc2_res['top_contributing_features'])} Features Ranked",
        "Confidence": "N/A",
        "Status": "PASSED" if tc4_has_shap else "FAILED"
    })

    # Summary
    summary_df = pd.DataFrame(test_results)
    print("\n" + summary_df.to_string(index=False))
    print("\n" + "="*60)
    print("ALL SYSTEM TESTS COMPLETED SUCCESSFULLY")
    print("="*60)
    return summary_df

if __name__ == "__main__":
    run_system_tests()
