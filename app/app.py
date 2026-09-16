"""
Production-Grade Multi-Role Web Application (Streamlit + SQLite)
Project: AI-Based Physiological Stress Monitoring System
Features: Modern SaaS/Healthcare UI, Conditional Sidebar, Auth, Persistent Data Storage, SHAP XAI & Admin Panel
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from predict import StressPredictionEngine
from database import (init_db, authenticate_user, register_user, 
                      save_stress_log, get_user_logs, get_all_logs, get_admin_analytics)

# Initialize Database on app start
init_db()

st.set_page_config(
    page_title="MindPulse Health AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Landing Page Card Container */
    .auth-container {
        max-width: 480px;
        margin: 40px auto;
        background: #ffffff;
        border-radius: 20px;
        padding: 36px 40px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.08);
    }
    
    .brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
        background: linear-gradient(90deg, #ffffff, #99f6e4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .result-badge-stressed {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff;
        padding: 16px 24px;
        border-radius: 14px;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.25);
        margin-bottom: 16px;
    }
    
    .result-badge-normal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        padding: 16px 24px;
        border-radius: 14px;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.25);
        margin-bottom: 16px;
    }
    
    /* Button Polish */
    .stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(13, 148, 136, 0.4) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
    st.session_state['user'] = None

if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = "🔑 Account Login"

if 'prefilled_username' not in st.session_state:
    st.session_state['prefilled_username'] = ""

if 'reg_success_msg' not in st.session_state:
    st.session_state['reg_success_msg'] = ""

# ==============================================================================
# VIEW 1: AUTHENTICATION LANDING PAGE (UNAUTHENTICATED)
# ==============================================================================
if not st.session_state['authenticated']:
    # Hide ML Model Engine section from sidebar before login
    st.sidebar.title("🛡️ MindPulse Portal")
    st.sidebar.info("Please sign in or create an account to access the AI Stress Evaluation system.")
    
    # Clean Centered Production Landing Page Layout
    st.markdown("<br>", unsafe_allow_html=True)
    col_center1, col_center2, col_center3 = st.columns([1, 2.2, 1])
    
    with col_center2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <div class="brand-badge">🧠 Health AI Portal</div>
            <h1 style="font-weight: 800; color: #0f172a; font-size: 2.3rem; margin-bottom: 6px; letter-spacing: -0.5px;">MindPulse Stress AI</h1>
            <p style="color: #64748b; font-size: 1.05rem; font-weight: 500;">Physiological Stress Monitoring & Explainable AI Engine</p>
        </div>
        """, unsafe_allow_html=True)
        
        auth_mode = st.radio(
            "Auth Navigation",
            ["🔑 Account Login", "📝 Register New Account"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_mode"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state['reg_success_msg']:
            st.success(st.session_state['reg_success_msg'])
            st.session_state['reg_success_msg'] = ""
            
        if auth_mode == "🔑 Account Login":
            st.markdown("### Sign In")
            login_user = st.text_input("Username", value=st.session_state['prefilled_username'], key="l_user")
            login_pass = st.text_input("Password", type="password", key="l_pass")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In to Portal", type="primary", use_container_width=True):
                user = authenticate_user(login_user, login_pass)
                if user:
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = user
                    st.session_state['prefilled_username'] = ""
                    st.success(f"Welcome back, {user['full_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please check your username and password.")

        elif auth_mode == "📝 Register New Account":
            st.markdown("### Create Account")
            reg_name = st.text_input("Full Name", key="r_name")
            reg_user = st.text_input("Desired Username", key="r_user")
            reg_pass = st.text_input("Password", type="password", key="r_pass")
            reg_role = st.selectbox("Account Role", ["User", "Admin"], key="r_role")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account", type="primary", use_container_width=True):
                if reg_name and reg_user and reg_pass:
                    success, msg = register_user(reg_user, reg_pass, reg_name, reg_role)
                    if success:
                        st.session_state['auth_mode'] = "🔑 Account Login"
                        st.session_state['prefilled_username'] = reg_user.strip().lower()
                        st.session_state['reg_success_msg'] = "🎉 Account registered successfully! Please enter your password to log in."
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please complete all registration fields.")

        st.markdown("""
        <hr style="margin: 32px 0 20px 0; border: none; border-top: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-around; font-size: 0.85rem; color: #64748b; font-weight: 600;">
            <span>⚡ Subject-Aware ML</span>
            <span>🔍 SHAP Explainable AI</span>
            <span>🔒 Encrypted Database</span>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ==============================================================================
# VIEW 2: AUTHENTICATED USER / ADMIN DASHBOARDS
# ==============================================================================

# Get Current User Info
current_user = st.session_state['user']

# Load Model Engine ONLY AFTER LOGIN
@st.cache_resource
def load_engine(model_file: str):
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    return StressPredictionEngine(models_dir=models_dir, model_file=model_file)

# Sidebar Details (SHOW ML MODEL ENGINE ONLY WHEN LOGGED IN)
st.sidebar.title("🛡️ Portal Access")
st.sidebar.markdown(f"**User:** `{current_user['full_name']}`")
st.sidebar.markdown(f"**Role:** `{current_user['role']}` | **ID:** `{current_user['username']}`")

st.sidebar.markdown("---")
st.sidebar.title("🤖 ML Model Engine")
selected_model_name = st.sidebar.selectbox(
    "Choose Active Algorithm:",
    ["Random Forest", "XGBoost", "Baseline (Decision Tree)"]
)

model_file_map = {
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "Baseline (Decision Tree)": "decision_tree.pkl"
}

try:
    active_file = model_file_map.get(selected_model_name, "best_stress_model.pkl")
    engine = load_engine(active_file)
    st.sidebar.success(f"Active Engine: **{selected_model_name}**")
except Exception as e:
    st.error(f"❌ Error loading model: {e}. Ensure src/train.py has been executed.")
    st.stop()

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Sign Out", use_container_width=True):
    st.session_state['authenticated'] = False
    st.session_state['user'] = None
    st.rerun()

# Modern Hero Header
st.markdown("""
<div class="hero-banner">
    <div>
        <div class="hero-title">🧠 MindPulse AI Dashboard</div>
        <div class="hero-subtitle">Physiological Stress Classification & Explainable AI Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)

# USER DASHBOARD (ROLE = USER)
if current_user['role'] == 'User':
    st.subheader(f"👋 Welcome, {current_user['full_name']}")
    
    user_tab1, user_tab2 = st.tabs(["⚡ Perform Stress Assessment", "📊 My Assessment History"])
    
    with user_tab1:
        st.markdown("### 🫀 Physiological Signal Input")
        
        scenario = st.selectbox("Quick Preset Scenario:", ["Custom Manual Input", "Relaxed Baseline State", "High Stress State"])
        
        if scenario == "Relaxed Baseline State":
            d_hr, d_rmssd, d_sdnn, d_eda, d_scr, d_temp, d_resp = 68.0, 48.0, 55.0, 0.8, 1, 33.2, 15.5
        elif scenario == "High Stress State":
            d_hr, d_rmssd, d_sdnn, d_eda, d_scr, d_temp, d_resp = 95.0, 18.0, 24.0, 5.2, 7, 31.5, 24.0
        else:
            d_hr, d_rmssd, d_sdnn, d_eda, d_scr, d_temp, d_resp = 75.0, 35.0, 40.0, 2.0, 3, 32.8, 18.0

        col1, col2, col3 = st.columns(3)
        with col1:
            mean_HR = st.slider("Mean Heart Rate (bpm)", 45.0, 140.0, float(d_hr), 0.5)
            std_HR = st.slider("Std Dev Heart Rate", 0.5, 15.0, 4.0, 0.1)
            RMSSD = st.slider("RMSSD HRV (ms)", 5.0, 90.0, float(d_rmssd), 0.5)
            SDNN = st.slider("SDNN HRV (ms)", 10.0, 100.0, float(d_sdnn), 0.5)

        with col2:
            mean_EDA = st.slider("Mean EDA Level (µS)", 0.1, 12.0, float(d_eda), 0.1)
            std_EDA = st.slider("Std Dev EDA", 0.01, 1.5, 0.15, 0.01)
            SCR_peaks = st.slider("SCR Peak Count", 0, 15, int(d_scr), 1)

        with col3:
            mean_Temp = st.slider("Skin Temperature (°C)", 28.0, 37.0, float(d_temp), 0.1)
            std_Temp = st.slider("Std Dev Skin Temp", 0.01, 0.5, 0.05, 0.01)
            mean_RESP = st.slider("Respiration Rate (rpm)", 10.0, 35.0, float(d_resp), 0.5)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Analyze & Log Stress Assessment", type="primary", use_container_width=True):
            inputs = {
                'mean_HR': mean_HR, 'std_HR': std_HR, 'RMSSD': RMSSD, 'SDNN': SDNN,
                'mean_EDA': mean_EDA, 'std_EDA': std_EDA, 'SCR_peaks': SCR_peaks,
                'mean_Temp': mean_Temp, 'std_Temp': std_Temp, 'mean_RESP': mean_RESP
            }
            
            result = engine.predict(inputs)
            
            # Save log to SQLite database
            save_stress_log(current_user['id'], current_user['username'], inputs, result)
            
            st.markdown("<br>", unsafe_allow_html=True)
            res_c1, res_c2 = st.columns([1, 2])
            
            with res_c1:
                st.subheader("Classification Outcome")
                if result['prediction_class'] == 1:
                    st.markdown(f'<div class="result-badge-stressed">🚨 {result["prediction_label"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-badge-normal">✅ {result["prediction_label"]}</div>', unsafe_allow_html=True)
                    
                st.metric("Model Confidence", f"{result['confidence_percentage']}%")
                st.metric("Stress Risk Probability", f"{result['stress_probability']}%")
                
            with res_c2:
                st.subheader("🔍 SHAP Feature Attribution (Explainable AI)")
                contrib_df = pd.DataFrame(result['top_contributing_features'])
                
                fig, ax = plt.subplots(figsize=(8, 4.2))
                colors = ['#ef4444' if imp > 0 else '#10b981' for imp in contrib_df['shap_impact']]
                sns.barplot(data=contrib_df, x='shap_impact', y='feature', palette=colors, ax=ax)
                ax.set_title(f"SHAP Feature Drivers ({engine.model_name})", fontweight='bold', fontsize=12)
                ax.set_xlabel("SHAP Value (Positive = Risk Increase, Negative = Risk Decrease)")
                sns.despine()
                plt.tight_layout()
                st.pyplot(fig)

    with user_tab2:
        st.markdown("### 📜 Your Assessment History")
        user_logs = get_user_logs(current_user['id'])
        
        if not user_logs.empty:
            st.dataframe(user_logs, use_container_width=True)
            
            st.subheader("📈 Stress Risk Trajectory")
            fig_t, ax_t = plt.subplots(figsize=(10, 3.5))
            sns.lineplot(data=user_logs, x='timestamp', y='stress_probability', marker='o', color='#0f766e', linewidth=2.5, ax=ax_t)
            plt.xticks(rotation=45)
            ax_t.set_ylabel("Stress Probability (%)")
            ax_t.set_ylim(-5, 105)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig_t)
        else:
            st.info("No stress records logged yet. Perform an assessment above to record data.")

# ADMINISTRATOR DASHBOARD (ROLE = ADMIN)
elif current_user['role'] == 'Admin':
    st.subheader(f"👨‍⚕️ Administrator Management Console: {current_user['full_name']}")
    
    # KPI Analytics Cards
    analytics = get_admin_analytics()
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Registered Users</div><div class="kpi-value">{analytics["total_users"]}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Logs Recorded</div><div class="kpi-value">{analytics["total_evaluations"]}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">High-Stress Alerts</div><div class="kpi-value" style="color:#ef4444;">{analytics["high_stress_count"]}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">High-Stress Rate</div><div class="kpi-value">{analytics["stress_rate_percentage"]}%</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    admin_tab1, admin_tab2, admin_tab3 = st.tabs([
        "📊 Live User Monitoring Table", 
        "🚨 High-Stress Intervention Alerts", 
        "📥 Data Export (CSV)"
    ])
    
    all_logs = get_all_logs()
    
    with admin_tab1:
        st.markdown("### 👥 Patient & User Stress Monitoring")
        
        if not all_logs.empty:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                selected_user = st.selectbox("Filter by User:", ["All Users"] + sorted(all_logs['username'].unique().tolist()))
            with col_f2:
                selected_status = st.selectbox("Filter by Stress Status:", ["All Statuses", "STRESSED", "NON-STRESSED"])
                
            filtered_logs = all_logs.copy()
            if selected_user != "All Users":
                filtered_logs = filtered_logs[filtered_logs['username'] == selected_user]
            if selected_status != "All Statuses":
                filtered_logs = filtered_logs[filtered_logs['prediction_label'] == selected_status]
                
            st.dataframe(filtered_logs, use_container_width=True)
        else:
            st.info("No user stress records in database.")

    with admin_tab2:
        st.markdown("### 🚨 High-Stress Intervention Flags")
        if not all_logs.empty:
            stressed_logs = all_logs[all_logs['prediction_label'] == 'STRESSED']
            if not stressed_logs.empty:
                st.warning(f"⚠️ **{len(stressed_logs)} High-Stress Evaluations Flagged for Intervention**")
                st.dataframe(stressed_logs[['username', 'full_name', 'timestamp', 'mean_HR', 'RMSSD', 'mean_EDA', 'confidence_percentage', 'top_driver']], use_container_width=True)
            else:
                st.success("🎉 No high-stress flags recorded.")
        else:
            st.info("No data recorded.")

    with admin_tab3:
        st.markdown("### 📥 Database Management & Reports")
        if not all_logs.empty:
            csv_data = all_logs.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Database Report (CSV)",
                data=csv_data,
                file_name="stress_monitoring_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No records available to export.")
