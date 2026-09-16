"""
Database Management Module (SQLite)
Project: AI-Based Physiological Stress Monitoring System
Handles User Authentication, Roles & Persistent Stress Logs Storage
"""

import os
import sqlite3
import hashlib
from datetime import datetime
import pandas as pd

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))
DB_PATH = os.path.join(DB_DIR, "stress_monitoring.db")

def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Users table (Authentication & Roles)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'User',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Stress logs table (Data Storage)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stress_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        username TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        mean_HR REAL,
        std_HR REAL,
        RMSSD REAL,
        SDNN REAL,
        mean_EDA REAL,
        std_EDA REAL,
        SCR_peaks INTEGER,
        mean_Temp REAL,
        std_Temp REAL,
        mean_RESP REAL,
        prediction_label TEXT NOT NULL,
        confidence_percentage REAL NOT NULL,
        stress_probability REAL NOT NULL,
        top_driver TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """)
    
    conn.commit()
    conn.close()
    seed_default_users()

def seed_default_users():
    """Seeds default demo accounts for Admin and standard User."""
    register_user("admin", "admin123", "System Administrator", role="Admin")
    register_user("user1", "user123", "John Doe (Patient S2)", role="User")
    register_user("user2", "user123", "Alice Smith (Patient S3)", role="User")

def register_user(username: str, password: str, full_name: str, role: str = "User") -> tuple[bool, str]:
    conn = get_connection()
    cursor = conn.cursor()
    
    pw_hash = hash_password(password)
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
            (username.strip().lower(), pw_hash, full_name.strip(), role)
        )
        conn.commit()
        conn.close()
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists. Please choose a different username."
    except Exception as e:
        conn.close()
        return False, f"Registration error: {e}"

def authenticate_user(username: str, password: str):
    conn = get_connection()
    cursor = conn.cursor()
    
    pw_hash = hash_password(password)
    cursor.execute(
        "SELECT id, username, full_name, role FROM users WHERE username = ? AND password_hash = ?",
        (username.strip().lower(), pw_hash)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None

def save_stress_log(user_id: int, username: str, inputs: dict, prediction_result: dict) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    
    top_driver_name = "N/A"
    if prediction_result.get("top_contributing_features"):
        top = prediction_result["top_contributing_features"][0]
        top_driver_name = f"{top['feature']} ({top['effect']})"
        
    try:
        cursor.execute("""
        INSERT INTO stress_logs (
            user_id, username, timestamp, mean_HR, std_HR, RMSSD, SDNN,
            mean_EDA, std_EDA, SCR_peaks, mean_Temp, std_Temp, mean_RESP,
            prediction_label, confidence_percentage, stress_probability, top_driver
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            inputs.get('mean_HR'), inputs.get('std_HR'), inputs.get('RMSSD'), inputs.get('SDNN'),
            inputs.get('mean_EDA'), inputs.get('std_EDA'), inputs.get('SCR_peaks'),
            inputs.get('mean_Temp'), inputs.get('std_Temp'), inputs.get('mean_RESP'),
            prediction_result['prediction_label'], prediction_result['confidence_percentage'],
            prediction_result['stress_probability'], top_driver_name
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save stress log: {e}")
        conn.close()
        return False

def get_user_logs(user_id: int) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT timestamp, mean_HR, RMSSD, mean_EDA, mean_Temp, mean_RESP, prediction_label, confidence_percentage, stress_probability, top_driver FROM stress_logs WHERE user_id = ? ORDER BY id DESC",
        conn, params=(user_id,)
    )
    conn.close()
    return df

def get_all_logs() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT sl.id, sl.username, u.full_name, sl.timestamp, sl.mean_HR, sl.RMSSD, sl.mean_EDA, sl.mean_Temp, 
               sl.prediction_label, sl.confidence_percentage, sl.stress_probability, sl.top_driver 
        FROM stress_logs sl
        LEFT JOIN users u ON sl.user_id = u.id
        ORDER BY sl.id DESC
        """,
        conn
    )
    conn.close()
    return df

def get_admin_analytics() -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'User'")
    total_users = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM stress_logs")
    total_evaluations = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM stress_logs WHERE prediction_label = 'STRESSED'")
    high_stress_count = cursor.fetchone()[0]
    
    conn.close()
    
    stress_rate = (high_stress_count / total_evaluations * 100.0) if total_evaluations > 0 else 0.0
    
    return {
        "total_users": total_users,
        "total_evaluations": total_evaluations,
        "high_stress_count": high_stress_count,
        "stress_rate_percentage": round(stress_rate, 1)
    }

if __name__ == "__main__":
    init_db()
    print("[SUCCESS] SQLite Database initialized at:", DB_PATH)
