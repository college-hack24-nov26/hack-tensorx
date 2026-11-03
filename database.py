import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "ecg.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Ensure table exists with base columns
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        prediction TEXT,
        nsr REAL,
        afib REAL,
        vfib REAL
    )
    """)
    
    # 2. Check for and add missing vital columns (Migration)
    # This ensures existing databases are updated without needing to delete them
    c.execute("PRAGMA table_info(history)")
    columns = [info[1] for info in c.fetchall()]
    
    new_columns = {
        "heart_rate": "INTEGER",
        "bp_sys": "INTEGER",
        "bp_dia": "INTEGER",
        "spo2": "INTEGER"
    }
    
    for col_name, col_type in new_columns.items():
        if col_name not in columns:
            print(f"Migrating: Adding column {col_name} to history table.")
            c.execute(f"ALTER TABLE history ADD COLUMN {col_name} {col_type}")
            
    conn.commit()
    conn.close()

def save_prediction(pred, probs, vitals=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hr = vitals.get('heart_rate') if vitals else None
    bp_sys = vitals.get('bp_sys') if vitals else None
    bp_dia = vitals.get('bp_dia') if vitals else None
    spo2 = vitals.get('spo2') if vitals else None
    
    c.execute(
        "INSERT INTO history (prediction, nsr, afib, vfib, heart_rate, bp_sys, bp_dia, spo2) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pred, probs["NSR"], probs["AFib"], probs["VTCC"], hr, bp_sys, bp_dia, spo2)
    )
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, prediction, nsr, afib, vfib, heart_rate, bp_sys, bp_dia, spo2 FROM history ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return rows

def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()
