import sqlite3
import os
from pathlib import Path

def get_db():
    # Pobiera ścieżkę absolutną do katalogu głównego aplikacji (/app na Railway)
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Tworzymy bazę bezpośrednio w folderze głównym lub w dedykowanym data/
    db_dir = os.path.join(BASE_DIR, "backend", "data")
    os.makedirs(db_dir, exist_ok=True)
    
    db_path = os.path.join(db_dir, "app.db")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # 🛠️ AUTOMATYCZNA INICJALIZACJA TABEL (jeśli nie istnieją)
    # Zapobiega to błędowi typu "no such table: users" na nowym serwerze
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                left_id TEXT NOT NULL,
                right_id TEXT NOT NULL,
                choice TEXT NOT NULL,
                strength TEXT NOT NULL
            );
        """)
        conn.commit()
    except Exception as e:
        print(f"Błąd inicjalizacji bazy danych: {e}")
        
    return conn