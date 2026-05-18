import sqlite3
from backend.db import DB_PATH
from backend.generator import generate_sample_candidates

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            run TEXT,
            profile TEXT,
            days TEXT,
            hours TEXT,
            cell_map_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            left_id TEXT,
            right_id TEXT,
            choice TEXT,
            strength TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shown_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            left_id TEXT,
            right_id TEXT,
            shown_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    
    # "Na brutalnie" — przy każdym wymuszeniu uruchomienia generujemy świeżą paczkę kandydatów
    print("[INIT_DB] Wywoływanie generatora planów zajęć...")
    generate_sample_candidates(12)

if __name__ == "__main__":
    init_db()
    print("[INIT_DB] Baza pomyślnie zresetowana i uzupełniona!")