import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "app.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()

    # USERS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
    """)

    # ANSWERS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            left_id TEXT,
            right_id TEXT,
            choice TEXT,
            strength TEXT
        )
    """)

    # CANDIDATES
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            run TEXT,
            profile TEXT,
            days TEXT,
            hours TEXT,
            cell_map_json TEXT
        )
    """)

    # SHOWN PAIRS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS shown_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            left_id TEXT,
            right_id TEXT
        )
    """)

    conn.commit()

    return conn