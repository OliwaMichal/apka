import os
import sqlite3
from pathlib import Path

# DB_PATH z zmiennej środowiskowej lub domyślnie
DB_PATH = os.getenv(
    "DB_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "app.db")
)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn