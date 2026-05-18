import sqlite3
import os
from pathlib import Path

def get_db():
    # Pobiera ścieżkę do folderu, w którym znajduje się obecny plik (backend/)
    BASE_DIR = Path(__file__).resolve().parent
    
    # Łączy ścieżkę z folderem data i plikiem app.db
    db_path = os.path.join(BASE_DIR, "data", "app.db")
    
    # Na wszelki wypadek: jeśli folder 'data' nie istnieje, stwórz go
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn