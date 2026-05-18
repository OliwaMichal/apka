import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "app.db"
conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM answers")
conn.execute("DELETE FROM shown_pairs")
# conn.execute("DELETE FROM users")  # odkomentuj, jeśli chcesz usunąć użytkowników
conn.commit()
conn.close()
print("Wyczyszczono odpowiedzi i historię par.")