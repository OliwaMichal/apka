import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent))
from db import get_db

def save_candidate(candidate: Dict[str, Any]):
    conn = get_db()
    cur = conn.cursor()

    # Konwersja cell_map: klucze mogą być krotkami lub stringami – normalizujemy do string
    cell_map_original = candidate.get("cell_map", {})
    cell_map_converted = {}
    for key, value in cell_map_original.items():
        if isinstance(key, tuple):
            new_key = f"{key[0]}|{key[1]}"
        else:
            new_key = str(key)
        cell_map_converted[new_key] = value

    cur.execute("""
        INSERT OR REPLACE INTO candidates (
            id, run, profile, days, hours, cell_map_json
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        candidate["id"],
        candidate.get("run", ""),
        candidate.get("profile", ""),
        json.dumps(candidate.get("days", [])),
        json.dumps(candidate.get("hours", [])),
        json.dumps(cell_map_converted)
    ))
    conn.commit()
    conn.close()

def load_candidates() -> List[Dict[str, Any]]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, run, profile, days, hours, cell_map_json FROM candidates")
    rows = cur.fetchall()
    conn.close()

    result = []
    for r in rows:
        cell_map = json.loads(r["cell_map_json"])  # klucze są stringami
        result.append({
            "id": r["id"],
            "run": r["run"],
            "profile": r["profile"],
            "days": json.loads(r["days"]),
            "hours": json.loads(r["hours"]),
            "cell_map": cell_map
        })
    return result