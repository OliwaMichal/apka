import json
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent))
from db import get_db

def save_candidate(candidate: Dict[str, Any]):
    conn = get_db()
    cur = conn.cursor()

    # Konwersja cell_map
    cell_map_original = candidate.get("cell_map", {})
    cell_map_converted = {}
    for key, value in cell_map_original.items():
        if isinstance(key, tuple):
            new_key = f"{key[0]}|{key[1]}"
        else:
            new_key = str(key)
        cell_map_converted[new_key] = value

    # Wstawianie lub aktualizacja (INSERT OR REPLACE) – najpierw podstawowe dane
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

    # Aktualizacja cech (metryk)
    feature_cols = [
        "campus_switch_0", "campus_switch_1",
        "gaps1", "gaps2p", "single_class_days", "long_streak_days",
        "dayoff_count", "days_with_classes", "total_activities",
        "earliest_start_mean", "latest_end_mean", "daily_span_mean",
        "morning_classes_count", "late_classes_count",
        "lab_days", "odd_even_imbalance", "mixed_type_days",
        "friday_penalty", "monday_bonus", "multi_campus_days",
        "friday_late_classes", "campus_rush_days", "daily_load_variance"
    ]
    set_clause = ", ".join([f"{col} = ?" for col in feature_cols])
    values = [candidate.get(col, 0) for col in feature_cols]
    values.append(candidate["id"])

    cur.execute(f"""
        UPDATE candidates
        SET {set_clause}
        WHERE id = ?
    """, values)

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
        cell_map = json.loads(r["cell_map_json"])
        result.append({
            "id": r["id"],
            "run": r["run"],
            "profile": r["profile"],
            "days": json.loads(r["days"]) if r["days"] else [],
            "hours": json.loads(r["hours"]) if r["hours"] else [],
            "cell_map": cell_map
        })
    return result