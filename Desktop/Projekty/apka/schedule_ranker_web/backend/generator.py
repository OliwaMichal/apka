import sys
import sqlite3
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from backend.db import DB_PATH
from backend.candidate_service import save_candidate
from backend.fet_ltr import build_synthetic_candidates

def generate_and_save_synthetic_candidates(n_candidates: int = 30, random_state: int = 42):
    df = build_synthetic_candidates(n_candidates, random_state)
    for _, row in df.iterrows():
        candidate = {
            "id": row["candidate_id"],
            "run": row["run"],
            "profile": row["profile"],
            "days": row["days"],
            "hours": row["hours"],
            "cell_map": row["cell_map"],
            "campus_switch_0": row["campus_switch_0"],
            "campus_switch_1": row["campus_switch_1"],
            "gaps1": row["gaps1"],
            "gaps2p": row["gaps2p"],
            "single_class_days": row["single_class_days"],
            "long_streak_days": row["long_streak_days"],
            "dayoff_count": row["dayoff_count"],
            "days_with_classes": row["days_with_classes"],
            "total_activities": row["total_activities"],
            "earliest_start_mean": row["earliest_start_mean"],
            "latest_end_mean": row["latest_end_mean"],
            "daily_span_mean": row["daily_span_mean"],
            "morning_classes_count": row["morning_classes_count"],
            "late_classes_count": row["late_classes_count"],
            "lab_days": row["lab_days"],
            "odd_even_imbalance": row["odd_even_imbalance"],
            "mixed_type_days": row["mixed_type_days"],
            "friday_penalty": row["friday_penalty"],
            "monday_bonus": row["monday_bonus"],
            "multi_campus_days": row["multi_campus_days"],
            "friday_late_classes": row["friday_late_classes"],
            "campus_rush_days": row["campus_rush_days"],
            "daily_load_variance": row["daily_load_variance"]
        }
        save_candidate(candidate)
        print(f"✅ Zapisano {candidate['id']}")

if __name__ == "__main__":
    if "--clear" in sys.argv:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM candidates")
        conn.commit()
        conn.close()
        print("🧹 Wyczyszczono tabelę candidates.")
    generate_and_save_synthetic_candidates(n_candidates=30, random_state=42)