import sys
import sqlite3

from backend.db import DB_PATH
from backend.candidate_service import save_candidate
from backend.fet_ltr import build_synthetic_candidates

def generate_and_save_synthetic_candidates(
    n_candidates: int = 30,
    random_state: int = 42
):
    df = build_synthetic_candidates(n_candidates, random_state)

    for _, row in df.iterrows():
        candidate = {
            "id": row["candidate_id"],
            "run": row["run"],
            "profile": row["profile"],
            "days": row["days"],
            "hours": row["hours"],
            "cell_map": row["cell_map"]
        }

        save_candidate(candidate)

        print(
            f"✅ Zapisano {candidate['id']} "
            f"(profil: {candidate['profile']})"
        )

if __name__ == "__main__":

    if "--clear" in sys.argv:
        conn = sqlite3.connect(DB_PATH)

        conn.execute("DELETE FROM candidates")

        conn.commit()
        conn.close()

        print("🧹 Wyczyszczono tabelę candidates.")

    generate_and_save_synthetic_candidates(
        n_candidates=30,
        random_state=42
    )