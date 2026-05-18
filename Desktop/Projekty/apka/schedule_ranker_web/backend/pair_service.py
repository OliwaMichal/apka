import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

sys.path.append(str(Path(__file__).parent))
from db import get_db

def get_unseen_pair(user_id: int, candidates: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if len(candidates) < 2:
        return None

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT left_id, right_id FROM shown_pairs WHERE user_id = ?", (user_id,))
    seen = set()
    for row in cur.fetchall():
        seen.add((row["left_id"], row["right_id"]))
        seen.add((row["right_id"], row["left_id"]))
    conn.close()

    cand_list = [(c["id"], c) for c in candidates]
    random.shuffle(cand_list)

    for i in range(len(cand_list)):
        for j in range(i+1, len(cand_list)):
            left_id, left_cand = cand_list[i]
            right_id, right_cand = cand_list[j]
            if (left_id, right_id) not in seen:
                return left_cand, right_cand
    return None

def mark_pair_as_shown(user_id: int, left_id: str, right_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO shown_pairs (user_id, left_id, right_id) VALUES (?, ?, ?)",
        (user_id, left_id, right_id)
    )
    conn.commit()
    conn.close()