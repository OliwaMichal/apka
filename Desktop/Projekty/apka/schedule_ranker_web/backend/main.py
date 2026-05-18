import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.db import get_db
from backend.models import AnswerRequest
from backend.candidate_service import load_candidates
from backend.pair_service import get_unseen_pair, mark_pair_as_shown

app = FastAPI()

# ==========================================
# 🌍 CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 👤 REQUEST MODELS
# ==========================================
class UserRequest(BaseModel):
    name: str

# ==========================================
# 🚀 STARTUP — inicjalizacja bazy i kandydatów
# ==========================================
@app.on_event("startup")
def startup_event():
    conn = get_db()
    cur = conn.cursor()

    # =========================
    # USERS
    # =========================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        );
    """)

    # =========================
    # ANSWERS
    # =========================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            left_id TEXT NOT NULL,
            right_id TEXT NOT NULL,
            choice TEXT NOT NULL,
            strength TEXT NOT NULL
        );
    """)

    # =========================
    # SHOWN PAIRS
    # =========================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS shown_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            left_id TEXT,
            right_id TEXT
        );
    """)

    # =========================
    # CANDIDATES (z cechami)
    # =========================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            run TEXT,
            profile TEXT,
            days TEXT,
            hours TEXT,
            cell_map_json TEXT,
            campus_switch_0 INTEGER,
            campus_switch_1 INTEGER,
            gaps1 INTEGER,
            gaps2p INTEGER,
            single_class_days INTEGER,
            long_streak_days INTEGER,
            dayoff_count INTEGER,
            days_with_classes INTEGER,
            total_activities INTEGER,
            earliest_start_mean REAL,
            latest_end_mean REAL,
            daily_span_mean REAL,
            morning_classes_count INTEGER,
            late_classes_count INTEGER,
            lab_days INTEGER,
            odd_even_imbalance INTEGER,
            mixed_type_days INTEGER,
            friday_penalty INTEGER,
            monday_bonus INTEGER,
            multi_campus_days INTEGER,
            friday_late_classes INTEGER,
            campus_rush_days INTEGER,
            daily_load_variance REAL
        );
    """)

    conn.commit()

    # ==========================================
    # 🔥 AUTO GENEROWANIE KANDYDATÓW (jeśli brak)
    # ==========================================
    cur.execute("SELECT COUNT(*) FROM candidates")
    count = cur.fetchone()[0]

    if count == 0:
        print("⚡ Brak kandydatów — generuję synthetic candidates...")
        from backend.generator import generate_and_save_synthetic_candidates
        generate_and_save_synthetic_candidates(n_candidates=30, random_state=42)
        print("✅ Kandydaci wygenerowani")

    conn.close()

# ==========================================
# 🏠 ROOT
# ==========================================
@app.get("/")
def read_root():
    return {"status": "Backend działa poprawnie"}

# ==========================================
# 👤 CREATE USER
# ==========================================
@app.post("/user")
def create_user(req: UserRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (name) VALUES (?)", (req.name,))
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {"user_id": user_id}

# ==========================================
# 📊 GET PAIR
# ==========================================
@app.get("/pair")
def get_pair(user_id: int):
    candidates = load_candidates()
    if len(candidates) < 2:
        raise HTTPException(404, "Za mało kandydatów w bazie danych.")
    pair = get_unseen_pair(user_id, candidates)
    if pair is None:
        raise HTTPException(404, "Brak nowych par dla użytkownika.")
    left, right = pair
    mark_pair_as_shown(user_id, left["id"], right["id"])
    return {"left": left, "right": right}

# ==========================================
# 💾 SAVE ANSWER
# ==========================================
@app.post("/answer")
def save_answer(req: AnswerRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO answers (user_id, left_id, right_id, choice, strength)
        VALUES (?, ?, ?, ?, ?)
    """, (req.user_id, req.left_id, req.right_id, req.choice, req.strength))
    conn.commit()
    conn.close()
    return {"status": "ok"}

# ==========================================
# 📈 PROGRESS
# ==========================================
@app.get("/progress")
def get_progress(user_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM answers WHERE user_id = ?", (user_id,))
    count = cur.fetchone()[0]
    conn.close()
    return {"count": count}

# ==========================================
# 📋 ANSWERS Z DANYMI KANDYDATÓW (cechy)
# ==========================================
def get_candidate_by_id(candidate_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

@app.get("/answers")
def get_answers():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM answers ORDER BY id DESC")
    answers = cur.fetchall()
    conn.close()

    result = []
    for ans in answers:
        left = get_candidate_by_id(ans["left_id"])
        right = get_candidate_by_id(ans["right_id"])
        result.append({
            "id": ans["id"],
            "user_id": ans["user_id"],
            "choice": ans["choice"],
            "strength": ans["strength"],
            "left_candidate": left,
            "right_candidate": right
        })
    return result