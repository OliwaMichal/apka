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
    # CANDIDATES
    # =========================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            run TEXT,
            profile TEXT,
            days TEXT,
            hours TEXT,
            cell_map_json TEXT
        );
    """)

    conn.commit()

    # ==========================================
    # 🔥 AUTO GENEROWANIE KANDYDATÓW
    # ==========================================
    cur.execute("SELECT COUNT(*) FROM candidates")
    count = cur.fetchone()[0]

    conn.close()

    if count == 0:
        print("⚡ Brak kandydatów — generuję synthetic candidates...")

        from backend.generator import generate_and_save_synthetic_candidates

        generate_and_save_synthetic_candidates(
            n_candidates=30,
            random_state=42
        )

        print("✅ Kandydaci wygenerowani")

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

    cur.execute(
        "INSERT INTO users (name) VALUES (?)",
        (req.name,)
    )

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
        raise HTTPException(
            status_code=404,
            detail="Za mało kandydatów w bazie danych."
        )

    pair = get_unseen_pair(user_id, candidates)

    if pair is None:
        raise HTTPException(
            status_code=404,
            detail="Brak nowych par dla użytkownika."
        )

    left, right = pair

    mark_pair_as_shown(
        user_id,
        left["id"],
        right["id"]
    )

    return {
        "left": left,
        "right": right
    }

# ==========================================
# 💾 SAVE ANSWER
# ==========================================
@app.post("/answer")
def save_answer(req: AnswerRequest):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO answers (
            user_id,
            left_id,
            right_id,
            choice,
            strength
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        req.user_id,
        req.left_id,
        req.right_id,
        req.choice,
        req.strength
    ))

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

    cur.execute(
        "SELECT COUNT(*) FROM answers WHERE user_id = ?",
        (user_id,)
    )

    count = cur.fetchone()[0]

    conn.close()

    return {"count": count}