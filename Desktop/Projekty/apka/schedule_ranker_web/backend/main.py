import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.db import get_db
from backend.models import CreateUserRequest, AnswerRequest
from backend.candidate_service import load_candidates
from backend.pair_service import get_unseen_pair, mark_pair_as_shown

app = FastAPI()

# CORS – pozwól na requesty z frontendu (Streamlit Cloud)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # w produkcji można ograniczyć do domeny frontendu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/user")
def create_user(req: CreateUserRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (name) VALUES (?)", (req.name,))
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {"user_id": user_id}

@app.get("/pair")
def get_pair(user_id: int):
    candidates = load_candidates()
    if len(candidates) < 2:
        raise HTTPException(404, "Za mało kandydatów w bazie (minimum 2).")

    pair = get_unseen_pair(user_id, candidates)
    if pair is None:
        raise HTTPException(404, "Brak nowych par dla tego użytkownika – wszystkie zostały już pokazane.")

    left, right = pair
    mark_pair_as_shown(user_id, left["id"], right["id"])

    return {"left": left, "right": right}

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
    return {"ok": True}

@app.get("/progress")
def progress(user_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM answers WHERE user_id = ?", (user_id,))
    count = cur.fetchone()[0]
    conn.close()
    return {"count": count}