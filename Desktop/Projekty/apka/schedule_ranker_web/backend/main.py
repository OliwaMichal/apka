import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.db import get_db
from backend.models import CreateUserRequest, AnswerRequest
from backend.candidate_service import load_candidates
from backend.pair_service import get_unseen_pair, mark_pair_as_shown

# 1. TWORZYMY OBIEKT APLIKACJI (Bez tego Railway rzucał błąd NameError)
app = FastAPI()

# 2. CORS – pozwala na zapytania ze Streamlita
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. STRONA GŁÓWNA - Automatyczne przekierowanie na Streamlita
@app.get("/")
def read_root():
    public_url = os.getenv("API_URL", "http://localhost:8505")
    
    # Jeśli uruchamiasz lokalnie (na komputerze)
    if "localhost" in public_url:
        return RedirectResponse(url="http://localhost:8505")
    
    # Jeśli aplikacja działa na Railway (honcho odpala Streamlit obok pod adresem IP pętli zwrotnej)
    return RedirectResponse(url="http://127.0.0.1:8505")

# 4. ENDPOINTY API

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
    return {"status": "ok"}

@app.get("/progress")
def get_progress(user_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM answers WHERE user_id = ?", (user_id,))
    answered = cur.fetchone()[0]
    conn.close()
    
    # POPRAWIONE: Zwracamy dokładnie strukturę {"answered": X, "total": Y},
    # której Streamlit w app.py szuka poprzez `.get('answered', 0)`
    return {"answered": answered, "total": 10}