"""
db_supabase.py
───────────────
Lekka warstwa dostępu do bazy Supabase (Postgres) — zastępuje lokalne pliki
users.jsonl / feedback/*.jsonl z wcześniejszej wersji aplikacji.

Wymagane w .streamlit/secrets.toml (lokalnie) albo w Settings → Secrets
(na Streamlit Community Cloud):

    SUPABASE_CONN_STRING = "postgresql://postgres.xxxxx:HASLO@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
    ADMIN_PASSWORD = "wybierz-haslo-administratora"

Założenia:
- Nazwa uczestnika (`users.name`) jest rezerwowana NATYCHMIAST przy starcie
  (żeby dwóch studentów nie mogło wybrać tej samej nazwy równolegle) —
  ale to tylko pusty wiersz z nazwą, bez żadnych odpowiedzi.
- Właściwe odpowiedzi (`answers`) trafiają do bazy DOPIERO, gdy student
  ukończy wszystkie 30 porównań — pojedynczym zapisem zbiorczym.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg as psycopg2
import psycopg.rows
import streamlit as st


SCHEMA_SQL = """
create table if not exists users (
    id serial primary key,
    name text not null unique,
    created_at timestamptz default now(),
    completed_at timestamptz
);

create table if not exists answers (
    id serial primary key,
    user_id integer references users(id) on delete cascade,
    pair_idx integer,
    left_id text,
    right_id text,
    choice text,
    strength text,
    preference_value double precision,
    created_at timestamptz default now()
);
"""


@contextmanager
def get_conn():
    conn = psycopg2.connect(st.secrets["SUPABASE_CONN_STRING"], row_factory=psycopg.rows.dict_row)
    try:
        yield conn
    finally:
        conn.close()


@st.cache_resource
def ensure_schema() -> bool:
    """Tworzy tabele, jeśli jeszcze nie istnieją. Wykonywane raz na proces (cache_resource)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        conn.commit()
    return True


def register_user(name: str) -> Optional[int]:
    """
    Próbuje zarezerwować unikalną nazwę uczestnika.
    Zwraca user_id (int) jeśli się udało, albo None, jeśli nazwa jest już zajęta.
    """
    name = name.strip()
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "insert into users (name) values (%s) returning id",
                (name,),
            )
            user_id = cur.fetchone()[0]
            conn.commit()
            return user_id
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            return None


def save_completed_answers(user_id: int, answers: List[Dict[str, Any]]) -> None:
    """Zapisuje WSZYSTKIE odpowiedzi naraz, jednym zapytaniem — wywoływać dopiero po 30/30."""
    if not answers:
        return
    with get_conn() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            insert into answers
                (user_id, pair_idx, left_id, right_id, choice, strength, preference_value)
            values %s
            """,
            [
                (
                    user_id,
                    a.get("pair_idx"),
                    a.get("left_id"),
                    a.get("right_id"),
                    a.get("choice"),
                    a.get("strength"),
                    a.get("preference_value"),
                )
                for a in answers
            ],
        )
        cur.execute("update users set completed_at = now() where id = %s", (user_id,))
        conn.commit()


def check_admin_password(pw: str) -> bool:
    expected = st.secrets.get("ADMIN_PASSWORD")
    return bool(pw) and bool(expected) and pw == expected


def fetch_all_answers_for_export() -> List[Dict[str, Any]]:
    """Pełny zrzut odpowiedzi (tylko ukończonych uczestników) — do ponownego treningu modelu."""
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            select u.id as user_id, u.name as user_name,
                   u.created_at, u.completed_at,
                   a.pair_idx, a.left_id, a.right_id,
                   a.choice, a.strength, a.preference_value
            from answers a
            join users u on u.id = a.user_id
            order by u.id, a.pair_idx
            """
        )
        rows = cur.fetchall()

    out = []
    for r in rows:
        d = dict(r)
        for k in ("created_at", "completed_at"):
            if d.get(k) is not None:
                d[k] = d[k].isoformat()
        out.append(d)
    return out
