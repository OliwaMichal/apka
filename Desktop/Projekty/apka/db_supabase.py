"""
db_supabase.py — warstwa bazy danych dla Streamlit + Supabase Postgres
Wersja poprawiona pod psycopg3.

Wymagane w Streamlit Secrets:

SUPABASE_CONN_STRING = "postgresql://postgres.xxxxx:HASLO@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
ADMIN_PASSWORD = "twoje_haslo_admina"
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
create table if not exists users (
    id serial primary key,
    name text not null,
    created_at timestamptz default now()
);

alter table users
    add column if not exists completed_at timestamptz;

create table if not exists answers (
    id serial primary key,
    user_id integer references users(id) on delete cascade,
    left_id text,
    right_id text,
    choice text,
    strength text,
    created_at timestamptz default now()
);

alter table answers
    add column if not exists pair_idx integer;

alter table answers
    add column if not exists preference_value double precision;
"""


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    """
    Otwiera połączenie do Supabase Postgres.

    Uwaga:
    Connection string musi być ustawiony w Streamlit Secrets jako:
    SUPABASE_CONN_STRING = "postgresql://..."
    """
    conn_string = st.secrets.get("SUPABASE_CONN_STRING")

    if not conn_string:
        raise RuntimeError(
            "Brak SUPABASE_CONN_STRING w Streamlit Secrets. "
            "Dodaj go w .streamlit/secrets.toml lokalnie albo w Settings → Secrets na Streamlit Cloud."
        )

    conn = psycopg.connect(conn_string)

    try:
        yield conn
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def ensure_schema() -> bool:
    """
    Tworzy tabele i brakujące kolumny, jeśli ich jeszcze nie ma.
    Wykonuje się raz na czas życia procesu Streamlit.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)

            # Dodajemy unikalność nazwy użytkownika, ale ostrożnie.
            # Jeśli masz już duplikaty w tabeli users, indeks może się nie utworzyć.
            try:
                cur.execute(
                    """
                    create unique index if not exists users_name_unique_idx
                    on users (lower(name));
                    """
                )
            except Exception:
                # Jeżeli są duplikaty nazw w starej bazie, nie crashujemy aplikacji.
                # register_user i tak robi ręczne sprawdzenie przed insertem.
                conn.rollback()

                with conn.cursor() as cur2:
                    cur2.execute(SCHEMA_SQL)

            conn.commit()

    return True


def register_user(name: str) -> Optional[int]:
    """
    Rejestruje użytkownika.

    Zwraca:
    - id użytkownika, jeśli udało się zarejestrować
    - None, jeśli nazwa jest już zajęta
    """
    name = name.strip()

    if not name:
        return None

    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                # Najpierw sprawdzamy ręcznie, bo starsza tabela mogła nie mieć UNIQUE.
                cur.execute(
                    "select id from users where lower(name) = lower(%s) limit 1",
                    (name,),
                )
                existing = cur.fetchone()

                if existing is not None:
                    conn.rollback()
                    return None

                cur.execute(
                    "insert into users (name) values (%s) returning id",
                    (name,),
                )
                row = cur.fetchone()

            conn.commit()

            if row is None:
                return None

            return int(row[0])

        except psycopg.errors.UniqueViolation:
            conn.rollback()
            return None

        except Exception:
            conn.rollback()
            raise


def save_completed_answers(user_id: int, answers: List[Dict[str, Any]]) -> None:
    """
    Zapisuje wszystkie odpowiedzi użytkownika po ukończeniu ankiety.

    WAŻNE:
    W psycopg3 nie używamy conn.executemany(),
    tylko cur.executemany().
    To naprawia błąd:
    'Connection' object has no attribute 'executemany'
    """
    if not answers:
        return

    user_id = int(user_id)

    rows = [
        (
            user_id,
            _safe_int(a.get("pair_idx")),
            _safe_str(a.get("left_id")),
            _safe_str(a.get("right_id")),
            _safe_str(a.get("choice")),
            _safe_str(a.get("strength")),
            _safe_float(a.get("preference_value")),
        )
        for a in answers
    ]

    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                # Opcjonalnie: jeśli użytkownik zapisuje ponownie po crashu/rerunie,
                # czyścimy jego stare odpowiedzi, żeby nie dublować rekordów.
                cur.execute(
                    "delete from answers where user_id = %s",
                    (user_id,),
                )

                cur.executemany(
                    """
                    insert into answers
                        (user_id, pair_idx, left_id, right_id, choice, strength, preference_value)
                    values
                        (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )

                cur.execute(
                    "update users set completed_at = now() where id = %s",
                    (user_id,),
                )

            conn.commit()

        except Exception:
            conn.rollback()
            raise


def check_admin_password(pw: str) -> bool:
    """
    Sprawdza hasło administratora z Streamlit Secrets.
    """
    expected = st.secrets.get("ADMIN_PASSWORD")
    return bool(pw) and bool(expected) and pw == expected


def fetch_all_answers_for_export() -> List[Dict[str, Any]]:
    """
    Pobiera wszystkie odpowiedzi ukończonych ankiet do eksportu JSON.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    u.id as user_id,
                    u.name as user_name,
                    u.created_at,
                    u.completed_at,
                    a.pair_idx,
                    a.left_id,
                    a.right_id,
                    a.choice,
                    a.strength,
                    a.preference_value
                from answers a
                join users u on u.id = a.user_id
                where u.completed_at is not null
                order by u.id, a.pair_idx
                """
            )

            rows = cur.fetchall()

    cols = [
        "user_id",
        "user_name",
        "created_at",
        "completed_at",
        "pair_idx",
        "left_id",
        "right_id",
        "choice",
        "strength",
        "preference_value",
    ]

    out: List[Dict[str, Any]] = []

    for r in rows:
        d = dict(zip(cols, r))

        for k in ("created_at", "completed_at"):
            if d.get(k) is not None:
                d[k] = d[k].isoformat()

        out.append(d)

    return out