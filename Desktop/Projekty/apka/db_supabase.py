"""
db_supabase.py — warstwa bazy danych (psycopg3 + Supabase Postgres)
"""
from __future__ import annotations
import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg
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
    conn = psycopg.connect(st.secrets["SUPABASE_CONN_STRING"])
    try:
        yield conn
    finally:
        conn.close()

@st.cache_resource
def ensure_schema() -> bool:
    with get_conn() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
    return True

def register_user(name: str) -> Optional[int]:
    name = name.strip()
    with get_conn() as conn:
        try:
            row = conn.execute(
                "insert into users (name) values (%s) returning id", (name,)
            ).fetchone()
            conn.commit()
            return row[0]

def save_completed_answers(user_id: int, answers: List[Dict[str, Any]]) -> None:
    if not answers:
        return
    with get_conn() as conn:
        conn.executemany(
            """insert into answers
               (user_id, pair_idx, left_id, right_id, choice, strength, preference_value)
               values (%s,%s,%s,%s,%s,%s,%s)""",
            [(
                user_id,
                a.get("pair_idx"),
                a.get("left_id"),
                a.get("right_id"),
                a.get("choice"),
                a.get("strength"),
                a.get("preference_value"),
            ) for a in answers],
        )
        conn.execute("update users set completed_at = now() where id = %s", (user_id,))
        conn.commit()

def check_admin_password(pw: str) -> bool:
    expected = st.secrets.get("ADMIN_PASSWORD")
    return bool(pw) and bool(expected) and pw == expected

def fetch_all_answers_for_export() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """select u.id as user_id, u.name as user_name,
                      u.created_at, u.completed_at,
                      a.pair_idx, a.left_id, a.right_id,
                      a.choice, a.strength, a.preference_value
               from answers a
               join users u on u.id = a.user_id
               order by u.id, a.pair_idx"""
        ).fetchall()
        cols = ["user_id","user_name","created_at","completed_at",
                "pair_idx","left_id","right_id","choice","strength","preference_value"]
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            for k in ("created_at","completed_at"):
                if d.get(k) is not None:
                    d[k] = d[k].isoformat()
            out.append(d)
    return out