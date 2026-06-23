"""
db_supabase.py — warstwa bazy danych (psycopg3 + Supabase Postgres)

Wymagane w Streamlit Secrets:
    SUPABASE_CONN_STRING = "postgresql://..."
    ADMIN_PASSWORD       = "twoje_haslo"
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import json

import psycopg
import streamlit as st

SCHEMA_SQL = """
create table if not exists users (
    id serial primary key,
    name text not null unique,
    created_at timestamptz default now()
);
alter table users add column if not exists completed_at timestamptz;

create table if not exists answers (
    id serial primary key,
    user_id integer references users(id) on delete cascade,
    left_id text,
    right_id text,
    choice text,
    strength text,
    created_at timestamptz default now()
);
alter table answers add column if not exists pair_idx integer;
alter table answers add column if not exists preference_value double precision;

create table if not exists schedules (
    id serial primary key,
    run_name  text not null,
    subgroup  text not null,
    direction text,
    year_tag  text,
    level     text,
    days_json      text,
    hours_json     text,
    cell_map_json  text,
    metrics_json   text,
    uploaded_at timestamptz default now(),
    unique(run_name, subgroup)
);
"""

@contextmanager
def get_conn():
    conn_str = st.secrets.get("SUPABASE_CONN_STRING")
    if not conn_str:
        raise RuntimeError("Brak SUPABASE_CONN_STRING w Streamlit Secrets.")
    conn = psycopg.connect(conn_str)
    try:
        yield conn
    finally:
        conn.close()

@st.cache_resource
def ensure_schema() -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
    return True

# ── użytkownicy ───────────────────────────────────────────────────────────────

def register_user(name: str) -> Optional[int]:
    name = name.strip()
    if not name:
        return None
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "select id from users where lower(name) = lower(%s) limit 1", (name,)
                )
                if cur.fetchone():
                    return None
                cur.execute(
                    "insert into users (name) values (%s) returning id", (name,)
                )
                row = cur.fetchone()
            conn.commit()
            return int(row[0]) if row else None
        except psycopg.errors.UniqueViolation:
            conn.rollback()
            return None
        except Exception:
            conn.rollback()
            raise

def save_completed_answers(user_id: int, answers: List[Dict[str, Any]]) -> None:
    if not answers:
        return
    rows = [
        (int(user_id), a.get("pair_idx"), a.get("left_id"), a.get("right_id"),
         a.get("choice"), a.get("strength"), a.get("preference_value"))
        for a in answers
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("delete from answers where user_id = %s", (int(user_id),))
            cur.executemany(
                "insert into answers "
                "(user_id,pair_idx,left_id,right_id,choice,strength,preference_value) "
                "values (%s,%s,%s,%s,%s,%s,%s)",
                rows,
            )
            cur.execute(
                "update users set completed_at = now() where id = %s", (int(user_id),)
            )
        conn.commit()

# ── admin ─────────────────────────────────────────────────────────────────────

def check_admin_password(pw: str) -> bool:
    expected = st.secrets.get("ADMIN_PASSWORD")
    return bool(pw) and bool(expected) and pw == expected

def fetch_all_answers_for_export() -> List[Dict[str, Any]]:
    sql = (
        "select u.id,u.name,u.created_at,u.completed_at,"
        "a.pair_idx,a.left_id,a.right_id,a.choice,a.strength,a.preference_value "
        "from answers a join users u on u.id=a.user_id "
        "where u.completed_at is not null order by u.id,a.pair_idx"
    )
    cols = ["user_id","user_name","created_at","completed_at",
            "pair_idx","left_id","right_id","choice","strength","preference_value"]
    with get_conn() as conn:
        rows = conn.execute(sql).fetchall()
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("created_at","completed_at"):
            if d.get(k):
                d[k] = d[k].isoformat()
        out.append(d)
    return out

# ── rozkłady (schedules) ──────────────────────────────────────────────────────

def clear_and_save_schedules(schedule_rows: List[Dict[str, Any]]) -> int:
    """Kasuje stare rozkłady i wstawia nowe. Zwraca liczbę wstawionych wierszy."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("truncate table schedules restart identity")
            cur.executemany(
                "insert into schedules "
                "(run_name,subgroup,direction,year_tag,level,"
                " days_json,hours_json,cell_map_json,metrics_json) "
                "values (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [
                    (
                        r["run_name"], r["subgroup"],
                        r.get("direction",""), r.get("year_tag",""), r.get("level",""),
                        json.dumps(r.get("days",[]),   ensure_ascii=False),
                        json.dumps(r.get("hours",[]),  ensure_ascii=False),
                        json.dumps(r.get("cell_map",{}), ensure_ascii=False),
                        json.dumps(r.get("metrics",{}),  ensure_ascii=False),
                    )
                    for r in schedule_rows
                ],
            )
        conn.commit()
    return len(schedule_rows)

def load_schedules_as_df():
    """Wczytuje rozkłady z bazy jako DataFrame gotowy do score_candidates_pairwise."""
    import pandas as pd
    sql = (
        "select run_name,subgroup,direction,year_tag,level,"
        "days_json,hours_json,cell_map_json,metrics_json from schedules"
    )
    with get_conn() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        run_name, subgroup, direction, year_tag, level, days_j, hours_j, cm_j, met_j = r
        rec = {
            "candidate_id": f"{run_name}::{subgroup}",
            "run":          run_name,
            "subgroup":     subgroup,
            "direction":    direction or "",
            "year":         year_tag  or "",
            "level":        level     or "",
            "days":         json.loads(days_j  or "[]"),
            "hours":        json.loads(hours_j or "[]"),
            "cell_map":     json.loads(cm_j    or "{}"),
        }
        rec.update(json.loads(met_j or "{}"))
        records.append(rec)
    return pd.DataFrame(records)

def schedules_count() -> int:
    with get_conn() as conn:
        row = conn.execute("select count(*) from schedules").fetchone()
    return row[0] if row else 0