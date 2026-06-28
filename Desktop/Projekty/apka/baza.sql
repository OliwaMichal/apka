-- ============================================================
-- SCHEMAT FINALNY — Preferencje Rozkładów Zajęć
-- Supabase / PostgreSQL
--
-- Uruchom w SQL Editor → New query → Run
-- Bezpieczne: używa IF EXISTS / IF NOT EXISTS wszędzie
-- ============================================================


-- ── 1. USUŃ STARE / NIEUŻYWANE TABELE ────────────────────────

drop table if exists timetable_entries  cascade;
drop table if exists timetables         cascade;
drop table if exists shown_pairs        cascade;
drop table if exists candidates         cascade;


-- ── 2. TABELA: users ─────────────────────────────────────────

create table if not exists users (
    id           serial primary key,
    name         text not null,
    created_at   timestamptz default now(),
    completed_at timestamptz
);

-- unikalność nazw (case-insensitive)
create unique index if not exists users_name_unique_idx
    on users (lower(name));


-- ── 3. TABELA: answers ───────────────────────────────────────

create table if not exists answers (
    id                serial primary key,
    user_id           integer references users(id) on delete cascade,
    pair_idx          integer,
    left_id           text,
    right_id          text,
    choice            text,             -- 'left' | 'right' | 'skip'
    strength          text,             -- 'strong' | 'slight' | 'skip'
    preference_value  double precision, -- 1.0 / 0.75 / 0.5 / 0.25 / 0.0
    created_at        timestamptz default now()
);


-- ── 4. TABELA: schedules ─────────────────────────────────────

create table if not exists schedules (
    id             serial primary key,
    run_name       text not null,
    subgroup       text not null,
    direction      text,        -- np. 'FSI', 'FSU', 'INF'
    year_tag       text,        -- np. '1', '2', '3'
    level          text,        -- 'LAB' | 'W' | 'C' | 'P' | ''
    days_json      text,        -- JSON array dni tygodnia
    hours_json     text,        -- JSON array godzin
    cell_map_json  text,        -- JSON siatki zajęć (dict day|hour → lista zajęć)
    metrics_json   text,        -- JSON z 23 metrykami (FEATURE_COLS z fet_ltr.py)
    uploaded_at    timestamptz default now(),
    unique(run_name, subgroup)
);