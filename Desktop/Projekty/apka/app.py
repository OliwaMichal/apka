"""
app.py — Preferencje Rozkładów Zajęć
Streamlit Community Cloud + Supabase Postgres

Secrets (Settings → Secrets na Streamlit Cloud):
    SUPABASE_CONN_STRING = "postgresql://..."
    ADMIN_PASSWORD       = "twoje_haslo"
"""

import io
import json
import zipfile
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from db_supabase import (
    ensure_schema,
    register_user,
    save_completed_answers,
    check_admin_password,
    fetch_all_answers_for_export,
    clear_and_save_schedules,
    load_schedules_as_df,
    schedules_count,
)
from fet_ltr import (
    FEATURE_COLS,
    HAS_LGBM, HAS_XGB,
    build_synthetic_candidates,
    select_diverse_candidates,
    generate_initial_pairs,
    build_pairwise_dataset,
    fit_models,
    score_candidates_pairwise,
    render_grid_html,
    explain_pair_difference,
    BayesianPreferenceModel,
    select_active_pair_ensemble,
    select_active_pair,
)

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Preferencje Rozkładów Zajęć")

_available_models = []
if HAS_LGBM: _available_models.append("LightGBM")
if HAS_XGB:  _available_models.append("XGBoost")
if not _available_models: _available_models = ["BradleyTerry"]

N_SYNTH_TOTAL    = 120
N_SYNTH_SELECTED = 24
INITIAL_PAIRS    = 20
MAX_TOTAL_PAIRS  = 30
RANDOM_SEED      = 42

ensure_schema()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

for k, v in {
    "user_id": None, "user_name": None, "started": False,
    "synth_df": None, "pairs": [], "answers": [],
    "pair_idx": 0, "saved_to_db": False,
    "trained_models": None, "real_df": None, "ready_to_rank": False,
    "imported_answers": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# PARSER CSV → METRYKI (do zapisu w tabeli schedules)
# ─────────────────────────────────────────────────────────────────────────────

def _hour_to_idx(hour_str: str, hours_list: List[str]) -> int:
    try:
        return hours_list.index(hour_str)
    except ValueError:
        return -1

def _compute_metrics(sg_slots: List[Tuple[str, str, str, str]],
                     days_list: List[str],
                     hours_list: List[str]) -> Dict[str, float]:
    """
    sg_slots: lista krotek (day, hour, subject, room)
    Oblicza wszystkie metryki z FEATURE_COLS.
    """
    n_hours = len(hours_list)
    n_days  = len(days_list)
    morning_thresh = int(n_hours * 0.35)
    late_thresh    = int(n_hours * 0.70)
    friday_idx     = next((i for i,d in enumerate(days_list)
                           if "pi" in d.lower() or "fri" in d.lower()), n_days - 1)
    monday_idx     = next((i for i,d in enumerate(days_list)
                           if "pon" in d.lower() or "mon" in d.lower()), 0)

    by_day: Dict[str, List[int]] = defaultdict(list)
    by_day_rooms: Dict[str, List[str]] = defaultdict(list)
    by_day_subjects: Dict[str, List[str]] = defaultdict(list)

    for day, hour, subj, room in sg_slots:
        idx = _hour_to_idx(hour, hours_list)
        if idx >= 0:
            by_day[day].append(idx)
            by_day_rooms[day].append(room or "")
            by_day_subjects[day].append(subj or "")

    days_with = [d for d in days_list if by_day.get(d)]
    days_without = [d for d in days_list if not by_day.get(d)]

    dayoff_count    = len(days_without)
    days_with_cnt   = len(days_with)
    total_activities = sum(len(v) for v in by_day.values())

    start_idxs, end_idxs, spans = [], [], []
    gaps1_total = gaps2p_total = 0
    single_class_days = long_streak_days = 0
    morning_cnt = late_cnt = 0
    multi_campus_days = campus_rush_days = 0
    mixed_type_days = 0
    friday_classes = friday_late = 0
    monday_free = 1
    campus_sw0 = campus_sw1 = 0
    load_per_day = []

    for d in days_list:
        slots = sorted(set(by_day.get(d, [])))
        if not slots:
            load_per_day.append(0)
            continue

        load_per_day.append(len(slots))
        s, e = slots[0], slots[-1]
        start_idxs.append(s)
        end_idxs.append(e)
        spans.append(e - s)

        # gaps
        for a, b in zip(slots, slots[1:]):
            gap = b - a - 1
            if gap == 1: gaps1_total  += 1
            elif gap > 1: gaps2p_total += 1

        if len(slots) == 1:
            single_class_days += 1

        # long streaks (4+ consecutive)
        streak = max_streak = 1
        for a, b in zip(slots, slots[1:]):
            if b == a + 1: streak += 1
            else:
                max_streak = max(max_streak, streak)
                streak = 1
        max_streak = max(max_streak, streak)
        if max_streak >= 4:
            long_streak_days += 1

        # morning / late
        morning_cnt += sum(1 for x in slots if x <= morning_thresh)
        late_cnt    += sum(1 for x in slots if x >= late_thresh)

        # friday
        if d == days_list[friday_idx]:
            friday_classes += len(slots)
            friday_late    += sum(1 for x in slots if x >= late_thresh)

        # monday
        if d == days_list[monday_idx] and slots:
            monday_free = 0

        # campus / room switches
        rooms = [by_day_rooms[d][i] for i, slot in enumerate(sorted(by_day[d]))
                 if slot in slots]
        unique_rooms = set(r for r in rooms if r)
        if len(unique_rooms) > 1:
            multi_campus_days += 1
        for ra, rb in zip(rooms, rooms[1:]):
            if ra and rb and ra != rb:
                campus_sw0 += 1
        sorted_pairs = list(zip(sorted(set(by_day[d])), rooms))
        for i in range(len(sorted_pairs) - 1):
            if sorted_pairs[i+1][0] - sorted_pairs[i][0] == 2:
                if sorted_pairs[i][1] != sorted_pairs[i+1][1]:
                    campus_sw1 += 1

        # rush days (switch + no gap)
        if campus_sw0 > 0 and not gaps1_total:
            campus_rush_days += 1

        # mixed type
        subjs = by_day_subjects.get(d, [])
        types = set()
        for s2 in subjs:
            s2u = s2.upper()
            if " - W" in s2u or "WYKŁAD" in s2u:   types.add("W")
            elif " - L" in s2u or "LAB" in s2u:    types.add("L")
            elif " - C" in s2u or "ĆWICZ" in s2u:  types.add("C")
            elif " - P" in s2u or "PROJ" in s2u:   types.add("P")
        if len(types) > 1:
            mixed_type_days += 1

    # odd/even imbalance
    odd_days  = sum(1 for d in days_with if days_list.index(d) % 2 == 0)
    even_days = days_with_cnt - odd_days
    odd_even_imbalance = abs(odd_days - even_days) / max(days_with_cnt, 1)

    # lab_days
    lab_days = sum(
        1 for d in days_list
        if any("lab" in s.lower() or " - l" in s.lower()
               for s in by_day_subjects.get(d, []))
    )

    # variance of daily load
    daily_load_variance = float(np.var(load_per_day)) if load_per_day else 0.0

    return {
        "campus_switch_0":    campus_sw0,
        "campus_switch_1":    campus_sw1,
        "gaps1":              gaps1_total,
        "gaps2p":             gaps2p_total,
        "single_class_days":  single_class_days,
        "long_streak_days":   long_streak_days,
        "dayoff_count":       dayoff_count,
        "days_with_classes":  days_with_cnt,
        "total_activities":   total_activities,
        "earliest_start_mean": float(np.mean(start_idxs)) if start_idxs else 0.0,
        "latest_end_mean":     float(np.mean(end_idxs))   if end_idxs   else 0.0,
        "daily_span_mean":     float(np.mean(spans))       if spans       else 0.0,
        "morning_classes_count": morning_cnt,
        "late_classes_count":    late_cnt,
        "lab_days":              lab_days,
        "odd_even_imbalance":    odd_even_imbalance,
        "mixed_type_days":       mixed_type_days,
        "friday_penalty":        friday_classes,
        "monday_bonus":          monday_free,
        "multi_campus_days":     multi_campus_days,
        "friday_late_classes":   friday_late,
        "campus_rush_days":      campus_rush_days,
        "daily_load_variance":   daily_load_variance,
    }


def parse_csv_to_schedule_rows(df: pd.DataFrame, run_name: str) -> List[Dict[str, Any]]:
    """
    Parsuje CSV z FET (kolumny: Day, Hour, Students Sets, Subject, Teacher/Teachers, Room/Rooms)
    → lista słowników gotowych do zapisu w tabeli `schedules`.
    """
    # normalizacja nazw kolumn
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("day",):                            col_map[c] = "Day"
        elif cl in ("hour",):                          col_map[c] = "Hour"
        elif cl in ("students sets","students","subgroup","students set"): col_map[c] = "Students"
        elif cl in ("subject",):                       col_map[c] = "Subject"
        elif cl in ("teacher","teachers"):             col_map[c] = "Teacher"
        elif cl in ("room","rooms"):                   col_map[c] = "Room"
        elif cl in ("building","buildings"):           col_map[c] = "Building"
    df = df.rename(columns=col_map)
    for req in ("Day","Hour","Students","Subject"):
        if req not in df.columns:
            raise ValueError(f"Brak kolumny '{req}' w CSV. Dostępne: {list(df.columns)}")

    df["Teacher"] = df.get("Teacher", pd.Series([""] * len(df)))
    df["Room"]    = df.get("Room",    pd.Series([""] * len(df)))

    # listy dni i godzin (zachowujemy oryginalną kolejność)
    days_order  = list(dict.fromkeys(df["Day"].dropna().str.strip()))
    hours_order = list(dict.fromkeys(df["Hour"].dropna().str.strip()))

    # zbieramy sloty per podgrupa
    by_sg: Dict[str, List[Tuple]] = defaultdict(list)
    cell_map_sg: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        day  = str(row.get("Day","")).strip()
        hour = str(row.get("Hour","")).strip()
        subj = str(row.get("Subject","")).strip()
        room = str(row.get("Room","")).strip()
        tchr = str(row.get("Teacher","")).strip()
        sgs  = str(row.get("Students","")).strip()

        for sg in re.split(r"[,;+]", sgs):
            sg = sg.strip()
            if not sg: continue
            by_sg[sg].append((day, hour, subj, room))
            key = f"{day}|{hour}"
            cell_map_sg[sg][key].append({
                "subject":  subj,
                "teachers": [tchr] if tchr else [],
                "room":     room,
                "tags":     [],
            })

    # wyciągamy poziom (LAB/W/C/P) i kierunek/rok z nazwy podgrupy
    def infer_level(sg: str) -> str:
        sg_u = sg.upper()
        if re.search(r"-L\d+", sg_u): return "LAB"
        if re.search(r"-W\d+", sg_u): return "W"
        if re.search(r"-C\d+", sg_u): return "C"
        if re.search(r"-P\d+", sg_u): return "P"
        return ""

    def infer_direction(sg: str) -> str:
        m = re.match(r"([A-Za-z]+)", sg)
        return m.group(1).upper() if m else ""

    def infer_year(sg: str) -> str:
        m = re.search(r"(\d+)", sg)
        return m.group(1) if m else ""

    rows_out = []
    for sg, slots in by_sg.items():
        metrics = _compute_metrics(slots, days_order, hours_order)
        rows_out.append({
            "run_name":  run_name,
            "subgroup":  sg,
            "direction": infer_direction(sg),
            "year_tag":  infer_year(sg),
            "level":     infer_level(sg),
            "days":      days_order,
            "hours":     hours_order,
            "cell_map":  dict(cell_map_sg[sg]),
            "metrics":   metrics,
        })
    return rows_out


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS RENDEROWANIA
# ─────────────────────────────────────────────────────────────────────────────

def render_metric_summary(row):
    st.dataframe(pd.DataFrame([{
        "🗓 Dni wolne":       int(row.get("dayoff_count", 0)),
        "⏱ Okienka 1-slot":  int(row.get("gaps1", 0)),
        "⏱ Okienka 2+":      int(row.get("gaps2p", 0)),
        "🌅 Średni start":   round(float(row.get("earliest_start_mean", 0)), 1),
        "🌆 Średni koniec":  round(float(row.get("latest_end_mean", 0)), 1),
        "🏫 Zmiana kampusu": int(row.get("campus_switch_0", 0)),
        "📅 Dni z zajęciami":int(row.get("days_with_classes", 0)),
    }]), hide_index=True, use_container_width=True)

def render_card(row, label: str):
    st.markdown(f"### {label}")
    days = row.get("days") if isinstance(row.get("days"), list) else []
    hours = row.get("hours") if isinstance(row.get("hours"), list) else []
    cell_map = row.get("cell_map") if isinstance(row.get("cell_map"), dict) else {}
    if days and hours and cell_map:
        html = render_grid_html(days, hours, cell_map, str(row.get("subgroup","")))
        st.markdown(html, unsafe_allow_html=True)
    render_metric_summary(row)

def maybe_add_active_pair():
    resolved = [a for a in st.session_state.answers if a["choice"] != "skip"]
    if len(resolved) < 6 or len(st.session_state.pairs) >= MAX_TOTAL_PAIRS:
        return
    df = st.session_state.synth_df
    X, y, w = build_pairwise_dataset(df, resolved)
    if X is None or len(X) < 8:
        return
    try:
        models_tmp = fit_models(X, y, sample_weight=w, model_types=_available_models)
        bayes = BayesianPreferenceModel(n_features=X.shape[1])
        bayes.fit(X, y)
        if len(models_tmp) >= 2:
            pair = select_active_pair_ensemble(df, models_tmp, st.session_state.pairs, bayes_model=bayes)
        else:
            pair = select_active_pair(df, list(models_tmp.values())[0],
                                      st.session_state.pairs, bayes_model=bayes)
        if pair and pair not in st.session_state.pairs:
            st.session_state.pairs.append(pair)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# PANEL ADMINA
# ─────────────────────────────────────────────────────────────────────────────

def render_admin_panel():
    st.divider()
    with st.expander("🔒 Panel administratora", expanded=False):
        pw = st.text_input("Hasło administratora", type="password", key="admin_pw_input")

        st.markdown("#### 📥 Pobierz odpowiedzi uczestników")
        if st.button("Pobierz odpowiedzi (JSON)", key="admin_dl_btn"):
            if check_admin_password(pw):
                with st.spinner("Pobieram…"):
                    data = fetch_all_answers_for_export()
                if not data:
                    st.warning("Baza jest pusta — nikt jeszcze nie ukończył ankiety.")
                else:
                    payload = json.dumps(data, ensure_ascii=False, indent=2)
                    st.download_button(
                        f"⬇️ Pobierz JSON ({len(data)} odpowiedzi)",
                        data=payload, file_name="answers_export.json",
                        mime="application/json", key="admin_dl_file",
                    )
                    st.success(f"Odpowiedzi od {len({r['user_id'] for r in data})} uczestników.")
            else:
                st.error("Nieprawidłowe hasło.")

        st.markdown("#### 📤 Wgraj rozkłady zajęć (CSV z FET)")
        st.caption(
            "Wgraj jeden lub więcej plików CSV z FET (np. `timetable_for_students.csv`). "
            "Każdy plik to osobny run. Stare rozkłady zostaną zastąpione."
        )
        uploaded_csvs = st.file_uploader(
            "Pliki CSV", type=["csv"], accept_multiple_files=True, key="admin_csv_upload"
        )
        if uploaded_csvs:
            if not check_admin_password(pw):
                st.error("Złe hasło — nie można zapisać rozkładów.")
            else:
                if st.button("💾 Wgraj i zastąp rozkłady", key="admin_save_btn"):
                    all_rows = []
                    errors = []
                    for f in uploaded_csvs:
                        run_name = Path(f.name).stem
                        try:
                            df_csv = pd.read_csv(f, sep=None, engine="python",
                                                 encoding="utf-8-sig")
                            rows = parse_csv_to_schedule_rows(df_csv, run_name)
                            all_rows.extend(rows)
                            st.write(f"✅ {run_name}: {len(rows)} podgrup")
                        except Exception as e:
                            errors.append(f"{f.name}: {e}")

                    if errors:
                        for err in errors:
                            st.error(err)
                    if all_rows:
                        with st.spinner(f"Zapisuję {len(all_rows)} rozkładów do bazy…"):
                            n = clear_and_save_schedules(all_rows)
                        st.success(f"✅ Zapisano {n} rozkładów. Stare dane zastąpione.")
                        st.session_state.real_df = None  # wymusz przeładowanie

# ─────────────────────────────────────────────────────────────────────────────
# EKRAN STARTOWY
# ─────────────────────────────────────────────────────────────────────────────

st.title("📋 Preferencje Rozkładów Zajęć")
st.caption("Porównaj kilka planów zajęć, a system dopasuje najlepszą grupę do Twoich preferencji.")

if not st.session_state.started:
    st.markdown("### Witaj! Podaj swoje imię, aby rozpocząć.")
    st.info("Twoje imię jest Twoim unikalnym identyfikatorem — wybierz coś charakterystycznego (np. imię + inicjał nazwiska).")
    name_input = st.text_input("Imię / pseudonim", placeholder="np. Michał K.")

    st.markdown("---")
    st.markdown("### 📂 Masz już wcześniejsze odpowiedzi?")
    st.caption("Wgraj plik JSON pobrany po poprzednim wypełnieniu — pominiesz 30 porównań i od razu zobaczysz ranking.")
    uploaded_json = st.file_uploader("Wgraj plik JSON z odpowiedziami", type=["json"], key="import_json")

    if uploaded_json is not None:
        try:
            data_imported = json.load(uploaded_json)
            if isinstance(data_imported, list) and len(data_imported) > 0 and isinstance(data_imported[0], dict):
                if st.button("▶️ Załaduj odpowiedzi i przejdź do rankingu", use_container_width=True):
                    if not name_input.strip():
                        st.warning("Podaj imię przed załadowaniem odpowiedzi.")
                    else:
                        user_id = register_user(name_input.strip())
                        if user_id is None:
                            st.error(f"Nazwa **{name_input.strip()}** jest już zajęta.")
                        else:
                            st.session_state.user_id        = user_id
                            st.session_state.user_name      = name_input.strip()
                            st.session_state.answers        = data_imported
                            st.session_state.pair_idx       = MAX_TOTAL_PAIRS
                            st.session_state.started        = True
                            st.session_state.saved_to_db    = True
                            st.session_state.ready_to_rank  = True
                            st.session_state.imported_answers = True
                            # od razu zapisz do bazy
                            try:
                                save_completed_answers(user_id, data_imported)
                            except Exception:
                                pass
                            st.rerun()
            else:
                st.error("Nieprawidłowy format pliku — oczekiwano listy obiektów JSON.")
        except Exception as e:
            st.error(f"Błąd wczytywania pliku: {e}")

    st.markdown("---")
    if st.button("▶️ Rozpocznij od nowa (30 porównań)", use_container_width=True):
        if not name_input.strip():
            st.warning("Podaj imię, aby kontynuować.")
            st.stop()
        user_id = register_user(name_input.strip())
        if user_id is None:
            st.error(f"Nazwa **{name_input.strip()}** jest już zajęta. Wybierz inną.")
            st.stop()
        synth = build_synthetic_candidates(N_SYNTH_TOTAL, RANDOM_SEED)
        synth = select_diverse_candidates(synth, N_SYNTH_SELECTED, RANDOM_SEED)
        st.session_state.user_id   = user_id
        st.session_state.user_name = name_input.strip()
        st.session_state.synth_df  = synth
        st.session_state.pairs     = generate_initial_pairs(synth, INITIAL_PAIRS)
        st.session_state.started   = True
        st.rerun()

    render_admin_panel()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FAZA PORÓWNAŃ
# ─────────────────────────────────────────────────────────────────────────────

pair_idx = st.session_state.pair_idx
pairs    = st.session_state.pairs or []
synth_df = st.session_state.synth_df

if not st.session_state.ready_to_rank and pair_idx < MAX_TOTAL_PAIRS and synth_df is not None and pair_idx < len(pairs):
    left_i, right_i = pairs[pair_idx]
    left  = synth_df.iloc[left_i]
    right = synth_df.iloc[right_i]

    st.markdown(f"## Porównanie {pair_idx + 1} / {MAX_TOTAL_PAIRS}")
    st.progress((pair_idx + 1) / MAX_TOTAL_PAIRS)

    c1, c2 = st.columns(2)
    with c1: render_card(left,  "📋 Plan A")
    with c2: render_card(right, "📋 Plan B")

    diff_df = explain_pair_difference(left, right)
    if diff_df is not None and not diff_df.empty:
        st.markdown("#### Co się różni między planami?")
        st.dataframe(diff_df.head(8), hide_index=True, use_container_width=True)

    st.markdown("#### Który plan bardziej Ci odpowiada?")
    b1, b2, b3, b4, b5 = st.columns(5)
    voted_choice = voted_pref = voted_str = None

    with b1:
        if st.button("⬅️⬅️ Zdecydowanie Plan A", use_container_width=True, key="v_ll"):
            voted_choice, voted_pref, voted_str = "left",  1.0,  "strong"
    with b2:
        if st.button("⬅️ Raczej Plan A", use_container_width=True, key="v_l"):
            voted_choice, voted_pref, voted_str = "left",  0.75, "slight"
    with b3:
        if st.button("⚖️ Bez różnicy", use_container_width=True, key="v_eq"):
            voted_choice, voted_pref, voted_str = "skip",  0.5,  "skip"
    with b4:
        if st.button("➡️ Raczej Plan B", use_container_width=True, key="v_r"):
            voted_choice, voted_pref, voted_str = "right", 0.25, "slight"
    with b5:
        if st.button("➡️➡️ Zdecydowanie Plan B", use_container_width=True, key="v_rr"):
            voted_choice, voted_pref, voted_str = "right", 0.0,  "strong"

    if voted_choice is not None:
        st.session_state.answers.append({
            "pair_idx": pair_idx,
            "left_id":  left["candidate_id"],
            "right_id": right["candidate_id"],
            "choice":   voted_choice,
            "strength": voted_str,
            "preference_value": voted_pref,
            "user_id":  st.session_state.user_id,
        })
        maybe_add_active_pair()
        st.session_state.pair_idx += 1

        if st.session_state.pair_idx >= MAX_TOTAL_PAIRS and not st.session_state.saved_to_db:
            with st.spinner("Zapisuję odpowiedzi…"):
                try:
                    save_completed_answers(st.session_state.user_id, st.session_state.answers)
                    st.session_state.saved_to_db = True
                except Exception as e:
                    st.error(f"Błąd zapisu: {e}")
                    st.stop()
        st.rerun()

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PO UKOŃCZENIU PORÓWNAŃ — podziękowanie + pobranie swoich odpowiedzi
# ─────────────────────────────────────────────────────────────────────────────

st.success(f"✅ Dziękujemy, {st.session_state.user_name}! Zakończono {MAX_TOTAL_PAIRS} porównań.")

# przycisk pobrania własnych odpowiedzi
if st.session_state.answers:
    own_payload = json.dumps(st.session_state.answers, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Pobierz swoje odpowiedzi (JSON)",
        data=own_payload,
        file_name=f"moje_odpowiedzi_{st.session_state.user_name.replace(' ','_')}.json",
        mime="application/json",
        key="own_dl_btn",
    )
    st.caption("Zachowaj ten plik — przy kolejnej wizycie możesz go wgrać i od razu zobaczyć ranking bez ponownego wypełniania.")

if not st.session_state.ready_to_rank:
    if st.button("🎯 Pokaż mój ranking grup laboratoryjnych", use_container_width=True):
        st.session_state.ready_to_rank = True
        st.rerun()
    render_admin_panel()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# TRENING MODELU
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.trained_models is None:
    synth_df_for_training = st.session_state.synth_df
    if synth_df_for_training is None:
        synth_df_for_training = build_synthetic_candidates(N_SYNTH_TOTAL, RANDOM_SEED)
        synth_df_for_training = select_diverse_candidates(synth_df_for_training, N_SYNTH_SELECTED, RANDOM_SEED)
        st.session_state.synth_df = synth_df_for_training

    with st.spinner("Uczę modelu Twoich preferencji…"):
        X, y, w = build_pairwise_dataset(synth_df_for_training, st.session_state.answers)
        if X is None or len(X) < 6:
            st.warning("Za mało jednoznacznych wyborów. Wróć i oceń więcej par.")
            st.session_state.ready_to_rank = False
            st.stop()
        trained = fit_models(X, y, sample_weight=w, model_types=_available_models)
        if not trained:
            st.error("Nie udało się wytrenować modelu.")
            st.stop()
        st.session_state.trained_models = trained

models = st.session_state.trained_models

# ─────────────────────────────────────────────────────────────────────────────
# WCZYTANIE ROZKŁADÓW Z BAZY
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.real_df is None:
    cnt = schedules_count()
    if cnt == 0:
        st.warning("⚠️ Baza nie zawiera żadnych rozkładów. Administrator musi wgrać pliki CSV.")
        render_admin_panel()
        st.stop()

    with st.spinner("Wczytuję rozkłady zajęć z bazy…"):
        real_df = load_schedules_as_df()
        if real_df.empty:
            st.error("Nie udało się wczytać rozkładów.")
            st.stop()

        for model_name, model in models.items():
            try:
                real_df[f"{model_name}_score"] = score_candidates_pairwise(real_df, model)
            except Exception:
                real_df[f"{model_name}_score"] = 0.0

        score_cols = [c for c in real_df.columns if c.endswith("_score")]
        real_df["ensemble_score"] = real_df[score_cols].mean(axis=1) if score_cols else 0.0
        st.session_state.real_df = real_df

real_df = st.session_state.real_df

# ─────────────────────────────────────────────────────────────────────────────
# RANKING GRUP LABORATORYJNYCH
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("## 🎯 Twój ranking grup laboratoryjnych")

if "run" not in real_df.columns:
    st.error("Brak kolumny 'run' w danych.")
    st.stop()

available_runs = sorted(real_df["run"].unique().tolist())
sel_run = st.selectbox("Wybierz run", options=available_runs, key="sl_run")
run_subdf = real_df[real_df["run"] == sel_run].copy()

# kierunek i rocznik
if "direction" not in run_subdf.columns or run_subdf["direction"].isna().all():
    run_subdf["direction"] = run_subdf["subgroup"].apply(
        lambda sg: re.match(r"([A-Za-z]+)", str(sg)).group(1).upper()
        if re.match(r"([A-Za-z]+)", str(sg)) else "?"
    )
if "year" not in run_subdf.columns or run_subdf["year"].isna().all():
    run_subdf["year"] = run_subdf["subgroup"].apply(
        lambda sg: re.search(r"(\d+)", str(sg)).group(1) if re.search(r"(\d+)", str(sg)) else ""
    )

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    dirs = sorted(run_subdf["direction"].dropna().unique())
    sel_dir = st.selectbox("Kierunek", ["(wszystkie)"] + dirs, key="sl_dir")
with col_f2:
    filtered = run_subdf if sel_dir == "(wszystkie)" else run_subdf[run_subdf["direction"] == sel_dir]
    years = sorted(filtered["year"].dropna().unique())
    sel_year = st.selectbox("Rocznik", ["(wszystkie)"] + years, key="sl_year")
with col_f3:
    show_lab = st.checkbox("Tylko grupy LAB", value=True, key="sl_lab")

filtered = run_subdf if sel_dir == "(wszystkie)" else run_subdf[run_subdf["direction"] == sel_dir]
filtered = filtered if sel_year == "(wszystkie)" else filtered[filtered["year"] == sel_year]

if show_lab:
    mask_lab = (
        filtered["level"] == "LAB"
        if "level" in filtered.columns
        else filtered["subgroup"].str.contains(r"-L\d+", regex=True, na=False)
    )
    filtered = filtered[mask_lab]

if filtered.empty:
    st.info("Brak grup spełniających kryteria — zmień filtry.")
else:
    metric_cols = [c for c in [
        "gaps1","gaps2p","dayoff_count","campus_switch_0",
        "earliest_start_mean","latest_end_mean","daily_load_variance",
        "friday_penalty","long_streak_days",
    ] if c in filtered.columns]

    result_df = filtered[["subgroup","ensemble_score"] + metric_cols].copy()
    result_df = result_df.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "miejsce", range(1, len(result_df)+1))
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    best = result_df.iloc[0]
    st.success(f"🥇 Najlepsza grupa dla Ciebie: **{best['subgroup']}** | Dopasowanie: `{best['ensemble_score']:.3f}`")

render_admin_panel()