"""
app.py — główny plik aplikacji do uruchomienia na Streamlit Community Cloud.

Wymagane w .streamlit/secrets.toml (lokalnie) lub w Settings → Secrets (Cloud):

    SUPABASE_CONN_STRING = "postgresql://postgres.xxxxx:HASLO@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
    ADMIN_PASSWORD       = "wybierz-haslo-administratora"

Wymagane w repozytorium:
    - fet_ltr.py              (bez zmian — Twój istniejący plik)
    - db_supabase.py          (nowy moduł bazy danych)
    - data/last_generation.json  (wgrywasz ręcznie po każdej nowej generacji FET)
    - requirements.txt        (patrz na końcu tego pliku — komentarz)
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

from db_supabase import (
    ensure_schema,
    register_user,
    save_completed_answers,
    check_admin_password,
    fetch_all_answers_for_export,
)
from fet_ltr import (
    FEATURE_COLS,
    HAS_LGBM, HAS_XGB,
    load_json,
    build_synthetic_candidates,
    select_diverse_candidates,
    generate_initial_pairs,
    build_pairwise_dataset,
    fit_models,
    score_candidates_pairwise,
    build_real_candidates,
    aggregate_run_scores,
    build_final_score,
    score_label,
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

DATA_DIR = Path(__file__).resolve().parent / "data"
LAST_GEN = DATA_DIR / "last_generation.json"

_available_models = []
if HAS_LGBM: _available_models.append("LightGBM")
if HAS_XGB:  _available_models.append("XGBoost")
if not _available_models:
    _available_models = ["BradleyTerry"]

N_SYNTH_TOTAL    = 120
N_SYNTH_SELECTED = 24
INITIAL_PAIRS    = 20
MAX_TOTAL_PAIRS  = 30
RANDOM_SEED      = 42

# upewniamy się, że tabele istnieją (wykona się raz na cały czas życia procesu)
ensure_schema()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "user_id":        None,
    "user_name":      None,
    "started":        False,
    "synth_df":       None,
    "pairs":          [],
    "answers":        [],      # lista słowników — trzymana w pamięci do czasu ukończenia
    "pair_idx":       0,
    "saved_to_db":    False,   # flaga: czy odpowiedzi trafiły już do Supabase
    "trained_models": None,
    "real_df":        None,
    "ready_to_rank":  False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_metric_summary(row: pd.Series):
    st.dataframe(pd.DataFrame([{
        "🗓 Dni wolne":       int(row.get("dayoff_count", 0)),
        "⏱ Okienka 1-slot":  int(row.get("gaps1", 0)),
        "⏱ Okienka 2+":      int(row.get("gaps2p", 0)),
        "🌅 Średni start":    round(float(row.get("earliest_start_mean", 0)), 2),
        "🌆 Średni koniec":   round(float(row.get("latest_end_mean", 0)), 2),
        "🏫 Zmiana kampusu":  int(row.get("campus_switch_0", 0)),
        "📅 Dni z zajęciami": int(row.get("days_with_classes", 0)),
    }]), hide_index=True, use_container_width=True)


def render_card(row: pd.Series, label: str):
    st.markdown(f"### {label}")
    html = render_grid_html(
        row["days"], row["hours"],
        row["cell_map"], str(row["subgroup"]),
    )
    st.markdown(html, unsafe_allow_html=True)
    render_metric_summary(row)


def build_answer(pair_idx, left_row, right_row, choice: str) -> dict:
    pref_map     = {"left": 1.0, "right": 0.0, "skip": 0.5}
    strength_map = {"left": "strong", "right": "strong", "skip": "skip"}
    return {
        "pair_idx":         pair_idx,
        "left_id":          left_row["candidate_id"],
        "right_id":         right_row["candidate_id"],
        "choice":           choice,
        "strength":         strength_map[choice],
        "preference_value": pref_map[choice],
        "user_id":          st.session_state.user_id,
    }


def maybe_add_active_pair():
    """Active learning: dobiera nową parę po >=6 jednoznacznych odpowiedziach."""
    resolved = [a for a in st.session_state.answers if a["choice"] != "skip"]
    if len(resolved) < 6:
        return
    if len(st.session_state.pairs) >= MAX_TOTAL_PAIRS:
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
            model_tmp = list(models_tmp.values())[0]
            pair = select_active_pair(df, model_tmp, st.session_state.pairs, bayes_model=bayes)

        if pair and pair not in st.session_state.pairs:
            st.session_state.pairs.append(pair)
    except Exception:
        pass  # active learning opcjonalne — nie crashujemy


# ─────────────────────────────────────────────────────────────────────────────
# PANEL ADMINA (ukryty na dole strony)
# ─────────────────────────────────────────────────────────────────────────────

def render_admin_panel():
    st.divider()
    with st.expander("🔒 Panel administratora", expanded=False):
        pw = st.text_input("Hasło administratora", type="password", key="admin_pw_input")
        if st.button("📥 Pobierz odpowiedzi (JSON)", key="admin_download_btn"):
            if check_admin_password(pw):
                with st.spinner("Pobieram dane z bazy…"):
                    data = fetch_all_answers_for_export()
                if not data:
                    st.warning("Baza jest pusta — nikt jeszcze nie ukończył ankiety.")
                else:
                    payload = json.dumps(data, ensure_ascii=False, indent=2)
                    st.download_button(
                        label=f"⬇️ Pobierz JSON ({len(data)} odpowiedzi)",
                        data=payload,
                        file_name="answers_export.json",
                        mime="application/json",
                        key="admin_download_file",
                    )
                    st.success(f"Gotowe! Odpowiedzi od {len({r['user_id'] for r in data})} uczestników.")
            else:
                st.error("Nieprawidłowe hasło.")


# ─────────────────────────────────────────────────────────────────────────────
# EKRAN STARTOWY
# ─────────────────────────────────────────────────────────────────────────────

st.title("📋 Preferencje Rozkładów Zajęć")
st.caption(
    "Porównaj kilka planów zajęć, a system dopasuje najlepszą grupę "
    "laboratoryjną do Twoich preferencji."
)

if not st.session_state.started:
    st.markdown("### Witaj! Podaj swoje imię, aby rozpocząć.")
    st.info(
        "Twoje imię jest jednocześnie Twoim unikalnym identyfikatorem — "
        "wybierz coś charakterystycznego (np. imię + inicjał nazwiska)."
    )
    name = st.text_input("Imię / pseudonim", placeholder="np. Michał K.")

    if st.button("▶️ Rozpocznij", use_container_width=True):
        if not name.strip():
            st.warning("Podaj imię, aby kontynuować.")
            st.stop()

        user_id = register_user(name.strip())
        if user_id is None:
            st.error(
                f"Nazwa **{name.strip()}** jest już zajęta przez innego uczestnika. "
                "Wybierz inną nazwę (np. dodaj inicjał nazwiska lub cyfrę)."
            )
            st.stop()

        synth = build_synthetic_candidates(N_SYNTH_TOTAL, RANDOM_SEED)
        synth = select_diverse_candidates(synth, N_SYNTH_SELECTED, RANDOM_SEED)

        st.session_state.user_id   = user_id
        st.session_state.user_name = name.strip()
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
pairs    = st.session_state.pairs
synth_df = st.session_state.synth_df

if pair_idx < MAX_TOTAL_PAIRS and pair_idx < len(pairs):

    left_i, right_i = pairs[pair_idx]
    left  = synth_df.iloc[left_i]
    right = synth_df.iloc[right_i]

    st.markdown(f"## Porównanie {pair_idx + 1} / {MAX_TOTAL_PAIRS}")
    st.progress((pair_idx + 1) / MAX_TOTAL_PAIRS)

    c1, c2 = st.columns(2)
    with c1:
        render_card(left,  "📋 Plan A")
    with c2:
        render_card(right, "📋 Plan B")

    st.markdown("#### Co się różni między planami?")
    st.dataframe(
        explain_pair_difference(left, right).head(8),
        hide_index=True, use_container_width=True,
    )

    def _vote(choice: str, pref_value: float, strength: str):
        ans = {
            "pair_idx":         pair_idx,
            "left_id":          left["candidate_id"],
            "right_id":         right["candidate_id"],
            "choice":           choice,
            "strength":         strength,
            "preference_value": pref_value,
            "user_id":          st.session_state.user_id,
        }
        st.session_state.answers.append(ans)
        maybe_add_active_pair()
        st.session_state.pair_idx += 1

        # ZAPIS DO BAZY — dopiero gdy student ukończy wszystkie 30 porównań
        if st.session_state.pair_idx >= MAX_TOTAL_PAIRS and not st.session_state.saved_to_db:
            try:
                save_completed_answers(st.session_state.user_id, st.session_state.answers)
                st.session_state.saved_to_db = True
            except Exception as e:
                st.warning(f"Odpowiedzi zostały zapamiętane, ale wystąpił błąd zapisu do bazy: {e}")

        st.rerun()

    st.markdown("#### Który plan bardziej Ci odpowiada?")
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        if st.button("⬅️⬅️\nZdecydowanie\nPlan A", use_container_width=True, key="vote_ll"):
            _vote("left",  1.0,  "strong")
    with b2:
        if st.button("⬅️\nRaczej\nPlan A",          use_container_width=True, key="vote_l"):
            _vote("left",  0.75, "slight")
    with b3:
        if st.button("⚖️\nBez\nróżnicy",             use_container_width=True, key="vote_eq"):
            _vote("skip",  0.5,  "skip")
    with b4:
        if st.button("➡️\nRaczej\nPlan B",            use_container_width=True, key="vote_r"):
            _vote("right", 0.25, "slight")
    with b5:
        if st.button("➡️➡️\nZdecydowanie\nPlan B",   use_container_width=True, key="vote_rr"):
            _vote("right", 0.0,  "strong")

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FAZA RANKINGU
# ─────────────────────────────────────────────────────────────────────────────

st.success(f"✅ Dziękujemy, {st.session_state.user_name}! Zakończono wszystkie {MAX_TOTAL_PAIRS} porównań.")

if not st.session_state.ready_to_rank:
    if st.button("🎯 Pokaż mój ranking grup laboratoryjnych", use_container_width=True):
        st.session_state.ready_to_rank = True
        st.rerun()
    render_admin_panel()
    st.stop()

# ── trening modelu (tylko raz — cachowane w session_state) ────────────────────
if st.session_state.trained_models is None:
    with st.spinner("Uczę modelu Twoich preferencji…"):
        X, y, w = build_pairwise_dataset(synth_df, st.session_state.answers)

        if X is None or len(X) < 6:
            st.warning("Za mało jednoznacznych wyborów do treningu modelu. Wróć i oceń jeszcze kilka par.")
            st.session_state.ready_to_rank = False
            st.stop()

        trained = fit_models(X, y, sample_weight=w, model_types=_available_models)
        if not trained:
            st.error("Nie udało się wytrenować modelu.")
            st.stop()

        st.session_state.trained_models = trained

models = st.session_state.trained_models

# ── wczytanie prawdziwych rozkładów (tylko raz) ───────────────────────────────
if st.session_state.real_df is None:
    with st.spinner("Wczytuję rozkłady zajęć…"):
        last = load_json(LAST_GEN)
        if not last or not last.get("root"):
            st.error("Brak pliku data/last_generation.json lub klucza 'root' — wgraj plik do repozytorium.")
            st.stop()

        try:
            real_df = build_real_candidates(Path(last["root"]), LAST_GEN)
        except Exception as e:
            st.error(f"Błąd wczytywania rozkładów: {e}")
            st.stop()

        for model_name, model in models.items():
            real_df[f"{model_name}_score"] = score_candidates_pairwise(real_df, model)

        score_cols = [c for c in real_df.columns if c.endswith("_score")]
        real_df["ensemble_score"] = real_df[score_cols].mean(axis=1)
        st.session_state.real_df = real_df

real_df = st.session_state.real_df

# ─────────────────────────────────────────────────────────────────────────────
# RANKING — tymczasowo wyłączony (last_generation.json niegotowy)
# ─────────────────────────────────────────────────────────────────────────────
st.success("✅ Twoje preferencje zostały zapisane. Dziękujemy za udział w badaniu!")
render_admin_panel()
st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RANKING GLOBALNY RUNÓW  (odkomentuj gdy last_generation.json będzie gotowy)
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("## 🏆 Ranking runów (globalnie)")

run_df = aggregate_run_scores(real_df, "ensemble_score").copy()
if not run_df.empty:
    run_df["final_score"] = build_final_score(run_df, "best_subgroup_score")
    run_df["ocena"]       = run_df["final_score"].apply(score_label)
    run_df = run_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    display_cols = [c for c in [
        "run", "final_score", "ocena",
        "best_subgroup", "best_subgroup_score",
        "top3_mean_score", "median_score", "subgroups_count",
    ] if c in run_df.columns]
    st.dataframe(run_df[display_cols], use_container_width=True, hide_index=True)

    best_run = run_df.iloc[0]
    st.success(
        f"🥇 Najlepszy run: **{best_run['run']}** "
        f"| Wynik: {best_run['final_score']:.1f} "
        f"| Ocena: {best_run['ocena']}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# RANKING PODGRUP LABORATORYJNYCH
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("## 🎯 Twój ranking grup laboratoryjnych")
st.caption("Wybierz run, kierunek i rocznik — zobaczysz, które grupy najlepiej pasują do Twoich preferencji.")

if "run" not in real_df.columns:
    st.error("Brak kolumny 'run' w danych.")
    st.stop()

available_runs = sorted(real_df["run"].unique().tolist())
sel_run = st.selectbox("Wybierz run", options=available_runs, key="sl_run")

run_subdf = real_df[real_df["run"] == sel_run].copy()
run_subdf = run_subdf.sort_values("ensemble_score", ascending=False).reset_index(drop=True)

if "direction" not in run_subdf.columns or run_subdf["direction"].isna().all():
    run_subdf["direction"] = run_subdf["subgroup"].apply(
        lambda sg: "".join(c for c in str(sg).split("-")[0] if c.isalpha()) or "?"
    )
if "year" not in run_subdf.columns or run_subdf["year"].isna().all():
    run_subdf["year"] = run_subdf["subgroup"].apply(
        lambda sg: str(sg).split("-")[0] + "-W1" if "-" in str(sg) else str(sg)
    )

all_dirs = sorted(run_subdf["direction"].dropna().unique())
sel_dir  = st.selectbox("Kierunek", options=["(wszystkie)"] + all_dirs, key="sl_dir")
filtered = run_subdf if sel_dir == "(wszystkie)" else run_subdf[run_subdf["direction"] == sel_dir]

all_years = sorted(filtered["year"].dropna().unique())
sel_year  = st.selectbox("Rocznik", options=["(wszystkie)"] + all_years, key="sl_year")
filtered  = filtered if sel_year == "(wszystkie)" else filtered[filtered["year"] == sel_year]

show_lab = st.checkbox("Tylko grupy laboratoryjne (-L)", value=True, key="sl_lab")
if show_lab:
    if "level" in filtered.columns:
        filtered = filtered[filtered["level"] == "LAB"]
    else:
        filtered = filtered[filtered["subgroup"].str.contains(r"-L\d+", regex=True, na=False)]

if filtered.empty:
    st.info("Brak grup spełniających kryteria. Zmień filtr kierunku, rocznika lub odznacz LAB.")
else:
    show_metrics = [c for c in [
        "gaps1", "gaps2p", "dayoff_count", "campus_switch_0",
        "earliest_start_mean", "latest_end_mean",
        "daily_load_variance", "friday_penalty", "long_streak_days",
    ] if c in filtered.columns]

    result_df = filtered[["subgroup", "ensemble_score"] + show_metrics].copy()
    result_df = result_df.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "miejsce", range(1, len(result_df) + 1))

    st.dataframe(result_df, use_container_width=True, hide_index=True)

    best = result_df.iloc[0]
    st.success(
        f"🥇 Najlepsza grupa dla Ciebie: **{best['subgroup']}** "
        f"| Dopasowanie: `{best['ensemble_score']:.3f}`"
    )

    with st.expander("📊 Szczegóły scoringu dla wszystkich modeli"):
        detail_df = filtered[
            ["subgroup"] + [c for c in real_df.columns if c.endswith("_score")]
        ].sort_values("ensemble_score", ascending=False).reset_index(drop=True)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

best_lab = run_subdf[
    run_subdf["level"] == "LAB" if "level" in run_subdf.columns
    else run_subdf["subgroup"].str.contains(r"-L\d+", regex=True, na=False)
]
if not best_lab.empty:
    b = best_lab.iloc[0]
    st.info(
        f"ℹ️ Najlepsza grupa LAB w całym runie `{sel_run}` (bez filtrów): "
        f"**{b['subgroup']}** | Dopasowanie: `{b['ensemble_score']:.3f}`"
    )

render_admin_panel()