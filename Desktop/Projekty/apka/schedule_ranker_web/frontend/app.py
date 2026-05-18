import os
import streamlit as st
import requests
import sys
import subprocess
import time
from pathlib import Path

# AUTOMATYCZNY START BACKENDU W TLE
@st.cache_resource
def start_backend():
    # Odpalamy uvicorn na porcie 8000 lokalnie
    cmd = "uvicorn backend.main:app --host 127.0.0.1 --port 8000"
    subprocess.Popen(cmd, shell=True)
    time.sleep(2)  # Dajemy mu 2 sekundy na rozruch

start_backend()

# Adres API - skoro FastAPI działa na tym samym serwerze, pytamy lokalnie przez localhost
API = "http://127.0.0.1:8000"

# Dodaj shared do ścieżek
sys.path.append(str(Path(__file__).parent.parent))
from shared.render import render_grid_html

st.set_page_config(page_title="Schedule Ranker – Zbieranie preferencji", layout="wide")
st.title("Porównywanie planów zajęć")

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "pair" not in st.session_state:
    st.session_state.pair = None

# Logowanie
if st.session_state.user_id is None:
    name = st.text_input("Twoje imię")
    if st.button("Start"):
        if name.strip():
            try:
                r = requests.post(f"{API}/user", params={"name": name})
                if r.status_code == 200:
                    st.session_state.user_id = r.json()["user_id"]
                    st.rerun()
                else:
                    st.error("Błąd tworzenia użytkownika")
            except Exception as e:
                st.error(f"Błąd połączenia: {e}")
        else:
            st.warning("Podaj imię")
else:
    st.success(f"Zalogowany jako użytkownik ID={st.session_state.user_id}")

    # Sprawdź postęp
    try:
        prog = requests.get(f"{API}/progress", params={"user_id": st.session_state.user_id})
        if prog.status_code == 200:
            count = prog.json()["count"]
            st.metric("Liczba udzielonych odpowiedzi", count)
    except:
        pass

    # Pobranie pary
    if st.session_state.pair is None:
        try:
            response = requests.get(f"{API}/pair", params={"user_id": st.session_state.user_id})
            if response.status_code == 200:
                st.session_state.pair = response.json()
            else:
                st.error(f"Błąd: {response.status_code} – {response.text}")
                st.stop()
        except Exception as e:
            st.error(f"Błąd połączenia: {e}")
            st.stop()

    pair = st.session_state.pair
    left = pair["left"]
    right = pair["right"]

    st.subheader("Porównaj dwa plany:")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Plan A (ID: `{left['id']}`)")
        html_left = render_grid_html(
            days=left["days"],
            hours=left["hours"],
            cell_map=left["cell_map"],
            title=left.get("profile", "Plan")
        )
        st.markdown(html_left, unsafe_allow_html=True)

    with col2:
        st.markdown(f"### Plan B (ID: `{right['id']}`)")
        html_right = render_grid_html(
            days=right["days"],
            hours=right["hours"],
            cell_map=right["cell_map"],
            title=right.get("profile", "Plan")
        )
        st.markdown(html_right, unsafe_allow_html=True)

    choice = st.radio("Który plan jest lepszy?", ["left", "right", "skip"], horizontal=True)
    strength = st.radio("Siła preferencji", ["slight", "strong"], horizontal=True, disabled=(choice == "skip"))

    if st.button("Zapisz odpowiedź"):
        payload = {
            "user_id": st.session_state.user_id,
            "left_id": left["id"],
            "right_id": right["id"],
            "choice": choice,
            "strength": strength if choice != "skip" else "skip"
        }
        try:
            resp = requests.post(f"{API}/answer", json=payload)
            if resp.status_code == 200:
                st.success("Zapisano!")
                st.session_state.pair = None
                st.rerun()
            else:
                st.error(f"Błąd zapisu: {resp.text}")
        except Exception as e:
            st.error(f"Błąd: {e}")

    if st.button("Pomiń tę parę (weź nową)"):
        st.session_state.pair = None
        st.rerun()