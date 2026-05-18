from pathlib import Path
import json
import random
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# Używamy prawdziwych rankerów LambdaMART
try:
    from lightgbm import LGBMRanker
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBRanker
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    optim = None
    HAS_TORCH = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False


FEATURE_COLS = [
    "campus_switch_0",
    "campus_switch_1",
    "gaps1",
    "gaps2p",
    "single_class_days",
    "long_streak_days",
    "dayoff_count",
    "days_with_classes",
    "total_activities",
    "earliest_start_mean",
    "latest_end_mean",
    "daily_span_mean",
    "morning_classes_count",
    "late_classes_count",
    "lab_days",
    "odd_even_imbalance",
    "mixed_type_days",
    "friday_penalty",
    "monday_bonus",
    "multi_campus_days",
    # NOWE FEATURE
    "friday_late_classes",
    "campus_rush_days",
    "daily_load_variance",
]

DIRECTIONS = {
    "campus_switch_0": "cost",
    "campus_switch_1": "cost",
    "gaps1": "cost",
    "gaps2p": "cost",
    "single_class_days": "cost",
    "long_streak_days": "cost",
    "dayoff_count": "benefit",
    "days_with_classes": "cost",
    "total_activities": "cost",
    "earliest_start_mean": "benefit",
    "latest_end_mean": "cost",
    "daily_span_mean": "cost",
    "morning_classes_count": "cost",
    "late_classes_count": "cost",
    "lab_days": "benefit",
    "odd_even_imbalance": "cost",
    "mixed_type_days": "cost",
    "friday_penalty": "cost",
    "monday_bonus": "benefit",
    "multi_campus_days": "cost",
    "friday_late_classes": "cost",
    "campus_rush_days": "cost",
    "daily_load_variance": "cost",
}

PAIR_GROUPS = {
    "TIME": ["earliest_start_mean", "latest_end_mean", "morning_classes_count", "late_classes_count"],
    "GAPS": ["gaps1", "gaps2p", "single_class_days", "daily_span_mean"],
    "CAMPUS": ["campus_switch_0", "campus_switch_1", "multi_campus_days", "campus_rush_days"],
    "FREE": ["dayoff_count", "days_with_classes", "monday_bonus", "friday_penalty", "friday_late_classes"],
    "LOAD": ["long_streak_days", "total_activities", "daily_load_variance"],
    "TYPE": ["lab_days", "mixed_type_days", "odd_even_imbalance"],
}

GROUP_IMPORTANCE = {
    "TIME": 1.0,
    "GAPS": 1.4,
    "CAMPUS": 1.2,
    "FREE": 1.5,
    "LOAD": 1.1,
    "TYPE": 0.8,
}

FEATURE_SCALER = None


# -------------------- generic utils
def build_ranking_dataset(synth_df, answers):
    """
    Transformuje odpowiedzi użytkownika na format akceptowany przez LGBMRanker/XGBRanker.
    Każde porównanie staje się osobną grupą (qid), zawierającą 2 elementy (lewy i prawy rozkład).
    """
    X_list = []
    y_list = []
    qid_list = []
    
    for idx, ans in enumerate(answers):
        choice = ans.get("choice")
        strength = ans.get("strength")
        
        # Mapowanie siły wyboru na stopnie relewancji (relevance grades)
        if choice == "skip":
            val_left, val_right = 1, 1
        elif choice == "left" and strength == "strong":
            val_left, val_right = 3, 0
        elif choice == "left" and strength == "slight":
            val_left, val_right = 2, 1
        elif choice == "right" and strength == "slight":
            val_left, val_right = 1, 2
        elif choice == "right" and strength == "strong":
            val_left, val_right = 0, 3
        else:
            continue
            
        left_row = synth_df[synth_df["candidate_id"] == ans["left_id"]]
        right_row = synth_df[synth_df["candidate_id"] == ans["right_id"]]
        
        if left_row.empty or right_row.empty:
            continue
            
        # Dodajemy surowe cechy lewego kandydata
        X_list.append(left_row[FEATURE_COLS].iloc[0])
        y_list.append(val_left)
        qid_list.append(idx)
        
        # Dodajemy surowe cechy prawego kandydata
        X_list.append(right_row[FEATURE_COLS].iloc[0])
        y_list.append(val_right)
        qid_list.append(idx)
        
    if not X_list:
        return None, None, None
        
    X = pd.DataFrame(X_list).reset_index(drop=True)
    y = np.array(y_list)
    qid = np.array(qid_list)
    
    return X, y, qid


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None


def save_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_text(node: Optional[ET.Element], paths: List[str], default: str = "") -> str:
    if node is None:
        return default
    for p in paths:
        try:
            val = node.findtext(p)
        except Exception:
            val = None
        if val is not None:
            s = str(val).strip()
            if s:
                return s
    return default


def attr_any(node: Optional[ET.Element], keys: List[str], default: str = "") -> str:
    if node is None:
        return default
    for k in keys:
        v = node.attrib.get(k)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return default


def name_of(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    n = first_text(node, ["Name"], "")
    if n:
        return n
    return (node.attrib.get("name") or node.attrib.get("Name") or "").strip()


def building_to_campus(building_id: str) -> str:
    b = str(building_id or "").strip().upper()
    return "C1" if b in {"A", "B", "C", "D", "E", "F"} else "C2"


def room_to_campus_from_room_name(room_name: str) -> str:
    r = str(room_name or "").strip().upper()
    if not r:
        return ""
    lead = r[0]
    return "C1" if lead in {"A", "B", "C", "D", "E", "F"} else "C2"


def is_lab_subgroup(name: str) -> bool:
    n = str(name or "").upper()
    return "SUBGROUP" in n or "AUTOMATIC SUBGROUP" in n


def pick_type_tag(tags: List[str], subject: str = "") -> str:
    tags_u = [str(x).strip().upper() for x in (tags or [])]

    mapping = {
        "WYKŁAD": "WYKŁAD",
        "LECTURE": "WYKŁAD",
        "ĆWICZENIA": "ĆWICZENIA",
        "CWICZENIA": "ĆWICZENIA",
        "EXERCISE": "ĆWICZENIA",
        "LABORATORIUM": "LABORATORIUM",
        "LAB": "LABORATORIUM",
        "PROJEKT": "PROJEKT",
        "PROJECT": "PROJEKT",
        "SEMINARIUM": "SEMINARIUM",
    }

    for raw, out in mapping.items():
        if raw in tags_u:
            return out

    s = str(subject or "").upper()
    if " - W" in s:
        return "WYKŁAD"
    if " - C" in s or " - Ć" in s:
        return "ĆWICZENIA"
    if " - L" in s:
        return "LABORATORIUM"
    if " - P" in s:
        return "PROJEKT"
    if "SEMINARIUM" in s:
        return "SEMINARIUM"

    return ""


def transform_features_for_learning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURE_COLS:
        if DIRECTIONS.get(c, "cost") == "cost":
            out[c] = -out[c].astype(float)
        else:
            out[c] = out[c].astype(float)
    return out


def fit_feature_scaler(df_candidates: pd.DataFrame):
    global FEATURE_SCALER
    X = transform_features_for_learning(df_candidates[FEATURE_COLS]).astype(float)
    scaler = MinMaxScaler()
    scaler.fit(X)
    FEATURE_SCALER = scaler
    return scaler


def transform_with_global_scaler(df_candidates: pd.DataFrame) -> np.ndarray:
    global FEATURE_SCALER

    X = transform_features_for_learning(df_candidates[FEATURE_COLS]).astype(float)

    if FEATURE_SCALER is None:
        FEATURE_SCALER = MinMaxScaler()
        FEATURE_SCALER.fit(X)

    return FEATURE_SCALER.transform(X)


def build_shap_importance(model, X, feature_cols=None, max_samples: int = 200) -> pd.DataFrame:
    if not HAS_SHAP:
        raise RuntimeError("SHAP nie jest dostępny.")

    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if X is None or len(X) == 0:
        raise RuntimeError("Brak danych do SHAP.")

    sample_X = X[: min(len(X), max_samples)]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)

    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values[1])
    else:
        shap_arr = np.array(shap_values)

    mean_abs = np.mean(np.abs(shap_arr), axis=0)

    return pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_pair_difference(row_left, row_right):
    rows = []

    for feat in FEATURE_COLS:
        lv = float(row_left.get(feat, 0))
        rv = float(row_right.get(feat, 0))
        diff = lv - rv

        direction = DIRECTIONS.get(feat, "cost")

        if direction == "cost":
            if lv < rv:
                preference = "left"
            elif rv < lv:
                preference = "right"
            else:
                preference = "equal"
        else:
            if lv > rv:
                preference = "left"
            elif rv > lv:
                preference = "right"
            else:
                preference = "equal"

        rows.append({
            "feature": feat,
            "left": lv,
            "right": rv,
            "abs_diff": abs(diff),
            "preferred_side": preference,
        })

    return pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)


def build_final_score(df, score_col):
    scaler = MinMaxScaler()
    vals = scaler.fit_transform(df[[score_col]]).reshape(-1)

    base_score = pd.Series(
        np.round(vals * 100, 2),
        index=df.index,
    )

    # Kary za niestabilność i słabe podgrupy
    if "score_std" in df.columns and "weak_ratio" in df.columns:
        penalty_std = np.tanh(df["score_std"] / 10.0) * 10  # max kara 10 pkt
        penalty_weak = df["weak_ratio"] * 15  # max kara 15 pkt
        final = base_score - penalty_std - penalty_weak
        final = final.clip(lower=0)
        return final
    return base_score


def score_label(v):
    if v >= 85:
        return "Excellent"
    if v >= 70:
        return "Very good"
    if v >= 55:
        return "Good"
    if v >= 40:
        return "Average"
    return "Weak"


def pair_difficulty_score(x1, x2):
    diff = np.abs(x1 - x2)
    total = diff.mean()
    concentrated = diff.max() / (diff.sum() + 1e-9)
    return concentrated - 0.35 * total


# -------------------- synthetic base problem
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
HOURS = [
    "07:00-08:30",
    "08:45-10:15",
    "10:30-12:00",
    "12:15-13:45",
    "14:00-15:30",
    "15:45-17:15",
    "17:30-19:00",
]

ROOMS_C1 = ["A101", "B204", "C301", "D110"]
ROOMS_C2 = ["G101", "H204", "J301", "K110"]

SUBJECTS_BY_TYPE = {
    "WYKŁAD": ["Analiza - W", "AI - W", "Systemy - W", "Bazy - W"],
    "ĆWICZENIA": ["Analiza - C", "AI - C", "Programowanie - C", "Sieci - C"],
    "LABORATORIUM": ["Analiza - L", "AI Lab - L", "Programowanie - L", "Bazy - L", "Robotyka - L"],
    "PROJEKT": ["Projekt AI - P", "Projekt BD - P", "Projekt Sys - P"],
}

TEACHER_POOL = ["T001", "T002", "T003", "T004", "T005", "T006"]


def build_synthetic_activity_base(random_state: int = 42) -> List[dict]:
    rng = random.Random(random_state)

    spec = [
        ("LABORATORIUM", "ALL"),
        ("LABORATORIUM", "ALL"),
        ("LABORATORIUM", "ALL"),
        ("LABORATORIUM", "ODD"),
        ("LABORATORIUM", "EVEN"),
        ("ĆWICZENIA", "ALL"),
        ("ĆWICZENIA", "ALL"),
        ("PROJEKT", "ALL"),
        ("PROJEKT", "ALL"),
        ("WYKŁAD", "ALL"),
        ("LABORATORIUM", "ALL"),
        ("ĆWICZENIA", "ALL"),
    ]

    base = []
    for i, (type_name, week_tag) in enumerate(spec, start=1):
        subject = rng.choice(SUBJECTS_BY_TYPE[type_name])
        teacher = rng.choice(TEACHER_POOL)
        base.append({
            "base_id": f"BASE_{i:03d}",
            "type_name": type_name,
            "subject": subject,
            "teacher": teacher,
            "week_tag": week_tag,
        })
    return base


def _make_activity_from_base(base_item: dict, idx: int, d: str, sidx: int, rng: random.Random, campus: str) -> dict:
    room = rng.choice(ROOMS_C1 if campus == "C1" else ROOMS_C2)
    type_name = base_item["type_name"]
    week_tag = base_item["week_tag"]

    tags = [type_name]
    if type_name == "LABORATORIUM":
        tags.append("LAB")
    if type_name == "WYKŁAD":
        tags.append("LECTURE")
    if type_name == "ĆWICZENIA":
        tags.append("EXERCISE")
    if type_name == "PROJEKT":
        tags.append("PROJECT")
    if week_tag in {"ODD", "EVEN"}:
        tags.append(week_tag)

    return {
        "activity_id": f"SYN_{idx}_{base_item['base_id']}_{d}_{sidx}",
        "room": room,
        "subject": base_item["subject"],
        "comments": "",
        "tags": tags,
        "teachers": [base_item["teacher"]],
    }


def generate_synthetic_candidate(idx: int, base_activities: List[dict], rng: random.Random) -> dict:
    profile = rng.choice([
        "compact",
        "free_day",
        "early",
        "late",
        "many_gaps",
        "campus_mixed",
        "balanced",
        "long_streak",
        "alternating_campus",
        "odd_even_split",
        "good_monday_bad_friday",
        "mixed_types_layout",
    ])

    cell_map = defaultdict(list)

    if profile == "free_day":
        preferred_days = ["Mon", "Tue", "Wed", "Thu"]
    else:
        preferred_days = DAYS[:]

    n_items = len(base_activities)
    candidate_slots = []

    if profile == "compact":
        starts = [1, 2, 2, 3, 3]
        for d in preferred_days:
            start = rng.choice(starts)
            for x in range(2):
                if start + x < len(HOURS):
                    candidate_slots.append((d, start + x))
    elif profile == "free_day":
        for d in preferred_days:
            start = rng.choice([1, 2, 3])
            for x in range(3):
                if start + x < len(HOURS):
                    candidate_slots.append((d, start + x))
    elif profile == "early":
        for d in preferred_days:
            for s in [0, 1, 2]:
                candidate_slots.append((d, s))
    elif profile == "late":
        for d in preferred_days:
            for s in [3, 4, 5, 6]:
                if s < len(HOURS):
                    candidate_slots.append((d, s))
    elif profile == "many_gaps":
        for d in DAYS:
            for s in [0, 2, 4, 6]:
                if s < len(HOURS):
                    candidate_slots.append((d, s))
    elif profile == "campus_mixed":
        for d in DAYS:
            for s in [1, 2, 3]:
                candidate_slots.append((d, s))
    elif profile == "alternating_campus":
        for d in DAYS:
            for s in [1, 2, 3]:
                candidate_slots.append((d, s))
    elif profile == "long_streak":
        for d in DAYS:
            for s in [0, 1, 2, 3, 4]:
                candidate_slots.append((d, s))
    elif profile == "good_monday_bad_friday":
        candidate_slots.extend([("Mon", 2), ("Mon", 3)])
        candidate_slots.extend([("Fri", 0), ("Fri", 2), ("Fri", 4), ("Fri", 6)])
        for d in ["Tue", "Wed", "Thu"]:
            for s in [1, 2, 3]:
                candidate_slots.append((d, s))
    elif profile == "odd_even_split":
        for d in DAYS:
            for s in [1, 2, 4]:
                candidate_slots.append((d, s))
    elif profile == "mixed_types_layout":
        for d in DAYS:
            for s in [1, 3, 4]:
                candidate_slots.append((d, s))
    else:
        for d in DAYS:
            for s in [1, 2, 3]:
                candidate_slots.append((d, s))

    if len(candidate_slots) < n_items:
        for d in DAYS:
            for s in range(len(HOURS)):
                candidate_slots.append((d, s))

    candidate_slots = list(dict.fromkeys(candidate_slots))
    rng.shuffle(candidate_slots)
    chosen_slots = candidate_slots[:n_items]

    # DODANIE MUTACJI (NOISE) - 15% szans na losowy slot, ale bez duplikatów
    mutation_prob = 0.15
    for idx_slot in range(len(chosen_slots)):
        if rng.random() < mutation_prob:
            # Losuj nowy slot, dopóki nie będzie unikalny w chosen_slots
            new_day = rng.choice(DAYS)
            new_hour_idx = rng.randrange(len(HOURS))
            new_slot = (new_day, new_hour_idx)
            # Sprawdź czy nowy slot nie jest już użyty w innych indeksach
            while new_slot in chosen_slots[:idx_slot] + chosen_slots[idx_slot+1:]:
                new_day = rng.choice(DAYS)
                new_hour_idx = rng.randrange(len(HOURS))
                new_slot = (new_day, new_hour_idx)
            chosen_slots[idx_slot] = new_slot

    # Używamy zbioru do śledzenia już wykorzystanych slotów (unikamy duplikatów w cell_map)
    used_slots = set()

    for k, base_item in enumerate(base_activities):
        d, sidx = chosen_slots[k]
        key = (d, HOURS[sidx])

        # Jeśli ten slot został już wykorzystany przez poprzednią aktywność – pomijamy
        if key in used_slots:
            continue

        used_slots.add(key)

        if profile in {"alternating_campus", "campus_mixed"}:
            campus = "C1" if (k % 2 == 0) else "C2"
        else:
            campus = rng.choice(["C1", "C1", "C1", "C2"])

        act = _make_activity_from_base(base_item, idx, d, sidx, rng, campus)

        # NADPISUJEMY (zamiast append) – w slocie będzie tylko jedna aktywność
        cell_map[key] = [act]

    return {
        "candidate_id": f"SYNTH::{idx:03d}",
        "run": "SYNTH",
        "subgroup": f"Synthetic Lab {idx:03d}",
        "profile": profile,
        "days": DAYS[:],
        "hours": HOURS[:],
        "cell_map": dict(cell_map),
        "base_activity_count": len(base_activities),
    }


# -------------------- metrics
def compute_per_subgroup_metrics(
    table: Dict[Tuple[str, str], List[dict]],
    days: List[str],
    hours: List[str],
    room_to_campus: Dict[str, str],
):
    hour_idx = {h: i for i, h in enumerate(hours)}

    gaps1 = 0
    gaps2p = 0
    single_class_days = 0
    campus_switch_0 = 0
    campus_switch_1 = 0
    long_streak_days = 0
    dayoff_count = 0
    total_days_with_classes = 0
    total_activities = 0
    earliest_start_sum = 0
    latest_end_sum = 0
    daily_span_sum = 0
    morning_classes_count = 0
    late_classes_count = 0
    lab_days = 0
    odd_even_imbalance = 0
    mixed_type_days = 0
    friday_penalty = 0
    monday_bonus = 0
    multi_campus_days = 0
    # NOWE METRYKI
    friday_late_classes = 0
    campus_rush_days = 0
    daily_loads = []

    for d in days:
        occ = []
        room_by_i = {}
        acts_by_i = defaultdict(list)

        for (dd, hh), acts in (table or {}).items():
            if dd != d:
                continue
            if hh not in hour_idx:
                continue
            i = hour_idx[hh]

            if acts:
                occ.append(i)
                total_activities += len(acts)
                for a in acts:
                    acts_by_i[i].append(a)
                    room = str(a.get("room") or "").strip()
                    if room:
                        room_by_i[i] = room

        if not occ:
            dayoff_count += 1
            continue

        total_days_with_classes += 1
        daily_loads.append(len(occ))
        occ = sorted(set(occ))

        if len(occ) == 1:
            single_class_days += 1

        earliest_start_sum += occ[0]
        latest_end_sum += occ[-1]
        daily_span_sum += (occ[-1] - occ[0] + 1)

        if occ[0] <= 1:
            morning_classes_count += 1
        if occ[-1] >= max(0, len(hours) - 2):
            late_classes_count += 1

        first_i, last_i = occ[0], occ[-1]
        occ_set = set(occ)
        j = first_i
        while j <= last_i:
            if j in occ_set:
                j += 1
                continue
            k = j
            while k <= last_i and k not in occ_set:
                k += 1
            gap_len = k - j
            if gap_len == 1:
                gaps1 += 1
            elif gap_len >= 2:
                gaps2p += 1
            j = k

        campuses_today = set()
        for a, b in zip(occ, occ[1:]):
            gap_slots = b - a - 1
            r1 = room_by_i.get(a, "")
            r2 = room_by_i.get(b, "")

            c1 = room_to_campus.get(r1, "") or room_to_campus_from_room_name(r1)
            c2 = room_to_campus.get(r2, "") or room_to_campus_from_room_name(r2)

            if c1:
                campuses_today.add(c1)
            if c2:
                campuses_today.add(c2)

            if c1 and c2 and c1 != c2:
                if gap_slots == 0:
                    campus_switch_0 += 1
                    campus_rush_days += 1  # natychmiastowa zmiana kampusu
                elif gap_slots == 1:
                    campus_switch_1 += 1

        if len(campuses_today) >= 2:
            multi_campus_days += 1

        max_streak = 1
        streak = 1
        for a, b in zip(occ, occ[1:]):
            if b == a + 1:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        if max_streak >= 5:
            long_streak_days += 1

        day_types = set()
        odd_count = 0
        even_count = 0
        for i in occ:
            for a in acts_by_i.get(i, []):
                t = pick_type_tag(a.get("tags") or [], a.get("subject") or "")
                if t:
                    day_types.add(t)
                tags_u = [str(x).strip().upper() for x in (a.get("tags") or [])]
                if "ODD" in tags_u:
                    odd_count += 1
                if "EVEN" in tags_u:
                    even_count += 1

        if "LABORATORIUM" in day_types:
            lab_days += 1
        if len(day_types) >= 2:
            mixed_type_days += 1

        odd_even_imbalance += abs(odd_count - even_count)

        if d == "Fri":
            friday_penalty += len(occ) + (2 if occ and occ[-1] >= len(hours) - 2 else 0)
            friday_late_classes += sum(1 for x in occ if x >= 4)
        if d == "Mon":
            monday_bonus += max(0, 4 - len(occ))

    days_with_classes_safe = max(total_days_with_classes, 1)
    daily_load_variance = np.var(daily_loads) if daily_loads else 0.0

    return {
        "campus_switch_0": campus_switch_0,
        "campus_switch_1": campus_switch_1,
        "gaps1": gaps1,
        "gaps2p": gaps2p,
        "single_class_days": single_class_days,
        "long_streak_days": long_streak_days,
        "dayoff_count": dayoff_count,
        "days_with_classes": total_days_with_classes,
        "total_activities": total_activities,
        "earliest_start_mean": earliest_start_sum / days_with_classes_safe,
        "latest_end_mean": latest_end_sum / days_with_classes_safe,
        "daily_span_mean": daily_span_sum / days_with_classes_safe,
        "morning_classes_count": morning_classes_count,
        "late_classes_count": late_classes_count,
        "lab_days": lab_days,
        "odd_even_imbalance": odd_even_imbalance,
        "mixed_type_days": mixed_type_days,
        "friday_penalty": friday_penalty,
        "monday_bonus": monday_bonus,
        "multi_campus_days": multi_campus_days,
        "friday_late_classes": friday_late_classes,
        "campus_rush_days": campus_rush_days,
        "daily_load_variance": daily_load_variance,
    }


class BradleyTerryModel:
    def __init__(self):
        self.model = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        coef = self.model.coef_[0]
        return np.abs(coef)


if HAS_TORCH:
    class RankNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            return self.net(x)


    class RankNetWrapper:
        def __init__(self, input_dim, hidden_dim=32, epochs=40, lr=1e-3):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.epochs = epochs
            self.lr = lr
            self.model = RankNet(input_dim, hidden_dim)
            self.is_fitted = False

        def fit(self, X, y):
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.BCEWithLogitsLoss()

            self.model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                logits = self.model(X_t)
                loss = criterion(logits, y_t)
                loss.backward()
                optimizer.step()

            self.is_fitted = True
            return self

        def predict_proba(self, X):
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32)
                logits = self.model(X_t)
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            return np.vstack([1 - probs, probs]).T

        def predict(self, X):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int)

        @property
        def feature_importances_(self):
            first = self.model.net[0]
            w = first.weight.detach().cpu().numpy()
            return np.mean(np.abs(w), axis=0)
else:
    class RankNetWrapper:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch nie jest dostępny, więc RankNet nie może być użyty.")


def build_synthetic_candidates(n_candidates: int, random_state: int = 42) -> pd.DataFrame:
    rng = random.Random(random_state)
    base_activities = build_synthetic_activity_base(random_state=random_state)

    room_to_campus = {r: "C1" for r in ROOMS_C1}
    room_to_campus.update({r: "C2" for r in ROOMS_C2})

    rows = []
    for i in range(n_candidates):
        cand = generate_synthetic_candidate(i + 1, base_activities, rng)
        metrics = compute_per_subgroup_metrics(
            table=cand["cell_map"],
            days=cand["days"],
            hours=cand["hours"],
            room_to_campus=room_to_campus,
        )
        row = {
            "candidate_id": cand["candidate_id"],
            "run": cand["run"],
            "subgroup": cand["subgroup"],
            "profile": cand["profile"],
            "days": cand["days"],
            "hours": cand["hours"],
            "cell_map": cand["cell_map"],
            "base_activity_count": cand["base_activity_count"],
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    if "total_activities" in df.columns and df["total_activities"].nunique() != 1:
        raise RuntimeError("Sztuczne kandydaty nie mają tej samej liczby aktywności. Generator wymaga poprawki.")

    return df


# -------------------- pair selection
def select_diverse_candidates(df: pd.DataFrame, n_select: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= n_select:
        return df.copy().reset_index(drop=True)

    X = transform_features_for_learning(df[FEATURE_COLS]).astype(float)
    scaler = MinMaxScaler()
    Xn = scaler.fit_transform(X)

    rng = random.Random(random_state)
    chosen = [rng.randrange(len(df))]
    remaining = set(range(len(df))) - set(chosen)

    while len(chosen) < n_select and remaining:
        best_i = None
        best_score = -1.0
        for i in remaining:
            dmin = min(np.linalg.norm(Xn[i] - Xn[j]) for j in chosen)
            if dmin > best_score:
                best_score = dmin
                best_i = i
        chosen.append(best_i)
        remaining.remove(best_i)

    return df.iloc[chosen].reset_index(drop=True)


def _normalized_matrix(df: pd.DataFrame, fit_scaler: bool = False):
    global FEATURE_SCALER

    X = transform_features_for_learning(df[FEATURE_COLS]).astype(float)

    if fit_scaler or FEATURE_SCALER is None:
        FEATURE_SCALER = MinMaxScaler()
        FEATURE_SCALER.fit(X)

    return FEATURE_SCALER.transform(X)


def generate_extreme_pairs(df: pd.DataFrame, n_pairs: int) -> List[Tuple[int, int]]:
    Xn = _normalized_matrix(df, fit_scaler=True)
    out = []
    used = set()

    for group_name, cols in PAIR_GROUPS.items():
        valid_cols = [c for c in cols if c in FEATURE_COLS]
        if not valid_cols:
            continue

        grp_idx = [FEATURE_COLS.index(c) for c in valid_cols]
        grp_score = Xn[:, grp_idx].mean(axis=1) * GROUP_IMPORTANCE.get(group_name, 1.0)

        i_best = int(np.argmax(grp_score))
        i_worst = int(np.argmin(grp_score))

        if i_best != i_worst:
            key = tuple(sorted((i_best, i_worst)))
            if key not in used:
                out.append(key)
                used.add(key)
                if len(out) >= n_pairs:
                    return out

    return out[:n_pairs]


def generate_tradeoff_pairs(df: pd.DataFrame, n_pairs: int):
    Xn = _normalized_matrix(df)

    candidates = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            score = pair_difficulty_score(Xn[i], Xn[j])
            candidates.append((i, j, score))

    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    out = []
    used = set()

    for i, j, _ in candidates:
        key = tuple(sorted((i, j)))

        if key in used:
            continue

        used.add(key)
        out.append(key)

        if len(out) >= n_pairs:
            break

    return out


def generate_initial_pairs(df: pd.DataFrame, n_pairs_total: int) -> List[Tuple[int, int]]:
    if len(df) < 2:
        return []

    _normalized_matrix(df, fit_scaler=True)

    q1 = max(2, n_pairs_total // 3)
    q2 = max(2, n_pairs_total // 3)
    q3 = max(0, n_pairs_total - q1 - q2)

    p1 = generate_extreme_pairs(df, q1)
    p2 = generate_tradeoff_pairs(df, q2)

    used = set(tuple(sorted(p)) for p in p1 + p2)
    rest = []

    Xn = _normalized_matrix(df)
    all_pairs = []
    all_dists = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            key = (i, j)
            if key in used:
                continue
            dist = float(np.linalg.norm(Xn[i] - Xn[j]))
            all_pairs.append((i, j, dist))
            all_dists.append(dist)

    if all_pairs:
        med = statistics.median(all_dists)
        all_pairs = sorted(all_pairs, key=lambda x: abs(x[2] - med))

    for i, j, _ in all_pairs:
        rest.append((i, j))
        if len(rest) >= q3:
            break

    out = []
    seen = set()
    for p in p1 + p2 + rest:
        key = tuple(sorted(p))
        if key not in seen:
            out.append(key)
            seen.add(key)
        if len(out) >= n_pairs_total:
            break

    return out[:n_pairs_total]


# -------------------- learning
def fit_models(X, y, qid=None):
    """
    Trenuje modele rankingowe przy użyciu LambdaMART (LGBMRanker, XGBRanker).
    Jeśli qid jest None, próbuje użyć klasyfikatorów (BradleyTerry, RankNet) – opcjonalnie.
    """
    models = {}
    
    if qid is not None:
        # Prawdziwe rankery listwise
        if HAS_LGBM:
            models["LightGBM"] = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                boosting_type="gbdt",
                n_estimators=100,
                random_state=42,
                verbose=-1
            )
            models["LightGBM"].fit(X, y, qid=qid)
            
        if HAS_XGB:
            models["XGBoost"] = XGBRanker(
                objective="rank:ndcg",
                eval_metric="ndcg",
                n_estimators=100,
                random_state=42
            )
            models["XGBoost"].fit(X, y, qid=qid)
    else:
        # Fallback do starych modeli (BradleyTerry, RankNet) – nie używane w głównym scenariuszu
        if HAS_LGBM:
            from lightgbm import LGBMClassifier
            lgbm = LGBMClassifier(n_estimators=80, random_state=42, verbose=-1)
            lgbm.fit(X, y)
            models["LightGBM"] = lgbm
        if HAS_XGB:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(n_estimators=80, random_state=42, eval_metric="logloss")
            xgb.fit(X, y)
            models["XGBoost"] = xgb
        models["BradleyTerry"] = BradleyTerryModel().fit(X, y)
        if HAS_TORCH:
            rn = RankNetWrapper(input_dim=X.shape[1], hidden_dim=32, epochs=40, lr=1e-3)
            rn.fit(X, y)
            models["RankNet"] = rn
        
    return models


def select_active_pair_ensemble(
    df: pd.DataFrame,
    models: dict,
    already_used_pairs: List[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    Xcand = _normalized_matrix(df).astype(float)
    used = set(tuple(sorted(p)) for p in already_used_pairs)

    best_pair = None
    best_uncertainty = -1.0

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            key = (i, j)
            if key in used:
                continue

            diff = (Xcand[i] - Xcand[j]).reshape(1, -1)

            probs = []
            for model in models.values():
                if hasattr(model, "predict_proba"):
                    p = float(model.predict_proba(diff)[0, 1])
                else:
                    # Dla rankerów: przewidujemy score, różnicę normalizujemy sigmoid
                    s_left = float(model.predict(Xcand[i].reshape(1, -1))[0])
                    s_right = float(model.predict(Xcand[j].reshape(1, -1))[0])
                    p = 1.0 / (1.0 + np.exp(s_right - s_left))  # sigmoid różnicy
                probs.append(p)

            if not probs:
                continue

            variance = float(np.var(probs))
            closeness = 1.0 - abs(np.mean(probs) - 0.5) * 2.0
            score = 0.7 * variance + 0.3 * closeness
            # DODANIE ODLEGŁOŚCI FEATURE DO ENSEMBLE
            feature_distance = np.linalg.norm(Xcand[i] - Xcand[j])
            score += 0.15 * feature_distance

            if score > best_uncertainty:
                best_uncertainty = score
                best_pair = key

    return best_pair


def select_active_pair(df: pd.DataFrame, model, already_used_pairs: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    Xcand = _normalized_matrix(df).astype(float)
    used = set(tuple(sorted(p)) for p in already_used_pairs)

    best_pair = None
    best_score = -1.0

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            key = (i, j)
            if key in used:
                continue

            diff = (Xcand[i] - Xcand[j]).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba(diff)[0, 1])
            else:
                s_left = float(model.predict(Xcand[i].reshape(1, -1))[0])
                s_right = float(model.predict(Xcand[j].reshape(1, -1))[0])
                p = 1.0 / (1.0 + np.exp(s_right - s_left))

            # NOWE SCORE: niepewność + różnorodność
            uncertainty = 1.0 - abs(p - 0.5) * 2.0  # im bliżej 0.5 tym wyższa niepewność
            feature_distance = np.linalg.norm(Xcand[i] - Xcand[j])
            score = 0.7 * uncertainty + 0.3 * feature_distance

            if score > best_score:
                best_score = score
                best_pair = key

    return best_pair


def score_candidates_pairwise(df_candidates: pd.DataFrame, model) -> pd.Series:
    """
    Dla rankerów (LGBMRanker, XGBRanker) predict zwraca ciągły skor.
    Dla innych modeli (BradleyTerry, RankNet) używamy starej metody pairwise.
    """
    # Sprawdzamy czy model jest rankerem (ma metodę predict i nie ma predict_proba? Heurystyka)
    X = df_candidates[FEATURE_COLS].copy()
    try:
        # Dla rankerów predict zadziała na całej macierzy
        scores = model.predict(X)
        return pd.Series(scores, index=df_candidates.index)
    except Exception:
        # Stara metoda pairwise (dla klasyfikatorów)
        Xcand = _normalized_matrix(df_candidates).astype(float)
        n = len(df_candidates)
        wins = np.zeros(n, dtype=float)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = (Xcand[i] - Xcand[j]).reshape(1, -1)
                if hasattr(model, "predict_proba"):
                    p = float(model.predict_proba(diff)[0, 1])
                else:
                    pred = int(model.predict(diff)[0])
                    p = float(pred)
                wins[i] += p

        if n > 1:
            wins = wins / (n - 1)
        return pd.Series(wins, index=df_candidates.index)


def aggregate_run_scores(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for run_name, g in df.groupby("run"):
        vals = sorted([float(x) for x in g[score_col].dropna().tolist()], reverse=True)
        if not vals:
            continue

        best_subgroup_score = vals[0]
        top3_mean_score = float(np.mean(vals[:3]))
        median_score = float(np.median(vals))
        score_std = float(np.std(vals))
        weak_ratio = float(np.mean(np.array(vals) < median_score))

        best_row = g.sort_values(score_col, ascending=False).iloc[0]

        rows.append({
            "run": run_name,
            "best_subgroup": best_row["subgroup"],
            "best_subgroup_score": best_subgroup_score,
            "top3_mean_score": top3_mean_score,
            "median_score": median_score,
            "score_std": score_std,
            "weak_ratio": weak_ratio,
            "subgroups_count": len(g),
        })

    return pd.DataFrame(rows).sort_values("best_subgroup_score", ascending=False).reset_index(drop=True)


# -------------------- consistency check
def estimate_preference_consistency(answers):
    """
    Oblicza procent spójnych odpowiedzi.
    Dla każdej pary (id1, id2) sprawdza czy nie ma sprzeczności (left > right i right > left).
    Zwraca procent par spójnych względem wszystkich rozstrzygniętych par.
    """
    pairs = {}
    for ans in answers:
        choice = ans.get("choice")
        if choice not in ("left", "right"):
            continue
        left_id = ans["left_id"]
        right_id = ans["right_id"]
        key = tuple(sorted((left_id, right_id)))
        # Zapamiętujemy kierunek: True jeśli left > right
        direction = (left_id == key[0] and choice == "left") or (right_id == key[0] and choice == "right")
        if key in pairs:
            # Jeśli już mamy, sprawdź zgodność
            if pairs[key] != direction:
                pairs[key] = None  # sprzeczność
        else:
            pairs[key] = direction

    total = 0
    consistent = 0
    for v in pairs.values():
        if v is not None:
            total += 1
            consistent += 1
        else:
            total += 1  # para sprzeczna też wliczana do total
    if total == 0:
        return 100.0
    return (consistent / total) * 100.0


# -------------------- render synthetic timetable
def dedup_tiles(tiles: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for a in tiles:
        key = (
            a.get("activity_id", ""),
            a.get("subject", ""),
            a.get("room", ""),
            tuple(a.get("teachers") or []),
            tuple(a.get("tags") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def render_grid_html(
    days: List[str],
    hours: List[str],
    cell_map: Dict[Tuple[str, str], List[dict]],
    title: str,
) -> str:
    css = """
    <style>
      .tt-wrap { width: 100%; overflow-x: auto; }
      table.tt { border-collapse: collapse; width: 100%; min-width: 720px; table-layout: fixed; }
      table.tt th, table.tt td { border: 1px solid #ddd; vertical-align: top; padding: 5px; }
      table.tt th { background: #f6f7f9; font-weight: 700; text-align: center; }
      table.tt td { height: 95px; background: #fff; }
      .hour { width: 120px; background: #fafafa; font-weight: 700; }
      .tile { border-radius: 8px; padding: 5px 6px; margin: 4px 0; border: 1px solid rgba(0,0,0,.10); }
      .tile .subj { font-weight: 700; font-size: 12px; margin-bottom: 3px; }
      .tile .meta { font-size: 11px; opacity: .86; line-height: 1.2; }
      .badge { display:inline-block; font-size: 10px; padding: 1px 6px; border-radius: 999px; background: rgba(0,0,0,.07); margin-right: 5px; }
      .WYKŁAD       { background: #eef6ff; }
      .ĆWICZENIA    { background: #f3f7ee; }
      .LABORATORIUM { background: #fff4e6; }
      .PROJEKT      { background: #f5efff; }
      .SEMINARIUM   { background: #fbefff; }
      .tt-title { font-size: 20px; font-weight: 800; margin: 8px 0 12px 0; }
    </style>
    """

    html = [css, f"<div class='tt-title'>{title}</div>", "<div class='tt-wrap'>", "<table class='tt'>"]
    html.append("<tr>")
    html.append("<th class='hour'>Godzina</th>")
    for d in days:
        html.append(f"<th>{d}</th>")
    html.append("</tr>")

    for h in hours:
        html.append("<tr>")
        html.append(f"<td class='hour'>{h}</td>")
        for d in days:
            acts = dedup_tiles(cell_map.get((d, h), []))
            cell = []
            for a in acts:
                tags = a.get("tags") or []
                type_tag = pick_type_tag(tags, a.get("subject") or "")
                week_tag = ""
                tags_u = [str(x).strip().upper() for x in tags]
                if "ODD" in tags_u:
                    week_tag = "ODD"
                elif "EVEN" in tags_u:
                    week_tag = "EVEN"

                cls = type_tag if type_tag in {"WYKŁAD", "ĆWICZENIA", "LABORATORIUM", "PROJEKT", "SEMINARIUM"} else ""
                badges = []
                if type_tag:
                    badges.append(f"<span class='badge'>{type_tag}</span>")
                if week_tag:
                    badges.append(f"<span class='badge'>{week_tag}</span>")

                subj = str(a.get("subject") or "").strip() or "Zajęcia"
                teachers = ", ".join(a.get("teachers") or [])
                room = str(a.get("room") or "").strip()

                meta_parts = []
                if teachers:
                    meta_parts.append(f"Prow.: {teachers}")
                if room:
                    meta_parts.append(f"Sala: {room}")

                meta = "<br/>".join(meta_parts)

                cell.append(
                    f"<div class='tile {cls}'>"
                    f"<div class='subj'>{''.join(badges)}{subj}</div>"
                    f"<div class='meta'>{meta}</div>"
                    f"</div>"
                )

            html.append("<td>" + "".join(cell) + "</td>")
        html.append("</tr>")

    html.append("侠</div>")
    return "\n".join(html)


# -------------------- real candidates from runs
def list_sessions(out_root: Path) -> List[Path]:
    if not out_root.exists():
        return []
    return sorted([p for p in out_root.iterdir() if p.is_dir()], reverse=True)


def load_session_summary(root: Path) -> dict:
    p = root / "generation_summary.json"
    if p.exists():
        obj = load_json(p)
        if obj:
            return obj
    return {}


def get_run_dirs_from_session(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def get_rankable_runs(root: Path) -> List[Path]:
    summary = load_session_summary(root)
    runs_meta = summary.get("runs") or []

    by_name = {p.name: p for p in get_run_dirs_from_session(root)}

    rankable = []
    if runs_meta:
        for r in runs_meta:
            if r.get("returncode") == 0 and r.get("ranking_ready"):
                rn = r.get("run_name")
                if rn in by_name:
                    rankable.append(by_name[rn])

    if rankable:
        return rankable

    return get_run_dirs_from_session(root)


def find_run_subgroups_xml(run_dir: Path) -> Optional[Path]:
    p = run_dir / "instance_subgroups.xml"
    if p.exists():
        return p
    cands = sorted(run_dir.rglob("*subgroups*.xml"))
    return cands[0] if cands else None


def find_input_fet_for_session(root: Path, last_gen_path: Path) -> Optional[Path]:
    p = root / "input_fet_info.json"
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            fp = obj.get("file_path")
            if fp and Path(fp).exists():
                return Path(fp)
        except Exception:
            pass

    summary = load_session_summary(root)
    fp = summary.get("input_fet")
    if fp and Path(fp).exists():
        return Path(fp)

    last = load_json(last_gen_path) or {}
    fp = last.get("input_fet")
    if fp and Path(fp).exists():
        return Path(fp)

    return None


def parse_input_fet(fet_path_str: str):
    fet_path = Path(fet_path_str)
    tree = ET.parse(fet_path)
    root = tree.getroot()

    room_to_building = {}
    rooms_list = root.find("Rooms_List")
    if rooms_list is not None:
        for r in rooms_list.findall("Room"):
            rid = first_text(r, ["Name"], "")
            bid = first_text(r, ["Building"], "")
            if rid:
                room_to_building[rid] = bid

    room_to_campus = {rid: building_to_campus(bid) for rid, bid in room_to_building.items()}

    act_idx = {}
    acts = root.find("Activities_List")
    if acts is not None:
        for a in acts.findall("Activity"):
            aid = first_text(a, ["Id", "Activity_Id"], "") or attr_any(a, ["Id", "Activity_Id", "id"], "")
            subject = first_text(a, ["Subject"], "")
            teachers = [str(x.text).strip() for x in a.findall("Teacher") if x.text and str(x.text).strip()]
            tags = [str(x.text).strip() for x in a.findall("Activity_Tag") if x.text and str(x.text).strip()]
            comments = first_text(a, ["Comments"], "")

            act_idx[aid] = {
                "subject": subject,
                "teachers": teachers,
                "tags": tags,
                "comments": comments,
            }

    return {
        "room_to_campus": room_to_campus,
        "activity_index": act_idx,
    }


def parse_subgroups_xml_full(xml_path_str: str):
    xml_path = Path(xml_path_str)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    schedule = defaultdict(lambda: defaultdict(list))
    days_seen = []
    hours_seen = []
    subgroups_seen = []

    for sub in root.findall(".//Subgroup"):
        sname = name_of(sub)
        if not sname:
            continue

        subgroups_seen.append(sname)

        for day_node in sub.findall("./Day"):
            dname = name_of(day_node)
            if not dname:
                continue
            if dname not in days_seen:
                days_seen.append(dname)

            for hour_node in day_node.findall("./Hour"):
                hname = name_of(hour_node)
                if not hname:
                    continue
                if hname not in hours_seen:
                    hours_seen.append(hname)

                act_nodes = hour_node.findall("./Activity")
                if not act_nodes:
                    continue

                room = ""
                room_node = hour_node.find("./Room")
                if room_node is not None:
                    room = (
                        room_node.attrib.get("name")
                        or room_node.attrib.get("Name")
                        or (room_node.text.strip() if room_node.text else "")
                    ).strip()

                subject = ""
                subject_node = hour_node.find("./Subject")
                if subject_node is not None:
                    subject = (
                        subject_node.attrib.get("name")
                        or subject_node.attrib.get("Name")
                        or (subject_node.text.strip() if subject_node.text else "")
                    ).strip()

                teachers = []
                for tn in hour_node.findall("./Teacher"):
                    val = (
                        tn.attrib.get("name")
                        or tn.attrib.get("Name")
                        or (tn.text.strip() if tn.text else "")
                    )
                    val = str(val).strip()
                    if val:
                        teachers.append(val)

                tags = []
                for tg in hour_node.findall("./Activity_Tag"):
                    val = (
                        tg.attrib.get("name")
                        or tg.attrib.get("Name")
                        or (tg.text.strip() if tg.text else "")
                    )
                    val = str(val).strip()
                    if val:
                        tags.append(val)

                for act in act_nodes:
                    aid = (
                        act.attrib.get("id")
                        or act.attrib.get("Id")
                        or first_text(act, ["Activity_Id", "Id", ".//Activity_Id", ".//Id"], "")
                    )
                    aid = str(aid).strip()

                    schedule[sname][(dname, hname)].append({
                        "activity_id": aid,
                        "room": room,
                        "subject": subject,
                        "comments": "",
                        "tags": tags[:],
                        "teachers": teachers[:],
                    })

    return {
        "days": days_seen,
        "hours": hours_seen,
        "subgroups": sorted(set(subgroups_seen)),
        "schedule": {sg: dict(cell_map) for sg, cell_map in schedule.items()}
    }


def enrich_schedule(schedule_raw, activity_index):
    out = defaultdict(lambda: defaultdict(list))
    for sg, cell_map in schedule_raw.items():
        for key, acts in cell_map.items():
            for a in acts:
                merged = dict(a)
                aid = str(merged.get("activity_id") or "").strip()
                ref = activity_index.get(aid, {})
                if ref:
                    if not str(merged.get("subject") or "").strip():
                        merged["subject"] = ref.get("subject", "")
                    if not (merged.get("teachers") or []):
                        merged["teachers"] = ref.get("teachers", [])
                    tags = list(merged.get("tags") or [])
                    for tg in (ref.get("tags") or []):
                        if tg not in tags:
                            tags.append(tg)
                    merged["tags"] = tags
                    if not str(merged.get("comments") or "").strip():
                        merged["comments"] = ref.get("comments", "")
                out[sg][key].append(merged)
    return out


def build_real_candidates(root: Path, last_gen_path: Path) -> pd.DataFrame:
    fet_path = find_input_fet_for_session(root, last_gen_path)
    if not fet_path or not fet_path.exists():
        raise RuntimeError("Nie udało się znaleźć wejściowego pliku .fet dla sesji.")

    fet_info = parse_input_fet(str(fet_path))
    run_dirs = get_rankable_runs(root)
    if not run_dirs:
        raise RuntimeError("Brak runów gotowych do rankingu.")

    rows = []

    for rd in run_dirs:
        xmlp = find_run_subgroups_xml(rd)
        if not xmlp or not xmlp.exists():
            continue

        parsed = parse_subgroups_xml_full(str(xmlp))
        raw_sched = parsed["schedule"]
        sched = enrich_schedule(raw_sched, fet_info["activity_index"])

        run_days = parsed["days"]
        run_hours = parsed["hours"]
        run_subgroups = parsed["subgroups"]

        selected_subgroups = [sg for sg in run_subgroups if is_lab_subgroup(sg)]
        if not selected_subgroups:
            selected_subgroups = run_subgroups

        for sg in selected_subgroups:
            table = sched.get(sg, {}) or {}
            metrics = compute_per_subgroup_metrics(
                table=table,
                days=run_days,
                hours=run_hours,
                room_to_campus=fet_info["room_to_campus"],
            )
            row = {
                "candidate_id": f"{rd.name}::{sg}",
                "run": rd.name,
                "subgroup": sg,
                "profile": "REAL",
                "days": run_days,
                "hours": run_hours,
                "cell_map": table,
            }
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Nie udało się przygotować kandydatów z prawdziwych runów.")

    return df.drop_duplicates(subset=["candidate_id"]).reset_index(drop=True)