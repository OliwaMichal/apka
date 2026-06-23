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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
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

# ── feature schema ────────────────────────────────────────────────────────────
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

# ── Pareto objective directions (for Pareto frontier) ──────────────────────
# Each group has a "primary" scalar: lower is better for cost groups, higher for benefit.
PARETO_OBJECTIVES = {
    "Gaps":   ("gaps1", "cost"),
    "Campus": ("campus_switch_0", "cost"),
    "Free":   ("dayoff_count", "benefit"),
    "Load":   ("daily_load_variance", "cost"),
    "Time":   ("earliest_start_mean", "benefit"),
    "Friday": ("friday_penalty", "cost"),
}

FEATURE_SCALER: Optional[MinMaxScaler] = None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  TRUE PAIRWISE LEARNING  (xi − xj feature differences)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_pairwise_dataset(synth_df: pd.DataFrame, answers: List[dict]):
    """
    Transforms user answers into a true pairwise dataset.

    Each comparison (left, right) becomes ONE row with features:
        Δx = x_left − x_right   (signed difference vector)

    Label:
        1  → left preferred
        0  → right preferred
        (skip answers are dropped)

    Additionally returns sample_weight based on answer strength.
    """
    X_list, y_list, w_list = [], [], []

    strength_weight = {
        ("left",  "strong"): (1, 1.0),
        ("left",  "slight"): (1, 0.55),
        ("right", "strong"): (0, 1.0),
        ("right", "slight"): (0, 0.55),
    }

    for ans in answers:
        choice   = ans.get("choice")
        strength = ans.get("strength")
        key = (choice, strength)
        if key not in strength_weight:
            continue  # skip / unknown

        label, weight = strength_weight[key]

        left_row  = synth_df[synth_df["candidate_id"] == ans["left_id"]]
        right_row = synth_df[synth_df["candidate_id"] == ans["right_id"]]
        if left_row.empty or right_row.empty:
            continue

        xl = left_row[FEATURE_COLS].iloc[0].values.astype(float)
        xr = right_row[FEATURE_COLS].iloc[0].values.astype(float)
        delta = xl - xr  # signed difference

        X_list.append(delta)
        y_list.append(label)
        w_list.append(weight)

        # Anti-symmetric augmentation: flip sign ↔ flip label
        X_list.append(-delta)
        y_list.append(1 - label)
        w_list.append(weight)

    if not X_list:
        return None, None, None

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    w = np.array(w_list, dtype=float)
    return X, y, w


# ── legacy helper kept for compatibility ────────────────────────────────────
def build_ranking_dataset(synth_df, answers):
    """Thin wrapper – returns (X, y, None) using pairwise differences."""
    X, y, _ = build_pairwise_dataset(synth_df, answers)
    return X, y, None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  PREFERENCE EMBEDDING  (latent utility vector per user session)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PreferenceEmbedding:
    """
    Lightweight latent-utility model.

    Learns a weight vector w ∈ R^d such that the utility of a schedule x is:
        U(x) = w · φ(x)
    where φ(x) is the sign-adjusted feature vector.

    The preference between two schedules is then:
        P(x_i > x_j) = σ(U(x_i) − U(x_j)) = σ(w · (x_i − x_j))

    This is a Bradley-Terry model with interpretable latent weights.
    After fitting, `self.weights` is the *preference embedding* of the user.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.weights = np.ones(n_features) / n_features  # uniform prior
        self._fitted = False

    def fit(self, X_delta: np.ndarray, y: np.ndarray, sample_weight=None):
        """X_delta: (n, d) pairwise differences;  y: binary labels."""
        if X_delta is None or len(X_delta) == 0:
            return self
        lr = LogisticRegression(
            max_iter=2000,
            C=2.0,
            solver="lbfgs",
            fit_intercept=False,  # no intercept → pure dot-product utility
            random_state=42,
        )
        lr.fit(X_delta, y, sample_weight=sample_weight)
        raw = lr.coef_[0]
        # Normalise so weights sum to 1 (interpretable as importance shares)
        self.weights = raw / (np.abs(raw).sum() + 1e-9)
        self._fitted = True
        return self

    def utility(self, X: np.ndarray) -> np.ndarray:
        """Scalar utility score for each row of X (raw features, sign-adjusted)."""
        return X @ self.weights

    def predict_proba_delta(self, X_delta: np.ndarray) -> np.ndarray:
        """P(left > right) for each pairwise-difference row."""
        logits = X_delta @ self.weights
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1 - p, p]).T

    def weight_dataframe(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"feature": FEATURE_COLS, "latent_weight": self.weights})
            .sort_values("latent_weight", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def is_fitted(self):
        return self._fitted


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  FEATURE INTERACTIONS  (GAM-style + DeepFM-lite)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── explicit interaction features ──────────────────────────────────────────
INTERACTION_PAIRS = [
    # (feature_a, feature_b, description)
    ("friday_penalty",    "late_classes_count", "friday_and_late"),
    ("gaps1",             "campus_switch_0",    "gaps_with_campus_switch"),
    ("gaps2p",            "campus_switch_1",    "long_gaps_with_campus"),
    ("earliest_start_mean", "daily_span_mean",  "early_and_compact"),
    ("long_streak_days",  "daily_load_variance","streak_variance"),
    ("multi_campus_days", "campus_rush_days",   "campus_stress"),
    ("dayoff_count",      "friday_penalty",     "freeday_vs_friday"),
    ("morning_classes_count", "earliest_start_mean", "morning_load"),
]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds explicit multiplicative interaction features to a DataFrame.
    Returns a copy with extra columns.
    """
    out = df.copy()
    for fa, fb, name in INTERACTION_PAIRS:
        if fa in out.columns and fb in out.columns:
            out[f"ix_{name}"] = out[fa].astype(float) * out[fb].astype(float)
    return out


def get_interaction_feature_names() -> List[str]:
    return [f"ix_{name}" for _, _, name in INTERACTION_PAIRS]


ALL_FEATURE_COLS_WITH_IX = FEATURE_COLS + get_interaction_feature_names()


if HAS_TORCH:
    class DeepFMUtility(nn.Module):
        """
        Lightweight DeepFM-inspired utility network.

        Input: sign-adjusted feature vector (d,)
        Output: scalar utility score

        Architecture:
          - FM layer:  captures all pairwise interactions Σ_{i<j} <vi, vj> xi xj
          - Deep part: 2-layer MLP on raw features
          - Output: sum of FM + Deep
        """

        def __init__(self, input_dim: int, embed_dim: int = 4, hidden: int = 32):
            super().__init__()
            self.embed_dim = embed_dim
            # FM embeddings
            self.embeddings = nn.Embedding(input_dim, embed_dim)
            nn.init.normal_(self.embeddings.weight, std=0.01)
            # Deep part
            self.deep = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            # FM first-order
            self.linear = nn.Linear(input_dim, 1, bias=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, input_dim)
            # FM second order
            idx = torch.arange(x.shape[1], device=x.device)
            emb = self.embeddings(idx)           # (d, k)
            # weighted embeddings
            wx = (x.unsqueeze(-1) * emb)         # (batch, d, k)
            sum_sq   = wx.sum(dim=1) ** 2        # (batch, k)
            sq_sum   = (wx ** 2).sum(dim=1)      # (batch, k)
            fm_out   = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)

            deep_out  = self.deep(x)
            linear_out = self.linear(x)
            return fm_out + deep_out + linear_out   # (batch, 1)

    class DeepFMWrapper:
        """Wraps DeepFMUtility for pairwise training (Bradley-Terry style)."""

        def __init__(self, input_dim: int, embed_dim: int = 4,
                     hidden: int = 32, epochs: int = 60, lr: float = 1e-3):
            self.input_dim = input_dim
            self.model = DeepFMUtility(input_dim, embed_dim, hidden)
            self.epochs = epochs
            self.lr = lr
            self._fitted = False

        def fit(self, X_delta: np.ndarray, y: np.ndarray,
                sample_weight: Optional[np.ndarray] = None):
            X_t = torch.tensor(X_delta, dtype=torch.float32)
            y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
            w_t = (torch.tensor(sample_weight, dtype=torch.float32).reshape(-1, 1)
                   if sample_weight is not None
                   else torch.ones(len(y), 1))

            opt = optim.Adam(self.model.parameters(), lr=self.lr)
            crit = nn.BCEWithLogitsLoss(reduction="none")

            self.model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                logits = self.model(X_t)
                loss = (crit(logits, y_t) * w_t).mean()
                loss.backward()
                opt.step()

            self._fitted = True
            return self

        def predict_proba(self, X_delta: np.ndarray) -> np.ndarray:
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_delta, dtype=torch.float32)
                logits = self.model(X_t).squeeze(-1)
                p = torch.sigmoid(logits).cpu().numpy()
            return np.vstack([1 - p, p]).T

        def predict(self, X_delta: np.ndarray) -> np.ndarray:
            return (self.predict_proba(X_delta)[:, 1] >= 0.5).astype(int)

        def score_single(self, x: np.ndarray) -> float:
            """Utility score for a single feature vector."""
            self.model.eval()
            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                s = self.model(x_t).item()
            return s

        @property
        def is_fitted(self):
            return self._fitted


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  BAYESIAN ACTIVE LEARNING                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class BayesianPreferenceModel:
    """
    Gaussian-Process based preference model.

    Maintains a posterior over pairwise preferences.
    Used for:
      - Thompson Sampling pair selection
      - Expected Information Gain (EIG) scoring
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features))
        self.gp = GaussianProcessClassifier(
            kernel=kernel,
            max_iter_predict=200,
            random_state=42,
            n_restarts_optimizer=0,
        )
        self._fitted = False
        self._X_train = None
        self._y_train = None

    def fit(self, X_delta: np.ndarray, y: np.ndarray):
        if X_delta is None or len(X_delta) < 4:
            return self
        # GP can be slow on large datasets — subsample if needed
        max_gp = 300
        if len(X_delta) > max_gp:
            idx = np.random.choice(len(X_delta), max_gp, replace=False)
            X_delta = X_delta[idx]
            y = y[idx]
        try:
            self.gp.fit(X_delta, y)
            self._fitted = True
            self._X_train = X_delta
            self._y_train = y
        except Exception:
            self._fitted = False
        return self

    def posterior_uncertainty(self, X_delta: np.ndarray) -> np.ndarray:
        """
        Returns uncertainty (entropy of Bernoulli) for each pairwise diff row.
        Higher = more uncertain = more valuable to query.
        """
        if not self._fitted:
            return np.ones(len(X_delta)) * 0.5
        proba = self.gp.predict_proba(X_delta)[:, 1]
        # Binary entropy
        p = np.clip(proba, 1e-9, 1 - 1e-9)
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return entropy

    def thompson_sample_score(self, X_feature: np.ndarray,
                               n_samples: int = 10) -> np.ndarray:
        """
        Thompson Sampling: draw utility weights from posterior, score candidates.
        Returns mean utility score per candidate (row of X_feature).
        """
        if not self._fitted or self._X_train is None:
            return np.zeros(len(X_feature))

        # Approximate posterior over utility weights via logistic + GP uncertainty
        proba_matrix = np.zeros((n_samples, len(X_feature)))
        for s in range(n_samples):
            # Perturb GP predictions with posterior noise
            try:
                p_mean = self.gp.predict_proba(
                    X_feature - X_feature.mean(axis=0, keepdims=True)
                )[:, 1]
                noise = np.random.normal(0, 0.1, size=p_mean.shape)
                proba_matrix[s] = np.clip(p_mean + noise, 0, 1)
            except Exception:
                proba_matrix[s] = 0.5
        return proba_matrix.mean(axis=0)

    def expected_information_gain(self, xi: np.ndarray, xj: np.ndarray) -> float:
        """
        EIG for querying the pair (xi, xj).
        Approximated as the entropy of P(xi > xj) under current posterior.
        """
        if not self._fitted:
            return 0.5
        delta = (xi - xj).reshape(1, -1)
        h = self.posterior_uncertainty(delta)
        return float(h[0])

    @property
    def is_fitted(self):
        return self._fitted


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  PARETO FRONTIER                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_pareto_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one column per Pareto objective, sign-adjusted
    so that HIGHER is always better (for dominance checks).
    """
    out = pd.DataFrame(index=df.index)
    out["candidate_id"] = df["candidate_id"]
    out["run"] = df.get("run", "")
    out["subgroup"] = df.get("subgroup", "")

    for obj_name, (feat, direction) in PARETO_OBJECTIVES.items():
        if feat not in df.columns:
            out[obj_name] = 0.0
            continue
        vals = df[feat].astype(float)
        if direction == "cost":
            out[obj_name] = -vals   # negate: lower cost → higher "good"
        else:
            out[obj_name] = vals
    return out


def is_pareto_dominated(scores: np.ndarray) -> np.ndarray:
    """
    Given scores matrix (n, k) where higher is always better,
    returns boolean array: dominated[i] = True if candidate i is dominated.
    """
    n = len(scores)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i on all objectives and j > i on at least one
            if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                dominated[i] = True
                break
    return dominated


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the full Pareto frontier analysis DataFrame.

    Columns: candidate_id, run, subgroup, <objectives>, is_pareto_optimal,
             pareto_rank (1 = frontier, 2 = next layer, …)
    """
    obj_df = compute_pareto_objectives(df)
    obj_names = list(PARETO_OBJECTIVES.keys())
    scores = obj_df[obj_names].values.astype(float)

    remaining = np.arange(len(df))
    pareto_rank = np.zeros(len(df), dtype=int)
    rank = 1

    while len(remaining) > 0:
        sub_scores = scores[remaining]
        dominated = is_pareto_dominated(sub_scores)
        frontier_local = remaining[~dominated]
        pareto_rank[frontier_local] = rank
        remaining = remaining[dominated]
        rank += 1

    obj_df["pareto_rank"] = pareto_rank
    obj_df["is_pareto_optimal"] = pareto_rank == 1
    return obj_df


def pareto_crowding_distance(scores: np.ndarray) -> np.ndarray:
    """
    Computes crowding distance for diversity preservation on Pareto frontier.
    Candidates with larger crowding distance are more diverse.
    """
    n, k = scores.shape
    crowd = np.zeros(n)

    for obj in range(k):
        order = np.argsort(scores[:, obj])
        crowd[order[0]]  = np.inf
        crowd[order[-1]] = np.inf
        obj_range = scores[order[-1], obj] - scores[order[0], obj]
        if obj_range == 0:
            continue
        for i in range(1, n - 1):
            crowd[order[i]] += (scores[order[i + 1], obj] - scores[order[i - 1], obj]) / obj_range

    return crowd


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL FITTING (pairwise, unified)                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def fit_models(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    qid=None,               # ignored – kept for API compatibility
    model_types: Optional[List[str]] = None,
) -> dict:
    """
    Trains pairwise models on Δx = x_left − x_right feature differences.

    Always returns a dict: { model_name: fitted_model }

    Models all expose:
        .predict_proba(X_delta) -> (n, 2)
        .predict(X_delta)       -> (n,)
    """
    models = {}

    if model_types is None:
        # Default: use whatever is available
        if HAS_LGBM:
            model_types = ["LightGBM"]
        else:
            model_types = ["BradleyTerry"]

    for mtype in model_types:
        try:
            if mtype == "LightGBM" and HAS_LGBM:
                m = LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1,
                )
                m.fit(X, y, sample_weight=sample_weight)
                models["LightGBM"] = m

            elif mtype == "XGBoost" and HAS_XGB:
                m = XGBClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    random_state=42,
                    eval_metric="logloss",
                    verbosity=0,
                )
                kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
                m.fit(X, y, **kw)
                models["XGBoost"] = m

            elif mtype == "BradleyTerry":
                m = BradleyTerryModel()
                m.fit(X, y, sample_weight=sample_weight)
                models["BradleyTerry"] = m

            elif mtype == "RankNet" and HAS_TORCH:
                m = RankNetWrapper(input_dim=X.shape[1])
                m.fit(X, y, sample_weight=sample_weight)
                models["RankNet"] = m

            elif mtype == "DeepFM" and HAS_TORCH:
                m = DeepFMWrapper(input_dim=X.shape[1])
                m.fit(X, y, sample_weight=sample_weight)
                models["DeepFM"] = m

        except Exception as e:
            import warnings
            warnings.warn(f"fit_models: {mtype} failed – {e}")

    return models


def score_candidates_pairwise(df_candidates: pd.DataFrame, model) -> pd.Series:
    """
    Scores every candidate against all others using pairwise model.

    For each candidate i:
        score_i = mean P(i > j)  over all j ≠ i

    Works with any model that has predict_proba(X_delta).
    """
    Xcand = _normalized_matrix(df_candidates).astype(float)
    n = len(Xcand)
    wins = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = (Xcand[i] - Xcand[j]).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba(delta)[0, 1])
            else:
                p = 0.5
            wins[i] += p

    if n > 1:
        wins /= (n - 1)
    return pd.Series(wins, index=df_candidates.index)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BAYESIAN ACTIVE PAIR SELECTION                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def select_active_pair_bayesian(
    df: pd.DataFrame,
    bayes_model: BayesianPreferenceModel,
    already_used_pairs: List[Tuple[int, int]],
    strategy: str = "eig",          # "eig" | "thompson"
) -> Optional[Tuple[int, int]]:
    """
    Bayesian active learning pair selector.

    Strategies:
      'eig'      – pick pair maximising Expected Information Gain
      'thompson' – Thompson Sampling: pick pair where models disagree most
                   across posterior samples
    """
    Xcand = _normalized_matrix(df).astype(float)
    used = {tuple(sorted(p)) for p in already_used_pairs}

    best_pair  = None
    best_score = -np.inf

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            key = (i, j)
            if key in used:
                continue

            if strategy == "eig":
                score = bayes_model.expected_information_gain(Xcand[i], Xcand[j])
            else:
                # Thompson: uncertainty + feature distance
                delta = (Xcand[i] - Xcand[j]).reshape(1, -1)
                unc = bayes_model.posterior_uncertainty(delta)[0]
                dist = float(np.linalg.norm(Xcand[i] - Xcand[j]))
                score = 0.6 * unc + 0.4 * dist

            if score > best_score:
                best_score = score
                best_pair = key

    return best_pair


def select_active_pair_ensemble(
    df: pd.DataFrame,
    models: dict,
    already_used_pairs: List[Tuple[int, int]],
    bayes_model: Optional[BayesianPreferenceModel] = None,
) -> Optional[Tuple[int, int]]:
    """
    Ensemble active learning.

    Score = 0.5 * model_variance + 0.3 * closeness_to_0.5 + 0.2 * feature_distance
    If Bayesian model is available, adds 0.25 * EIG boost.
    """
    Xcand = _normalized_matrix(df).astype(float)
    used = {tuple(sorted(p)) for p in already_used_pairs}

    best_pair      = None
    best_uncertainty = -1.0

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            key = (i, j)
            if key in used:
                continue

            delta = (Xcand[i] - Xcand[j]).reshape(1, -1)
            probs = []
            for model in models.values():
                if hasattr(model, "predict_proba"):
                    p = float(model.predict_proba(delta)[0, 1])
                    probs.append(p)

            if not probs:
                continue

            variance  = float(np.var(probs))
            closeness = 1.0 - abs(np.mean(probs) - 0.5) * 2.0
            feat_dist = float(np.linalg.norm(Xcand[i] - Xcand[j]))
            score = 0.5 * variance + 0.3 * closeness + 0.2 * feat_dist

            if bayes_model is not None and bayes_model.is_fitted:
                eig = bayes_model.expected_information_gain(Xcand[i], Xcand[j])
                score += 0.25 * eig

            if score > best_uncertainty:
                best_uncertainty = score
                best_pair = key

    return best_pair


def select_active_pair(
    df: pd.DataFrame,
    model,
    already_used_pairs: List[Tuple[int, int]],
    bayes_model: Optional[BayesianPreferenceModel] = None,
) -> Optional[Tuple[int, int]]:
    """
    Single-model active learning with optional Bayesian boost.
    """
    Xcand = _normalized_matrix(df).astype(float)
    used = {tuple(sorted(p)) for p in already_used_pairs}

    best_pair  = None
    best_score = -1.0

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            key = (i, j)
            if key in used:
                continue

            delta = (Xcand[i] - Xcand[j]).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba(delta)[0, 1])
            else:
                p = 0.5

            uncertainty = 1.0 - abs(p - 0.5) * 2.0
            feat_dist   = float(np.linalg.norm(Xcand[i] - Xcand[j]))
            score = 0.7 * uncertainty + 0.3 * feat_dist

            if bayes_model is not None and bayes_model.is_fitted:
                eig = bayes_model.expected_information_gain(Xcand[i], Xcand[j])
                score += 0.25 * eig

            if score > best_score:
                best_score = score
                best_pair = key

    return best_pair


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLASSIC MODELS (kept, updated for pairwise interface)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class BradleyTerryModel:
    """Logistic regression on Δx differences (true pairwise)."""

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            fit_intercept=True,
            random_state=42,
        )

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return np.abs(self.model.coef_[0])


if HAS_TORCH:
    class RankNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            return self.net(x)

    class RankNetWrapper:
        def __init__(self, input_dim, hidden_dim=32, epochs=60, lr=1e-3):
            self.input_dim  = input_dim
            self.hidden_dim = hidden_dim
            self.epochs     = epochs
            self.lr         = lr
            self.model      = RankNet(input_dim, hidden_dim)
            self._fitted    = False

        def fit(self, X, y, sample_weight=None):
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
            w_t = (torch.tensor(sample_weight, dtype=torch.float32).reshape(-1, 1)
                   if sample_weight is not None
                   else torch.ones(len(y), 1))

            opt  = optim.Adam(self.model.parameters(), lr=self.lr)
            crit = nn.BCEWithLogitsLoss(reduction="none")

            self.model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                logits = self.model(X_t)
                loss = (crit(logits, y_t) * w_t).mean()
                loss.backward()
                opt.step()

            self._fitted = True
            return self

        def predict_proba(self, X):
            self.model.eval()
            with torch.no_grad():
                X_t  = torch.tensor(X, dtype=torch.float32)
                logits = self.model(X_t).squeeze(-1)
                probs  = torch.sigmoid(logits).cpu().numpy()
            return np.vstack([1 - probs, probs]).T

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        @property
        def feature_importances_(self):
            first = self.model.net[0]
            w = first.weight.detach().cpu().numpy()
            return np.mean(np.abs(w), axis=0)

        @property
        def is_fitted(self):
            return self._fitted

else:
    class RankNetWrapper:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GENERIC UTILS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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
    return "C1" if r[0] in {"A", "B", "C", "D", "E", "F"} else "C2"


def is_lab_subgroup(name: str) -> bool:
    n = str(name or "").upper()
    return "SUBGROUP" in n or "AUTOMATIC SUBGROUP" in n


def pick_type_tag(tags: List[str], subject: str = "") -> str:
    tags_u = [str(x).strip().upper() for x in (tags or [])]
    mapping = {
        "WYKŁAD": "WYKŁAD", "LECTURE": "WYKŁAD",
        "ĆWICZENIA": "ĆWICZENIA", "CWICZENIA": "ĆWICZENIA", "EXERCISE": "ĆWICZENIA",
        "LABORATORIUM": "LABORATORIUM", "LAB": "LABORATORIUM",
        "PROJEKT": "PROJEKT", "PROJECT": "PROJEKT",
        "SEMINARIUM": "SEMINARIUM",
    }
    for raw, out in mapping.items():
        if raw in tags_u:
            return out
    s = str(subject or "").upper()
    if " - W" in s: return "WYKŁAD"
    if " - C" in s or " - Ć" in s: return "ĆWICZENIA"
    if " - L" in s: return "LABORATORIUM"
    if " - P" in s: return "PROJEKT"
    if "SEMINARIUM" in s: return "SEMINARIUM"
    return ""


def transform_features_for_learning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURE_COLS:
        if c not in out.columns:
            continue
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
        raise RuntimeError("SHAP not available.")
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if X is None or len(X) == 0:
        raise RuntimeError("No data for SHAP.")
    sample_X = X[: min(len(X), max_samples)]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values[1])
    else:
        shap_arr = np.array(shap_values)
    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    return (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_pair_difference(row_left, row_right) -> pd.DataFrame:
    rows = []
    for feat in FEATURE_COLS:
        lv = float(row_left.get(feat, 0))
        rv = float(row_right.get(feat, 0))
        diff = lv - rv
        direction = DIRECTIONS.get(feat, "cost")
        if direction == "cost":
            preference = "left" if lv < rv else ("right" if rv < lv else "equal")
        else:
            preference = "left" if lv > rv else ("right" if rv > lv else "equal")
        rows.append({
            "feature": feat,
            "left": lv,
            "right": rv,
            "abs_diff": abs(diff),
            "preferred_side": preference,
        })
    return pd.DataFrame(rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)


def build_final_score(df: pd.DataFrame, score_col: str) -> pd.Series:
    scaler = MinMaxScaler()
    vals = scaler.fit_transform(df[[score_col]]).reshape(-1)
    base_score = pd.Series(np.round(vals * 100, 2), index=df.index)
    if "score_std" in df.columns and "weak_ratio" in df.columns:
        penalty_std  = np.tanh(df["score_std"]  / 10.0) * 10
        penalty_weak = df["weak_ratio"] * 15
        return (base_score - penalty_std - penalty_weak).clip(lower=0)
    return base_score


def score_label(v: float) -> str:
    if v >= 85: return "Excellent"
    if v >= 70: return "Very good"
    if v >= 55: return "Good"
    if v >= 40: return "Average"
    return "Weak"


def pair_difficulty_score(x1, x2) -> float:
    diff = np.abs(x1 - x2)
    total = diff.mean()
    concentrated = diff.max() / (diff.sum() + 1e-9)
    return concentrated - 0.35 * total


def estimate_preference_consistency(answers: List[dict]) -> float:
    pairs: Dict[tuple, Optional[bool]] = {}
    for ans in answers:
        choice = ans.get("choice")
        if choice not in ("left", "right"):
            continue
        left_id, right_id = ans["left_id"], ans["right_id"]
        key = tuple(sorted((left_id, right_id)))
        direction = (left_id == key[0] and choice == "left") or (right_id == key[0] and choice == "right")
        if key in pairs:
            if pairs[key] != direction:
                pairs[key] = None
        else:
            pairs[key] = direction
    total = len(pairs)
    if total == 0:
        return 100.0
    consistent = sum(1 for v in pairs.values() if v is not None)
    return (consistent / total) * 100.0


# ── feature normalisation ────────────────────────────────────────────────────
def _normalized_matrix(df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
    global FEATURE_SCALER
    X = transform_features_for_learning(df[FEATURE_COLS]).astype(float)
    if fit_scaler or FEATURE_SCALER is None:
        FEATURE_SCALER = MinMaxScaler()
        FEATURE_SCALER.fit(X)
    return FEATURE_SCALER.transform(X)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SYNTHETIC CANDIDATE GENERATION                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DAYS  = ["Mon", "Tue", "Wed", "Thu", "Fri"]
HOURS = [
    "07:00-08:30", "08:45-10:15", "10:30-12:00",
    "12:15-13:45", "14:00-15:30", "15:45-17:15", "17:30-19:00",
]
ROOMS_C1 = ["A101", "B204", "C301", "D110"]
ROOMS_C2 = ["G101", "H204", "J301", "K110"]
SUBJECTS_BY_TYPE = {
    "WYKŁAD":      ["Analiza - W", "AI - W", "Systemy - W", "Bazy - W"],
    "ĆWICZENIA":   ["Analiza - C", "AI - C", "Programowanie - C", "Sieci - C"],
    "LABORATORIUM":["Analiza - L", "AI Lab - L", "Programowanie - L", "Bazy - L", "Robotyka - L"],
    "PROJEKT":     ["Projekt AI - P", "Projekt BD - P", "Projekt Sys - P"],
}
TEACHER_POOL = ["T001", "T002", "T003", "T004", "T005", "T006"]


def build_synthetic_activity_base(random_state: int = 42) -> List[dict]:
    rng = random.Random(random_state)
    spec = [
        ("LABORATORIUM","ALL"),("LABORATORIUM","ALL"),("LABORATORIUM","ALL"),
        ("LABORATORIUM","ODD"),("LABORATORIUM","EVEN"),
        ("ĆWICZENIA","ALL"),("ĆWICZENIA","ALL"),
        ("PROJEKT","ALL"),("PROJEKT","ALL"),
        ("WYKŁAD","ALL"),
        ("LABORATORIUM","ALL"),("ĆWICZENIA","ALL"),
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


def _make_activity_from_base(base_item, idx, d, sidx, rng, campus):
    room = rng.choice(ROOMS_C1 if campus == "C1" else ROOMS_C2)
    type_name = base_item["type_name"]
    week_tag  = base_item["week_tag"]
    tags = [type_name]
    if type_name == "LABORATORIUM": tags.append("LAB")
    if type_name == "WYKŁAD":       tags.append("LECTURE")
    if type_name == "ĆWICZENIA":    tags.append("EXERCISE")
    if type_name == "PROJEKT":      tags.append("PROJECT")
    if week_tag in {"ODD","EVEN"}:  tags.append(week_tag)
    return {
        "activity_id": f"SYN_{idx}_{base_item['base_id']}_{d}_{sidx}",
        "room": room, "subject": base_item["subject"],
        "comments": "", "tags": tags, "teachers": [base_item["teacher"]],
    }


def generate_synthetic_candidate(idx, base_activities, rng):
    profile = rng.choice([
        "compact","free_day","early","late","many_gaps","campus_mixed",
        "balanced","long_streak","alternating_campus","odd_even_split",
        "good_monday_bad_friday","mixed_types_layout",
    ])
    cell_map = defaultdict(list)
    preferred_days = ["Mon","Tue","Wed","Thu"] if profile == "free_day" else DAYS[:]
    n_items = len(base_activities)
    candidate_slots = []

    if profile == "compact":
        starts = [1,2,2,3,3]
        for d in preferred_days:
            start = rng.choice(starts)
            for x in range(2):
                if start+x < len(HOURS): candidate_slots.append((d,start+x))
    elif profile == "free_day":
        for d in preferred_days:
            start = rng.choice([1,2,3])
            for x in range(3):
                if start+x < len(HOURS): candidate_slots.append((d,start+x))
    elif profile == "early":
        for d in preferred_days:
            for s in [0,1,2]: candidate_slots.append((d,s))
    elif profile == "late":
        for d in preferred_days:
            for s in [3,4,5,6]:
                if s < len(HOURS): candidate_slots.append((d,s))
    elif profile == "many_gaps":
        for d in DAYS:
            for s in [0,2,4,6]:
                if s < len(HOURS): candidate_slots.append((d,s))
    elif profile in {"campus_mixed","alternating_campus"}:
        for d in DAYS:
            for s in [1,2,3]: candidate_slots.append((d,s))
    elif profile == "long_streak":
        for d in DAYS:
            for s in [0,1,2,3,4]: candidate_slots.append((d,s))
    elif profile == "good_monday_bad_friday":
        candidate_slots.extend([("Mon",2),("Mon",3)])
        candidate_slots.extend([("Fri",0),("Fri",2),("Fri",4),("Fri",6)])
        for d in ["Tue","Wed","Thu"]:
            for s in [1,2,3]: candidate_slots.append((d,s))
    elif profile in {"odd_even_split","mixed_types_layout"}:
        for d in DAYS:
            for s in [1,2,4]: candidate_slots.append((d,s))
    else:
        for d in DAYS:
            for s in [1,2,3]: candidate_slots.append((d,s))

    if len(candidate_slots) < n_items:
        for d in DAYS:
            for s in range(len(HOURS)): candidate_slots.append((d,s))

    candidate_slots = list(dict.fromkeys(candidate_slots))
    rng.shuffle(candidate_slots)
    chosen_slots = candidate_slots[:n_items]

    for idx_slot in range(len(chosen_slots)):
        if rng.random() < 0.15:
            new_day = rng.choice(DAYS)
            new_hour_idx = rng.randrange(len(HOURS))
            new_slot = (new_day, new_hour_idx)
            while new_slot in chosen_slots[:idx_slot] + chosen_slots[idx_slot+1:]:
                new_day = rng.choice(DAYS)
                new_hour_idx = rng.randrange(len(HOURS))
                new_slot = (new_day, new_hour_idx)
            chosen_slots[idx_slot] = new_slot

    for k, base_item in enumerate(base_activities):
        d, sidx = chosen_slots[k]
        campus = "C1" if (k % 2 == 0) else "C2" if profile in {"alternating_campus","campus_mixed"} else rng.choice(["C1","C1","C1","C2"])
        act = _make_activity_from_base(base_item, idx, d, sidx, rng, campus)
        cell_map[(d, HOURS[sidx])].append(act)

    return {
        "candidate_id": f"SYNTH::{idx:03d}",
        "run": "SYNTH", "subgroup": f"Synthetic Lab {idx:03d}",
        "profile": profile, "days": DAYS[:], "hours": HOURS[:],
        "cell_map": dict(cell_map), "base_activity_count": len(base_activities),
    }


# ── metrics ─────────────────────────────────────────────────────────────────
def compute_per_subgroup_metrics(table, days, hours, room_to_campus):
    hour_idx = {h: i for i, h in enumerate(hours)}
    gaps1=gaps2p=single_class_days=campus_switch_0=campus_switch_1=0
    long_streak_days=dayoff_count=total_days_with_classes=total_activities=0
    earliest_start_sum=latest_end_sum=daily_span_sum=0
    morning_classes_count=late_classes_count=lab_days=0
    odd_even_imbalance=mixed_type_days=friday_penalty=monday_bonus=0
    multi_campus_days=friday_late_classes=campus_rush_days=0
    daily_loads = []

    for d in days:
        occ = []
        room_by_i = {}
        acts_by_i = defaultdict(list)

        for (dd, hh), acts in (table or {}).items():
            if dd != d or hh not in hour_idx:
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
        latest_end_sum     += occ[-1]
        daily_span_sum     += (occ[-1] - occ[0] + 1)

        if occ[0] <= 1:                          morning_classes_count += 1
        if occ[-1] >= max(0, len(hours) - 2):    late_classes_count += 1

        first_i, last_i = occ[0], occ[-1]
        occ_set = set(occ)
        j = first_i
        while j <= last_i:
            if j in occ_set:
                j += 1; continue
            k = j
            while k <= last_i and k not in occ_set: k += 1
            gap_len = k - j
            if gap_len == 1:   gaps1 += 1
            elif gap_len >= 2: gaps2p += 1
            j = k

        campuses_today = set()
        for a, b in zip(occ, occ[1:]):
            gap_slots = b - a - 1
            r1 = room_by_i.get(a, ""); r2 = room_by_i.get(b, "")
            c1 = room_to_campus.get(r1,"") or room_to_campus_from_room_name(r1)
            c2 = room_to_campus.get(r2,"") or room_to_campus_from_room_name(r2)
            if c1: campuses_today.add(c1)
            if c2: campuses_today.add(c2)
            if c1 and c2 and c1 != c2:
                if gap_slots == 0:
                    campus_switch_0 += 1; campus_rush_days += 1
                elif gap_slots == 1:
                    campus_switch_1 += 1

        if len(campuses_today) >= 2: multi_campus_days += 1

        max_streak = streak = 1
        for a, b in zip(occ, occ[1:]):
            if b == a + 1:
                streak += 1; max_streak = max(max_streak, streak)
            else:
                streak = 1
        if max_streak >= 5: long_streak_days += 1

        day_types = set(); odd_count = even_count = 0
        for i in occ:
            for a in acts_by_i.get(i, []):
                t = pick_type_tag(a.get("tags") or [], a.get("subject") or "")
                if t: day_types.add(t)
                tags_u = [str(x).strip().upper() for x in (a.get("tags") or [])]
                if "ODD"  in tags_u: odd_count  += 1
                if "EVEN" in tags_u: even_count += 1

        if "LABORATORIUM" in day_types: lab_days += 1
        if len(day_types) >= 2:         mixed_type_days += 1
        odd_even_imbalance += abs(odd_count - even_count)

        if d == "Fri":
            friday_penalty     += len(occ) + (2 if occ and occ[-1] >= len(hours)-2 else 0)
            friday_late_classes += sum(1 for x in occ if x >= 4)
        if d == "Mon":
            monday_bonus += max(0, 4 - len(occ))

    dws = max(total_days_with_classes, 1)
    daily_load_variance = float(np.var(daily_loads)) if daily_loads else 0.0

    return {
        "campus_switch_0":    campus_switch_0,
        "campus_switch_1":    campus_switch_1,
        "gaps1":              gaps1,
        "gaps2p":             gaps2p,
        "single_class_days":  single_class_days,
        "long_streak_days":   long_streak_days,
        "dayoff_count":       dayoff_count,
        "days_with_classes":  total_days_with_classes,
        "total_activities":   total_activities,
        "earliest_start_mean": earliest_start_sum / dws,
        "latest_end_mean":     latest_end_sum / dws,
        "daily_span_mean":     daily_span_sum / dws,
        "morning_classes_count": morning_classes_count,
        "late_classes_count": late_classes_count,
        "lab_days":           lab_days,
        "odd_even_imbalance": odd_even_imbalance,
        "mixed_type_days":    mixed_type_days,
        "friday_penalty":     friday_penalty,
        "monday_bonus":       monday_bonus,
        "multi_campus_days":  multi_campus_days,
        "friday_late_classes": friday_late_classes,
        "campus_rush_days":   campus_rush_days,
        "daily_load_variance": daily_load_variance,
    }


def build_synthetic_candidates(n_candidates: int, random_state: int = 42) -> pd.DataFrame:
    rng = random.Random(random_state)
    base_activities = build_synthetic_activity_base(random_state=random_state)
    room_to_campus = {r: "C1" for r in ROOMS_C1}
    room_to_campus.update({r: "C2" for r in ROOMS_C2})
    rows = []
    for i in range(n_candidates):
        cand    = generate_synthetic_candidate(i + 1, base_activities, rng)
        metrics = compute_per_subgroup_metrics(
            table=cand["cell_map"], days=cand["days"],
            hours=cand["hours"], room_to_campus=room_to_campus,
        )
        row = {
            "candidate_id": cand["candidate_id"], "run": cand["run"],
            "subgroup": cand["subgroup"], "profile": cand["profile"],
            "days": cand["days"], "hours": cand["hours"],
            "cell_map": cand["cell_map"],
            "base_activity_count": cand["base_activity_count"],
        }
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "total_activities" in df.columns and df["total_activities"].nunique() != 1:
        raise RuntimeError("Synthetic candidates have different activity counts.")
    return df


# ── pair selection ────────────────────────────────────────────────────────────
def select_diverse_candidates(df: pd.DataFrame, n_select: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= n_select:
        return df.copy().reset_index(drop=True)
    X = transform_features_for_learning(df[FEATURE_COLS]).astype(float)
    Xn = MinMaxScaler().fit_transform(X)
    rng = random.Random(random_state)
    chosen = [rng.randrange(len(df))]
    remaining = set(range(len(df))) - set(chosen)
    while len(chosen) < n_select and remaining:
        best_i, best_score = None, -1.0
        for i in remaining:
            dmin = min(np.linalg.norm(Xn[i] - Xn[j]) for j in chosen)
            if dmin > best_score:
                best_score = dmin; best_i = i
        chosen.append(best_i); remaining.remove(best_i)
    return df.iloc[chosen].reset_index(drop=True)


def generate_extreme_pairs(df: pd.DataFrame, n_pairs: int) -> List[Tuple[int, int]]:
    Xn = _normalized_matrix(df, fit_scaler=True)
    out, used = [], set()
    for group_name, cols in PAIR_GROUPS.items():
        valid_cols = [c for c in cols if c in FEATURE_COLS]
        if not valid_cols: continue
        grp_idx   = [FEATURE_COLS.index(c) for c in valid_cols]
        grp_score = Xn[:, grp_idx].mean(axis=1) * GROUP_IMPORTANCE.get(group_name, 1.0)
        i_best    = int(np.argmax(grp_score))
        i_worst   = int(np.argmin(grp_score))
        if i_best != i_worst:
            key = tuple(sorted((i_best, i_worst)))
            if key not in used:
                out.append(key); used.add(key)
                if len(out) >= n_pairs: return out
    return out[:n_pairs]


def generate_tradeoff_pairs(df: pd.DataFrame, n_pairs: int) -> List[Tuple[int, int]]:
    Xn = _normalized_matrix(df)
    candidates = sorted(
        [(i, j, pair_difficulty_score(Xn[i], Xn[j]))
         for i in range(len(df)) for j in range(i+1, len(df))],
        key=lambda x: x[2], reverse=True,
    )
    out, used = [], set()
    for i, j, _ in candidates:
        key = tuple(sorted((i, j)))
        if key not in used:
            used.add(key); out.append(key)
            if len(out) >= n_pairs: break
    return out


def generate_initial_pairs(df: pd.DataFrame, n_pairs_total: int) -> List[Tuple[int, int]]:
    if len(df) < 2: return []
    _normalized_matrix(df, fit_scaler=True)
    q1 = max(2, n_pairs_total // 3)
    q2 = max(2, n_pairs_total // 3)
    q3 = max(0, n_pairs_total - q1 - q2)
    p1 = generate_extreme_pairs(df, q1)
    p2 = generate_tradeoff_pairs(df, q2)
    used = set(tuple(sorted(p)) for p in p1 + p2)
    Xn = _normalized_matrix(df)
    all_pairs = [(i, j, float(np.linalg.norm(Xn[i]-Xn[j])))
                 for i in range(len(df)) for j in range(i+1, len(df))
                 if (i, j) not in used]
    if all_pairs:
        med = statistics.median(x[2] for x in all_pairs)
        all_pairs.sort(key=lambda x: abs(x[2] - med))
    rest = [(i, j) for i, j, _ in all_pairs[:q3]]
    out, seen = [], set()
    for p in p1 + p2 + rest:
        key = tuple(sorted(p))
        if key not in seen:
            out.append(key); seen.add(key)
        if len(out) >= n_pairs_total: break
    return out[:n_pairs_total]


# ── aggregation + run ranking ─────────────────────────────────────────────────
def aggregate_run_scores(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for run_name, g in df.groupby("run"):
        vals = sorted(g[score_col].dropna().tolist(), reverse=True)
        if not vals: continue
        best_row = g.sort_values(score_col, ascending=False).iloc[0]
        rows.append({
            "run": run_name,
            "best_subgroup": best_row["subgroup"],
            "best_subgroup_score": vals[0],
            "top3_mean_score": float(np.mean(vals[:3])),
            "median_score": float(np.median(vals)),
            "score_std": float(np.std(vals)),
            "weak_ratio": float(np.mean(np.array(vals) < np.median(vals))),
            "subgroups_count": len(g),
        })
    return pd.DataFrame(rows).sort_values("best_subgroup_score", ascending=False).reset_index(drop=True)


# ── timetable rendering ────────────────────────────────────────────────────────
def dedup_tiles(tiles):
    seen, out = set(), []
    for a in tiles:
        key = (a.get("activity_id",""), a.get("subject",""), a.get("room",""),
               tuple(a.get("teachers") or []), tuple(a.get("tags") or []))
        if key not in seen:
            seen.add(key); out.append(a)
    return out


def render_grid_html(days, hours, cell_map, title) -> str:
    css = """
    <style>
      .tt-wrap{width:100%;overflow-x:auto}
      table.tt{border-collapse:collapse;width:100%;min-width:720px;table-layout:fixed}
      table.tt th,table.tt td{border:1px solid #ddd;vertical-align:top;padding:5px}
      table.tt th{background:#f6f7f9;font-weight:700;text-align:center}
      table.tt td{height:95px;background:#fff}
      .hour{width:120px;background:#fafafa;font-weight:700}
      .tile{border-radius:8px;padding:5px 6px;margin:4px 0;border:1px solid rgba(0,0,0,.10)}
      .tile .subj{font-weight:700;font-size:12px;margin-bottom:3px}
      .tile .meta{font-size:11px;opacity:.86;line-height:1.2}
      .badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:999px;background:rgba(0,0,0,.07);margin-right:5px}
      .WYKŁAD{background:#eef6ff}.ĆWICZENIA{background:#f3f7ee}
      .LABORATORIUM{background:#fff4e6}.PROJEKT{background:#f5efff}.SEMINARIUM{background:#fbefff}
      .tt-title{font-size:20px;font-weight:800;margin:8px 0 12px 0}
    </style>"""
    html = [css, f"<div class='tt-title'>{title}</div>",
            "<div class='tt-wrap'>", "<table class='tt'>", "<tr>",
            "<th class='hour'>Godzina</th>"]
    html += [f"<th>{d}</th>" for d in days]
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
                if "ODD"  in tags_u: week_tag = "ODD"
                elif "EVEN" in tags_u: week_tag = "EVEN"
                cls = type_tag if type_tag in {"WYKŁAD","ĆWICZENIA","LABORATORIUM","PROJEKT","SEMINARIUM"} else ""
                badges = []
                if type_tag: badges.append(f"<span class='badge'>{type_tag}</span>")
                if week_tag: badges.append(f"<span class='badge'>{week_tag}</span>")
                subj = str(a.get("subject") or "").strip() or "Zajęcia"
                teachers = ", ".join(a.get("teachers") or [])
                room = str(a.get("room") or "").strip()
                meta_parts = []
                if teachers: meta_parts.append(f"Prow.: {teachers}")
                if room:     meta_parts.append(f"Sala: {room}")
                meta = "<br/>".join(meta_parts)
                cell.append(
                    f"<div class='tile {cls}'>"
                    f"<div class='subj'>{''.join(badges)}{subj}</div>"
                    f"<div class='meta'>{meta}</div>"
                    f"</div>"
                )
            html.append("<td>" + "".join(cell) + "</td>")
        html.append("</tr>")
    html.append("</table></div>")
    return "\n".join(html)


# ── real candidates from session ──────────────────────────────────────────────
def list_sessions(out_root: Path) -> List[Path]:
    if not out_root.exists(): return []
    return sorted([p for p in out_root.iterdir() if p.is_dir()], reverse=True)


def load_session_summary(root: Path) -> dict:
    p = root / "generation_summary.json"
    obj = load_json(p) if p.exists() else None
    return obj or {}


def get_run_dirs_from_session(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def get_rankable_runs(root: Path) -> List[Path]:
    summary = load_session_summary(root)
    runs_meta = summary.get("runs") or []
    by_name = {p.name: p for p in get_run_dirs_from_session(root)}
    rankable = [
        by_name[r["run_name"]]
        for r in runs_meta
        if r.get("returncode") == 0 and r.get("ranking_ready") and r.get("run_name") in by_name
    ]
    return rankable or get_run_dirs_from_session(root)


def find_run_subgroups_xml(run_dir: Path) -> Optional[Path]:
    p = run_dir / "instance_subgroups.xml"
    if p.exists(): return p
    cands = sorted(run_dir.rglob("*subgroups*.xml"))
    return cands[0] if cands else None


def find_input_fet_for_session(root: Path, last_gen_path: Path) -> Optional[Path]:
    p = root / "input_fet_info.json"
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            fp = obj.get("file_path")
            if fp and Path(fp).exists(): return Path(fp)
        except Exception: pass
    summary = load_session_summary(root)
    fp = summary.get("input_fet")
    if fp and Path(fp).exists(): return Path(fp)
    last = load_json(last_gen_path) or {}
    fp = last.get("input_fet")
    if fp and Path(fp).exists(): return Path(fp)
    return None


def parse_input_fet(fet_path_str: str):
    tree = ET.parse(Path(fet_path_str))
    root = tree.getroot()
    room_to_building = {}
    rooms_list = root.find("Rooms_List")
    if rooms_list is not None:
        for r in rooms_list.findall("Room"):
            rid = first_text(r, ["Name"], "")
            bid = first_text(r, ["Building"], "")
            if rid: room_to_building[rid] = bid
    room_to_campus = {rid: building_to_campus(bid) for rid, bid in room_to_building.items()}
    act_idx = {}
    acts = root.find("Activities_List")
    if acts is not None:
        for a in acts.findall("Activity"):
            aid = first_text(a, ["Id","Activity_Id"],"") or attr_any(a, ["Id","Activity_Id","id"],"")
            act_idx[aid] = {
                "subject":  first_text(a, ["Subject"], ""),
                "teachers": [str(x.text).strip() for x in a.findall("Teacher") if x.text and str(x.text).strip()],
                "tags":     [str(x.text).strip() for x in a.findall("Activity_Tag") if x.text and str(x.text).strip()],
                "comments": first_text(a, ["Comments"], ""),
            }
    return {"room_to_campus": room_to_campus, "activity_index": act_idx}


def parse_subgroups_xml_full(xml_path_str: str):
    tree = ET.parse(Path(xml_path_str))
    root = tree.getroot()
    schedule = defaultdict(lambda: defaultdict(list))
    days_seen = []; hours_seen = []; subgroups_seen = []

    for sub in root.findall(".//Subgroup"):
        sname = name_of(sub)
        if not sname: continue
        subgroups_seen.append(sname)
        for day_node in sub.findall("./Day"):
            dname = name_of(day_node)
            if not dname: continue
            if dname not in days_seen: days_seen.append(dname)
            for hour_node in day_node.findall("./Hour"):
                hname = name_of(hour_node)
                if not hname: continue
                if hname not in hours_seen: hours_seen.append(hname)
                act_nodes = hour_node.findall("./Activity")
                if not act_nodes: continue

                def _text_or_attr(node, attr_keys):
                    if node is None: return ""
                    for k in attr_keys:
                        v = node.attrib.get(k)
                        if v: return str(v).strip()
                    return (node.text or "").strip()

                room = _text_or_attr(hour_node.find("./Room"), ["name","Name"])
                subject = _text_or_attr(hour_node.find("./Subject"), ["name","Name"])
                teachers = [_text_or_attr(tn, ["name","Name"]) for tn in hour_node.findall("./Teacher")]
                teachers = [t for t in teachers if t]
                tags = [_text_or_attr(tg, ["name","Name"]) for tg in hour_node.findall("./Activity_Tag")]
                tags = [t for t in tags if t]

                for act in act_nodes:
                    aid = (act.attrib.get("id") or act.attrib.get("Id") or
                           first_text(act, ["Activity_Id","Id",".//Activity_Id",".//Id"],""))
                    schedule[sname][(dname, hname)].append({
                        "activity_id": str(aid).strip(),
                        "room": room, "subject": subject,
                        "comments": "", "tags": tags[:], "teachers": teachers[:],
                    })

    return {
        "days": days_seen, "hours": hours_seen,
        "subgroups": sorted(set(subgroups_seen)),
        "schedule": {sg: dict(cm) for sg, cm in schedule.items()},
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
                        merged["subject"] = ref.get("subject","")
                    if not (merged.get("teachers") or []):
                        merged["teachers"] = ref.get("teachers",[])
                    tags = list(merged.get("tags") or [])
                    for tg in (ref.get("tags") or []):
                        if tg not in tags: tags.append(tg)
                    merged["tags"] = tags
                    if not str(merged.get("comments") or "").strip():
                        merged["comments"] = ref.get("comments","")
                out[sg][key].append(merged)
    return out


# ── CSV helpers (identyczna logika jak w 04_Ranking.py) ──────────────────────

def _csv_base_prefix(key: str) -> str:
    return key.split("-")[0]

def _csv_direction_from_prefix(prefix: str) -> str:
    return "".join(c for c in prefix if c.isalpha())

def _csv_parse_student_key(val: str):
    """Zwraca (direction, year, raw_key, level)."""
    raw       = str(val).split()[0]
    prefix    = _csv_base_prefix(raw)
    direction = _csv_direction_from_prefix(prefix)
    year      = prefix + "-W1"
    import re as _re
    if   _re.search(r'-L\d', raw): level = "LAB"
    elif _re.search(r'-C\d', raw): level = "GROUP"
    else:                           level = "YEAR"
    return direction, year, raw, level

def _csv_extract_structure(df: pd.DataFrame):
    import re as _re
    year_map  = defaultdict(set)
    group_map = defaultdict(set)
    col = "Students Sets" if "Students Sets" in df.columns else "Students"
    if col not in df.columns:
        return year_map, group_map
    for val in df[col].dropna().unique():
        raw = str(val).strip().split()[0]
        if _re.search(r'-L\d', raw):
            group = _re.sub(r'-L\d+', '-C1', raw)
            group_map[group].add(raw)
        elif _re.search(r'-C\d', raw):
            year = _csv_base_prefix(raw) + "-W1"
            year_map[year].add(raw)
    for group in list(group_map.keys()):
        year = _csv_base_prefix(group) + "-W1"
        year_map[year].add(group)
    return year_map, group_map

def _csv_ancestry(key: str, level: str, lab_to_group: dict, group_to_year: dict) -> list:
    chain = [key]
    if level == "LAB":
        grp = lab_to_group.get(key)
        if grp:
            chain.append(grp)
            yr = group_to_year.get(grp)
            if yr:
                chain.append(yr)
    elif level == "GROUP":
        yr = group_to_year.get(key)
        if yr:
            chain.append(yr)
    return chain

def _find_timetable_csv(run_dir: Path) -> Optional[Path]:
    hits = list(run_dir.rglob("*timetable*.csv"))
    if not hits:
        return None
    for h in hits:
        if "highest" in str(h):
            return h
    return hits[0]


def build_real_candidates(root: Path, last_gen_path: Path) -> pd.DataFrame:
    """
    Buduje DataFrame kandydatów z plików CSV — identyczna logika jak 04_Ranking.py.
    Dodaje kolumny: direction, year, level (potrzebne do filtrowania w sekcji 8).
    Zachowuje stary XML jako fallback gdy brak CSV.
    """
    meta = load_json(last_gen_path) or {}

    # runs_ok może być listą nazw lub int (licznik)
    runs_ok_raw = meta.get("runs_ok", [])
    if isinstance(runs_ok_raw, list) and runs_ok_raw:
        run_names = runs_ok_raw
    else:
        run_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()]

    rows = []
    for run_name in run_names:
        run_dir  = root / run_name
        if not run_dir.is_dir():
            continue
        csv_path = _find_timetable_csv(run_dir)
        if not csv_path:
            continue

        df = pd.read_csv(csv_path, sep=",", encoding="utf-8")
        df.columns = [c.strip() for c in df.columns]
        col_stud = "Students Sets" if "Students Sets" in df.columns else "Students"
        if col_stud not in df.columns:
            continue

        days_order:  List[str] = []
        hours_order: List[str] = []
        raw: Dict[str, Dict[tuple, list]] = defaultdict(lambda: defaultdict(list))

        for _, row in df.iterrows():
            d    = str(row.get("Day",    "")).strip()
            h    = str(row.get("Hour",   "")).strip()
            stud = str(row.get(col_stud, "")).strip()
            room = str(row.get("Room",   "")).strip()
            subj = str(row.get("Subject","")).strip()
            tags = str(row.get("Activity Tags", "")).strip()
            teach = str(row.get("Teachers", "")).strip()
            if not stud or stud == "nan" or not d or not h:
                continue
            stud = stud.split()[0]
            if d not in days_order:  days_order.append(d)
            if h not in hours_order: hours_order.append(h)
            tags_list  = [t.strip() for t in tags.split(",")  if t.strip()]
            teach_list = [t.strip() for t in teach.split(",") if t.strip()]
            raw[stud][(d, h)].append({
                "room": room, "subject": subj,
                "tags": tags_list, "teachers": teach_list,
            })

        year_map, group_map = _csv_extract_structure(df)
        lab_to_group  = {lab: grp for grp, labs in group_map.items() for lab in labs}
        group_to_year = {grp: yr  for yr, grps in year_map.items()   for grp in grps}

        aggregated: Dict[str, Dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
        for key in set(raw.keys()):
            _, _, raw_key, level = _csv_parse_student_key(key)
            ancestors = _csv_ancestry(raw_key, level, lab_to_group, group_to_year)
            for ancestor in ancestors:
                for slot, acts in raw.get(ancestor, {}).items():
                    aggregated[key][slot].extend(acts)

        cid = 0
        for key in raw.keys():
            direction, year, _, level = _csv_parse_student_key(key)
            metrics = compute_per_subgroup_metrics(
                table=dict(aggregated[key]),
                days=days_order,
                hours=hours_order,
                room_to_campus={},   # fallback: dedukuje z nazwy sali
            )
            metrics.update({
                "candidate_id": f"{run_name}__{key}__{cid}",
                "run":          run_name,
                "subgroup":     key,
                "direction":    direction,
                "year":         year,
                "level":        level,
                "profile":      "REAL",
                "days":         days_order,
                "hours":        hours_order,
                "cell_map":     dict(aggregated[key]),
            })
            rows.append(metrics)
            cid += 1

    if not rows:
        raise RuntimeError(
            "Nie znaleziono plików CSV z rozkładami. "
            "Upewnij się, że w folderach runów są pliki *timetable*.csv."
        )

    df_out = pd.DataFrame(rows)
    return df_out.drop_duplicates(subset=["candidate_id"]).reset_index(drop=True)

def build_real_candidates_from_db(df: pd.DataFrame):
    groups = []

    for subgroup, g in df.groupby("subgroup"):
        cell_map = {}

        for _, row in g.iterrows():
            key = (row["day"], row["hour"])

            if key not in cell_map:
                cell_map[key] = []

            cell_map[key].append({
                "subject": row["subject"],
                "teachers": [row["teacher"]],
                "room": row["room"],
                "tags": []
            })

        groups.append({
            "subgroup": subgroup,
            "cell_map": cell_map,
            "days": sorted(df["day"].unique()),
            "hours": sorted(df["hour"].unique())
        })

    return pd.DataFrame(groups)