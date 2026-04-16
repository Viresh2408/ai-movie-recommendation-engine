"""
╔══════════════════════════════════════════════════════════════════╗
║         CineAI — Premium AI Movie Recommendation Engine          ║
║  Stack: Streamlit • TF-IDF • SVD • XGBoost • CatBoost • TMDB    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import random
import os
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI — Smart Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
TMDB_API_KEY   = "7b995d3c6fd91a2284b4ad8cb390c7b8"
TMDB_BASE      = "https://api.themoviedb.org/3"
TMDB_IMG       = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER    = "https://placehold.co/300x450/0a0a0f/555555?text=🎬+No+Poster"

MODEL_INFO = {
    "🎯 Cosine Similarity": {
        "key": "cosine",
        "desc": "TF-IDF vectors + Cosine distance. Fast, interpretable baseline.",
        "badge": "BASELINE",
        "color": "#4488ff",
    },
    "🧠 SVD / Latent Semantic": {
        "key": "svd",
        "desc": "300-dim latent space via Truncated SVD. Captures hidden themes.",
        "badge": "SEMANTIC",
        "color": "#44cc88",
    },
    "⚡ XGBoost Re-ranker": {
        "key": "xgb",
        "desc": "Gradient boosting on 6 pairwise similarity features. Non-linear.",
        "badge": "ML",
        "color": "#ffaa00",
    },
    "🐱 CatBoost Re-ranker": {
        "key": "catboost",
        "desc": "Yandex's ordered boosting. Best at handling structured features.",
        "badge": "ML+",
        "color": "#ff4488",
    },
}

MOOD_TAGS = {
    "🎭 All Genres":         [],
    "😊 Feel-Good":          ["comedi", "animat", "famili", "music", "adventur"],
    "😱 Thriller & Horror":  ["horror", "thriller", "mysteri", "crime", "suspens"],
    "💔 Romance & Drama":    ["romanc", "drama", "love", "relationship"],
    "🚀 Epic & Sci-Fi":      ["sciencefict", "action", "adventur", "fantasi", "superhero"],
    "🧠 Thought-Provoking":  ["documentari", "histori", "biographi", "psycholog"],
    "😂 Pure Comedy":        ["comedi", "humor", "satir", "parodi"],
}

# ─────────────────────────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────────────────────────
PREMIUM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── GLOBAL ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #070711 !important;
    color: #e4e4f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Cinema grain texture */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 100% !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #070711; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #E50914, #8b0000); border-radius: 10px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0c1e 0%, #0e0716 100%) !important;
    border-right: 1px solid rgba(229, 9, 20, 0.12) !important;
}
[data-testid="stSidebar"] > div { padding: 1rem 0.75rem !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown p  { color: #e4e4f0 !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #E50914 0%, #a00008 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82em !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 35px rgba(229, 9, 20, 0.45) !important;
    background: linear-gradient(135deg, #ff1a27 0%, #cc000c 100%) !important;
}
.stButton > button:active { transform: translateY(-1px) !important; }

/* ── SELECT BOX ── */
[data-testid="stSelectbox"] > label { color: rgba(220,220,240,0.7) !important; font-size: 0.85em !important; }
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(229,9,20,0.22) !important;
    border-radius: 10px !important;
    color: #e4e4f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:hover { border-color: rgba(229,9,20,0.5) !important; }

/* ── RADIO ── */
[data-testid="stRadio"] label { color: #e4e4f0 !important; font-size: 0.9em !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: rgba(200,200,220,0.65) !important; font-size: 0.78em !important; }

/* ── SLIDER ── */
[data-testid="stSlider"] > label { color: rgba(220,220,240,0.7) !important; font-size: 0.85em !important; }
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stTickBarMin"] { color: rgba(220,220,240,0.4) !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.015) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    margin-top: 4px !important;
}
[data-testid="stExpander"] summary { color: rgba(200,200,230,0.75) !important; font-size: 0.85em !important; }

/* ── METRIC ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.3s ease !important;
}
[data-testid="stMetric"]:hover { border-color: rgba(229,9,20,0.35) !important; }
[data-testid="stMetric"] label { color: rgba(200,200,230,0.6) !important; font-size: 0.8em !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
[data-testid="stMetricValue"] { color: #E50914 !important; font-family: 'Cinzel', serif !important; font-weight: 700 !important; font-size: 1.5em !important; }
[data-testid="stMetricDelta"] { color: #44cc88 !important; font-size: 0.8em !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(200,200,230,0.6) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88em !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(229,9,20,0.25), rgba(229,9,20,0.1)) !important;
    color: #E50914 !important;
    border: 1px solid rgba(229,9,20,0.3) !important;
}

/* ── DIVIDER ── */
hr { border-color: rgba(229,9,20,0.15) !important; margin: 1.5rem 0 !important; }

/* ── TEXT INPUT ── */
[data-testid="stTextInput"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(229,9,20,0.22) !important;
    border-radius: 10px !important;
    color: #e4e4f0 !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: #E50914 !important; }

/* ══════════════════════════════════════════════════════════════
   KEYFRAME ANIMATIONS
══════════════════════════════════════════════════════════════ */
@keyframes heroGlow {
    0%, 100% { opacity: 0.4; transform: translate(-50%,-50%) scale(1); }
    50%       { opacity: 0.9; transform: translate(-50%,-50%) scale(1.35); }
}
@keyframes shimmer {
    0%   { background-position: 0%   center; }
    100% { background-position: 200% center; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0px); }
}
@keyframes floatIcon {
    0%, 100% { transform: translateY(0px)  rotate(-6deg); }
    50%       { transform: translateY(-14px) rotate(6deg); }
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(229,9,20,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(229,9,20,0); }
}
@keyframes borderDance {
    0%, 100% { border-color: rgba(229,9,20,0.2); }
    50%       { border-color: rgba(229,9,20,0.55); }
}

/* ══════════════════════════════════════════════════════════════
   MOVIES CARDS (custom HTML)
══════════════════════════════════════════════════════════════ */
.movie-card {
    position: relative;
    border-radius: 15px;
    overflow: hidden;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    transition: all 0.38s cubic-bezier(0.4,0,0.2,1);
    cursor: pointer;
    margin-bottom: 6px;
    animation: fadeInUp 0.55s ease both;
}
.movie-card:hover {
    transform: translateY(-11px) scale(1.025);
    border-color: rgba(229,9,20,0.55);
    box-shadow:
        0 28px 60px rgba(229,9,20,0.22),
        0 0 0 1px rgba(229,9,20,0.1),
        inset 0 0 30px rgba(229,9,20,0.04);
}
.card-glow {
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(229,9,20,0.06), transparent 70%);
    opacity: 0;
    transition: opacity 0.38s ease;
    pointer-events: none;
    z-index: 2;
}
.movie-card:hover .card-glow { opacity: 1; }

/* Card rank/trending badges     */
.rank-badge {
    position: absolute;
    top: 9px; right: 9px;
    background: rgba(0,0,0,0.75);
    border: 1px solid rgba(255,215,0,0.55);
    color: #FFD700;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center; justify-content: center;
    font-size: 0.72em;
    font-weight: 700;
    font-family: 'Cinzel', serif;
    z-index: 10;
    backdrop-filter: blur(6px);
}
.hot-badge {
    position: absolute;
    top: 9px; left: 9px;
    background: linear-gradient(135deg, #E50914, #ff6b00);
    color: #fff;
    padding: 3px 9px;
    border-radius: 5px;
    font-size: 0.6em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    z-index: 10;
}
.genre-pill {
    display: inline-block;
    background: rgba(229,9,20,0.12);
    border: 1px solid rgba(229,9,20,0.3);
    color: #ff8888;
    padding: 2px 7px;
    border-radius: 20px;
    font-size: 0.6em;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin: 1px 1px;
}

/* ══════════════════════════════════════════════════════════════
   WATCHLIST ITEMS
══════════════════════════════════════════════════════════════ */
.wl-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 11px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 9px;
    margin: 4px 0;
    font-size: 0.82em;
    color: #c8c8e0;
    transition: border-color 0.2s ease;
}
.wl-item:hover { border-color: rgba(229,9,20,0.35); }
.wl-dot { width: 7px; height: 7px; border-radius: 50%; background: #E50914; margin-right: 8px; }

/* ══════════════════════════════════════════════════════════════
   SECTION HEADERS
══════════════════════════════════════════════════════════════ */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(229,9,20,0.18);
    margin: 2rem 0 1.2rem;
}
.section-hdr-title {
    font-family: 'Cinzel', serif;
    font-size: 1.45em;
    font-weight: 700;
    color: #fff;
    margin: 0;
}
.section-badge {
    background: linear-gradient(135deg, #E50914, #8b0000);
    color: #fff;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.65em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ══════════════════════════════════════════════════════════════
   HISTORY CHIPS
══════════════════════════════════════════════════════════════ */
.hist-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 0.78em;
    color: rgba(210,210,235,0.75);
    cursor: pointer;
    transition: all 0.2s ease;
}
.hist-chip:hover { background: rgba(229,9,20,0.1); border-color: rgba(229,9,20,0.35); color: #ff8888; }

/* ══════════════════════════════════════════════════════════════
   METRICS PANEL
══════════════════════════════════════════════════════════════ */
.metrics-panel {
    background: rgba(255,255,255,0.018);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.metrics-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.metrics-row:last-child { border-bottom: none; }
.metrics-model { font-weight: 600; color: #dde; font-size: 0.9em; }
.metrics-val   { font-family: 'Cinzel', serif; color: #E50914; font-weight: 700; font-size: 0.95em; }
.bar-bg { background: rgba(255,255,255,0.06); border-radius: 4px; height: 5px; flex: 1; margin: 0 12px; }
.bar-fill { background: linear-gradient(90deg, #E50914, #ff4444); border-radius: 4px; height: 5px; transition: width 0.8s ease; }

/* Active model card glow in sidebar */
.active-model {
    border: 1px solid rgba(229,9,20,0.5) !important;
    background: rgba(229,9,20,0.08) !important;
    animation: borderDance 3s ease-in-out infinite;
}

/* Stagger card animations */
.c0 { animation-delay: 0.05s; }
.c1 { animation-delay: 0.12s; }
.c2 { animation-delay: 0.19s; }
.c3 { animation-delay: 0.26s; }
.c4 { animation-delay: 0.33s; }
.c5 { animation-delay: 0.40s; }
.c6 { animation-delay: 0.47s; }
.c7 { animation-delay: 0.54s; }
.c8 { animation-delay: 0.61s; }
.c9 { animation-delay: 0.68s; }
"""

st.markdown(f"<style>{PREMIUM_CSS}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
for key, val in {
    "watchlist": [],
    "history":   [],
    "selected":  None,
    "model_key": "cosine",
    "mood":      "🎭 All Genres",
    "n_recs":    10,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────
# DATA LOADING  (cached across reruns)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Loading AI models…")
def load_all_models():
    data = {}

    # Cosine
    movies, cosine_sim = pickle.load(open("moviedata.pkl", "rb"))
    data["movies"]     = movies
    data["cosine_sim"] = cosine_sim

    # SVD
    try:
        _, svd_sim        = pickle.load(open("moviedata_svd.pkl", "rb"))
        data["svd_sim"]   = svd_sim
        data["svd_matrix"]= pickle.load(open("svd_matrix.pkl",    "rb"))
    except Exception:
        data["svd_sim"]   = cosine_sim
        data["svd_matrix"]= None

    # XGBoost
    try:
        data["xgb"] = pickle.load(open("model_xgb.pkl", "rb"))
    except Exception:
        data["xgb"] = None

    # CatBoost
    try:
        data["catboost"] = pickle.load(open("model_catboost.pkl", "rb"))
    except Exception:
        data["catboost"] = None

    # Metrics
    try:
        data["metrics"] = pickle.load(open("model_metrics.pkl", "rb"))
    except Exception:
        data["metrics"] = {}

    return data


def get_data():
    return load_all_models()

# ─────────────────────────────────────────────────────────────────
# FEATURE BUILDER (mirrors train_models.py)
# ─────────────────────────────────────────────────────────────────
def build_features(anchor_idx, candidate_indices, cosine_sim, svd_norm, movies_df):
    anchor_tags = set(movies_df["tags"].iloc[anchor_idx].split())
    anchor_len  = max(len(anchor_tags), 1)
    features    = []
    for cand in candidate_indices:
        cand_tags = set(movies_df["tags"].iloc[cand].split())
        cand_len  = max(len(cand_tags), 1)
        inter     = len(anchor_tags & cand_tags)
        union     = len(anchor_tags | cand_tags)
        features.append([
            float(cosine_sim[anchor_idx, cand]),
            float(np.dot(svd_norm[anchor_idx], svd_norm[cand])) if svd_norm is not None else 0.0,
            inter / union if union > 0 else 0.0,
            inter / max(anchor_len, cand_len),
            min(anchor_len, cand_len) / max(anchor_len, cand_len),
            abs(anchor_idx - cand) / len(movies_df),
        ])
    return np.array(features, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────
def get_recommendations(title, model_key="cosine", mood="🎭 All Genres", n=10):
    d          = get_data()
    movies_df  = d["movies"]
    cosine_sim = d["cosine_sim"]
    svd_sim    = d["svd_sim"]
    xgb        = d["xgb"]
    catboost   = d["catboost"]
    svd_norm   = d["svd_matrix"]

    try:
        idx = movies_df[movies_df["title"] == title].index[0]
    except IndexError:
        return pd.DataFrame()

    # Choose similarity matrix / re-ranker
    if model_key == "svd":
        sims      = svd_sim[idx]
        top_cands = np.argsort(sims)[::-1][1 : n * 6 + 1]
        movie_idx = top_cands[:n * 4]

    elif model_key == "xgb" and xgb is not None:
        top60  = np.argsort(cosine_sim[idx])[::-1][1:61]
        feats  = build_features(idx, top60, cosine_sim, svd_norm, movies_df)
        scores = xgb.predict(feats)
        ranked = [top60[i] for i in np.argsort(scores)[::-1]]
        movie_idx = ranked[:n * 4]

    elif model_key == "catboost" and catboost is not None:
        top60  = np.argsort(cosine_sim[idx])[::-1][1:61]
        feats  = build_features(idx, top60, cosine_sim, svd_norm, movies_df)
        scores = catboost.predict(feats)
        ranked = [top60[i] for i in np.argsort(scores)[::-1]]
        movie_idx = ranked[:n * 4]

    else:  # cosine (default)
        sims      = cosine_sim[idx]
        top_cands = np.argsort(sims)[::-1][1 : n * 6 + 1]
        movie_idx = top_cands[:n * 4]

    recs = movies_df[["title", "movie_id", "tags"]].iloc[movie_idx].copy()

    # Mood filtering
    kws = MOOD_TAGS.get(mood, [])
    if kws:
        mask = recs["tags"].apply(lambda t: any(k in str(t) for k in kws))
        filtered = recs[mask]
        recs = filtered if len(filtered) >= max(n // 2, 3) else recs

    return recs.head(n).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────
# TMDB API  (cached 1h per movie)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_movie_details(movie_id: int) -> dict:
    try:
        url  = f"{TMDB_BASE}/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return {}
        data    = resp.json()
        cast    = data.get("credits", {}).get("cast", [])
        crew    = data.get("credits", {}).get("crew", [])
        director= next((c["name"] for c in crew if c.get("job") == "Director"), "N/A")
        genres  = [g["name"] for g in data.get("genres", [])]
        poster  = data.get("poster_path")
        return {
            "poster":    f"{TMDB_IMG}{poster}" if poster else PLACEHOLDER,
            "year":      data.get("release_date", "")[:4],
            "rating":    round(data.get("vote_average", 0), 1),
            "votes":     data.get("vote_count", 0),
            "genres":    genres,
            "overview":  data.get("overview", ""),
            "runtime":   data.get("runtime", 0),
            "cast":      [c["name"] for c in cast[:5]],
            "director":  director,
            "tagline":   data.get("tagline", ""),
        }
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_trending() -> list:
    try:
        url  = f"{TMDB_BASE}/trending/movie/day?api_key={TMDB_API_KEY}"
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return []
        results = resp.json().get("results", [])[:10]
        return [
            {
                "title":   r.get("title", ""),
                "movie_id": r.get("id", 0),
                "rating":  round(r.get("vote_average", 0), 1),
                "year":    r.get("release_date", "")[:4],
                "poster":  f"{TMDB_IMG}{r['poster_path']}" if r.get("poster_path") else PLACEHOLDER,
                "overview":r.get("overview", ""),
            }
            for r in results
        ]
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────────
# HTML CARD BUILDER
# ─────────────────────────────────────────────────────────────────
def card_html(poster, title, rating, year, genres, rank=None, is_trending=False, anim_cls="c0"):
    genre_pills = "".join(
        f'<span class="genre-pill">{g[:14]}</span>'
        for g in (genres or [])[:3]
    )
    rank_badge = (
        f'<div class="rank-badge">#{rank}</div>' if rank else ""
    )
    hot_badge = (
        '<div class="hot-badge">🔥 HOT</div>' if is_trending else ""
    )
    short_title = title[:28] + "…" if len(title) > 28 else title

    return f"""
<div class="movie-card {anim_cls}" style="position:relative;">
  <div class="card-glow"></div>
  {rank_badge}{hot_badge}
  <div style="position:relative;">
    <img src="{poster}"
         style="width:100%; display:block; object-fit:cover; border-radius:13px 13px 0 0; aspect-ratio:2/3;"
         onerror="this.src='{PLACEHOLDER}'" loading="lazy"/>
    <div style="position:absolute; bottom:0; left:0; right:0;
                background:linear-gradient(transparent 0%, rgba(7,7,17,0.88) 55%, rgba(7,7,17,0.99) 100%);
                padding:36px 10px 11px; border-radius:0 0 13px 13px;">
      <p style="font-family:Inter,sans-serif; font-weight:600; font-size:0.88em;
                color:#fff; margin:0 0 4px; line-height:1.35;">{short_title}</p>
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px;">
        <span style="color:#FFD700; font-size:0.78em; font-weight:600;">⭐ {rating:.1f}</span>
        {"<span style='color:rgba(200,200,220,0.38); font-size:0.72em;'>• " + str(year) + "</span>" if year else ""}
      </div>
      <div>{genre_pills}</div>
    </div>
  </div>
</div>
"""

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
def render_sidebar(d):
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align:center; padding:14px 0 18px; border-bottom:1px solid rgba(229,9,20,0.2); margin-bottom:18px;">
          <p style="font-family:'Cinzel',serif; font-size:1.9em; font-weight:900;
                    background:linear-gradient(135deg,#E50914,#ff4444,#FFD700);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; margin:0; letter-spacing:3px;">CineAI</p>
          <p style="color:rgba(180,180,210,0.55); font-size:0.68em; letter-spacing:2.5px;
                    text-transform:uppercase; margin:3px 0 0;">Powered by Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        movies_df = d["movies"]
        has_xgb   = d["xgb"] is not None
        has_cat   = d["catboost"] is not None
        n_models  = 2 + int(has_xgb) + int(has_cat)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Movies", f"{len(movies_df):,}")
        with col2:
            st.metric("Models", n_models)

        st.markdown("---")

        # ── Model Selection ──
        st.markdown("""
        <p style="font-family:'Cinzel',serif; font-size:0.88em; font-weight:700;
                  color:#fff; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;">
          🤖 AI Model
        </p>""", unsafe_allow_html=True)

        available_models = {k: v for k, v in MODEL_INFO.items()
                            if v["key"] not in ("xgb", "catboost")
                            or (v["key"] == "xgb" and has_xgb)
                            or (v["key"] == "catboost" and has_cat)}

        model_label = st.radio(
            "Select model",
            list(available_models.keys()),
            index=0,
            label_visibility="collapsed",
        )
        st.session_state.model_key = available_models[model_label]["key"]

        # Show model description
        minfo = available_models[model_label]
        st.markdown(f"""
        <div style="background:rgba(229,9,20,0.07); border:1px solid rgba(229,9,20,0.25);
                    border-radius:9px; padding:9px 12px; margin:6px 0 14px; font-size:0.78em;
                    color:rgba(210,210,240,0.75); line-height:1.5;">
          {minfo['desc']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Mood Filter ──
        st.markdown("""
        <p style="font-family:'Cinzel',serif; font-size:0.88em; font-weight:700;
                  color:#fff; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">
          🎭 Mood Filter
        </p>""", unsafe_allow_html=True)
        st.session_state.mood = st.selectbox(
            "Mood", list(MOOD_TAGS.keys()), label_visibility="collapsed"
        )

        st.markdown("---")

        # ── Recommendations count ──
        st.markdown("""
        <p style="font-family:'Cinzel',serif; font-size:0.88em; font-weight:700;
                  color:#fff; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">
          🎯 Number of Results
        </p>""", unsafe_allow_html=True)
        st.session_state.n_recs = st.slider(
            "Results", 5, 20, 10, step=5, label_visibility="collapsed"
        )

        st.markdown("---")

        # ── Watchlist ──
        st.markdown("""
        <p style="font-family:'Cinzel',serif; font-size:0.88em; font-weight:700;
                  color:#fff; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">
          📋 My Watchlist
        </p>""", unsafe_allow_html=True)

        if not st.session_state.watchlist:
            st.markdown("""
            <p style="color:rgba(180,180,210,0.45); font-size:0.8em; text-align:center; padding:12px 0;">
              Your watchlist is empty.<br>Click ➕ on any movie card.
            </p>""", unsafe_allow_html=True)
        else:
            for i, movie in enumerate(st.session_state.watchlist):
                col_m, col_x = st.columns([5, 1])
                with col_m:
                    st.markdown(
                        f'<div class="wl-item"><div class="wl-dot"></div>{movie[:22]}</div>',
                        unsafe_allow_html=True
                    )
                with col_x:
                    if st.button("✕", key=f"rm_{i}_{movie[:8]}"):
                        st.session_state.watchlist.remove(movie)
                        st.rerun()
            if st.button("🗑️ Clear All", key="clear_wl"):
                st.session_state.watchlist = []
                st.rerun()

        st.markdown("---")

        # ── Footer ──
        st.markdown("""
        <div style="text-align:center; color:rgba(150,150,180,0.4); font-size:0.7em; margin-top:10px; line-height:1.8;">
          Powered by TMDB API<br>
          XGBoost • CatBoost • sklearn<br>
          <span style="color:rgba(229,9,20,0.5);">♥</span> CineAI 2025
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────
def render_hero(n_movies):
    st.markdown(f"""
    <div style="position:relative; overflow:hidden;
                background:linear-gradient(135deg,#070711 0%,#12001c 35%,#080826 65%,#1a0005 90%,#070711 100%);
                background-size:400% 400%;
                animation:heroGradient 14s ease infinite;
                padding:58px 40px 50px; text-align:center;
                border-bottom:1px solid rgba(229,9,20,0.22); margin-bottom:0;">

      <!-- radial glow -->
      <div style="position:absolute; top:50%; left:50%;
                  transform:translate(-50%,-50%);
                  width:650px; height:320px; border-radius:50%;
                  background:radial-gradient(ellipse, rgba(229,9,20,0.18) 0%, transparent 70%);
                  animation:heroGlow 5s ease-in-out infinite; pointer-events:none;"></div>

      <!-- floating film emoji -->
      <div style="position:absolute; top:18px; right:52px; font-size:62px;
                  opacity:0.08; animation:floatIcon 7s ease-in-out infinite;">🎞️</div>
      <div style="position:absolute; top:22px; left:52px; font-size:58px;
                  opacity:0.07; animation:floatIcon 9s ease-in-out infinite reverse;">🍿</div>

      <!-- title -->
      <h1 style="font-family:'Cinzel',serif; font-size:3.8em; font-weight:900; margin:0 0 6px;
                 background:linear-gradient(135deg,#E50914 0%,#ff4444 40%,#FFD700 70%,#E50914 100%);
                 background-size:200% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 background-clip:text; animation:shimmer 3.5s linear infinite; letter-spacing:3px;">
        CINE<span style="-webkit-text-fill-color:#E50914;">AI</span>
      </h1>

      <p style="color:rgba(210,210,240,0.6); font-size:1.08em; font-weight:300;
                letter-spacing:4px; text-transform:uppercase; margin:0 0 28px;">
        Your Personal AI Film Sommelier
      </p>

      <!-- stats bar -->
      <div style="display:inline-flex; gap:36px; background:rgba(255,255,255,0.04);
                  border:1px solid rgba(255,255,255,0.08); border-radius:50px;
                  padding:14px 32px; backdrop-filter:blur(10px);">
        <div style="text-align:center;">
          <span style="font-family:'Cinzel',serif; font-size:1.6em; font-weight:700; color:#E50914; display:block;">{n_movies:,}</span>
          <span style="font-size:0.65em; color:rgba(200,200,230,0.5); text-transform:uppercase; letter-spacing:2px;">Movies</span>
        </div>
        <div style="width:1px; background:rgba(255,255,255,0.1);"></div>
        <div style="text-align:center;">
          <span style="font-family:'Cinzel',serif; font-size:1.6em; font-weight:700; color:#E50914; display:block;">4</span>
          <span style="font-size:0.65em; color:rgba(200,200,230,0.5); text-transform:uppercase; letter-spacing:2px;">AI Models</span>
        </div>
        <div style="width:1px; background:rgba(255,255,255,0.1);"></div>
        <div style="text-align:center;">
          <span style="font-family:'Cinzel',serif; font-size:1.6em; font-weight:700; color:#E50914; display:block;">TMDB</span>
          <span style="font-size:0.65em; color:rgba(200,200,230,0.5); text-transform:uppercase; letter-spacing:2px;">Live Data</span>
        </div>
      </div>
    </div>

    <style>
    @keyframes heroGradient {{
      0%   {{ background-position: 0%   50%; }}
      50%  {{ background-position: 100% 50%; }}
      100% {{ background-position: 0%   50%; }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SEARCH HISTORY BAR
# ─────────────────────────────────────────────────────────────────
def render_history():
    if not st.session_state.history:
        return
    chips_html = " ".join(
        f'<span class="hist-chip">🕐 {h}</span>'
        for h in st.session_state.history[-6:][::-1]
    )
    st.markdown(f"""
    <div style="padding:10px 0 6px;">
      <span style="font-size:0.75em; color:rgba(180,180,210,0.5); text-transform:uppercase;
                   letter-spacing:1.5px; margin-right:8px;">Recent:</span>
      {chips_html}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# RECOMMENDATION GRID
# ─────────────────────────────────────────────────────────────────
def render_rec_grid(recs: pd.DataFrame):
    if recs is None or recs.empty:
        st.warning("No recommendations found for this selection.")
        return

    n = len(recs)
    cols_per_row = 5
    for row_start in range(0, n, cols_per_row):
        row_slice = recs.iloc[row_start : row_start + cols_per_row]
        cols      = st.columns(cols_per_row)

        for col_i, (_, rec_row) in enumerate(row_slice.iterrows()):
            global_i = row_start + col_i
            with cols[col_i]:
                title    = rec_row["title"]
                movie_id = int(rec_row["movie_id"])
                details  = fetch_movie_details(movie_id)

                poster   = details.get("poster",   PLACEHOLDER)
                rating   = details.get("rating",   0.0)
                year     = details.get("year",     "")
                genres   = details.get("genres",   [])
                overview = details.get("overview", "")
                cast_str = ", ".join(details.get("cast",     []))
                runtime  = details.get("runtime",  0)
                director = details.get("director", "N/A")

                # Card HTML
                st.markdown(
                    card_html(poster, title, rating, year, genres,
                              rank=global_i + 1,
                              anim_cls=f"c{global_i % 10}"),
                    unsafe_allow_html=True,
                )

                # Interactive controls
                b1, b2 = st.columns(2)
                with b1:
                    wl_label = "✓ Added" if title in st.session_state.watchlist else "➕ Watch"
                    if st.button(wl_label, key=f"wl_{movie_id}_{global_i}"):
                        if title not in st.session_state.watchlist:
                            st.session_state.watchlist.append(title)
                        else:
                            st.session_state.watchlist.remove(title)
                        st.rerun()

                with b2:
                    st.button("ℹ️ Info", key=f"info_btn_{movie_id}_{global_i}")

                # Details expander
                with st.expander("📖 Details"):
                    if overview:
                        st.markdown(
                            f'<p style="font-size:0.82em; color:rgba(210,210,240,0.8); line-height:1.6;">{overview[:280]}…</p>',
                            unsafe_allow_html=True,
                        )
                    if cast_str:
                        st.markdown(
                            f'<p style="font-size:0.78em; color:rgba(180,180,220,0.65);">🎭 <b>Cast:</b> {cast_str}</p>',
                            unsafe_allow_html=True,
                        )
                    if director != "N/A":
                        st.markdown(
                            f'<p style="font-size:0.78em; color:rgba(180,180,220,0.65);">🎬 <b>Director:</b> {director}</p>',
                            unsafe_allow_html=True,
                        )
                    if runtime:
                        st.markdown(
                            f'<p style="font-size:0.78em; color:rgba(180,180,220,0.65);">⏱️ <b>Runtime:</b> {runtime} min</p>',
                            unsafe_allow_html=True,
                        )

# ─────────────────────────────────────────────────────────────────
# TRENDING SECTION
# ─────────────────────────────────────────────────────────────────
def render_trending():
    st.markdown("""
    <div class="section-hdr">
      <span style="font-size:1.5em;">🔥</span>
      <h2 class="section-hdr-title">Trending Today</h2>
      <span class="section-badge">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    trending = fetch_trending()
    if not trending:
        st.info("Could not load trending data. Check your internet connection.")
        return

    cols = st.columns(5)
    for i, movie in enumerate(trending[:5]):
        with cols[i]:
            st.markdown(
                card_html(
                    movie["poster"], movie["title"], movie["rating"],
                    movie["year"], [], rank=i + 1,
                    is_trending=True, anim_cls=f"c{i}",
                ),
                unsafe_allow_html=True,
            )
            with st.expander("📖"):
                st.markdown(
                    f'<p style="font-size:0.8em; color:rgba(210,210,240,0.8); line-height:1.6;">'
                    f'{movie["overview"][:200]}…</p>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────
# MODEL METRICS PANEL
# ─────────────────────────────────────────────────────────────────
def render_metrics(metrics_dict: dict):
    if not metrics_dict:
        return

    st.markdown("""
    <div class="section-hdr">
      <span style="font-size:1.5em;">📊</span>
      <h2 class="section-hdr-title">Model Benchmark</h2>
      <span class="section-badge">Precision • Recall • NDCG</span>
    </div>
    """, unsafe_allow_html=True)

    MODEL_COLORS = {
        "Cosine Similarity":  "#4488ff",
        "SVD (LSA)":          "#44cc88",
        "XGBoost Re-ranker":  "#ffaa00",
        "CatBoost Re-ranker": "#ff4488",
    }

    # Summary metrics at top
    cols = st.columns(len(metrics_dict))
    for col, (mname, m) in zip(cols, metrics_dict.items()):
        with col:
            st.metric(
                label=mname.replace(" Re-ranker", "").replace(" (LSA)", ""),
                value=f"{m['n']:.3f}",
                delta=f"P@10: {m['p']:.3f}",
            )

    # Detailed bar chart table
    st.markdown('<div class="metrics-panel">', unsafe_allow_html=True)
    for metric_name, key in [("Precision@10", "p"), ("Recall@10", "r"), ("NDCG@10", "n")]:
        st.markdown(f"""
        <p style="font-family:'Cinzel',serif; font-size:0.8em; color:rgba(200,200,230,0.5);
                  text-transform:uppercase; letter-spacing:1.5px; margin:0 0 8px;">{metric_name}</p>
        """, unsafe_allow_html=True)

        max_val = max(m[key] for m in metrics_dict.values()) or 1
        for mname, m in metrics_dict.items():
            pct   = m[key] / max_val * 100
            color = MODEL_COLORS.get(mname, "#E50914")
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin:5px 0;">
              <span style="min-width:160px; font-size:0.82em; color:#ccc;">{mname}</span>
              <div class="bar-bg" style="flex:1;">
                <div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
              </div>
              <span style="min-width:48px; font-size:0.82em; font-weight:700; color:{color}; font-family:'Cinzel',serif;">{m[key]:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<hr style="margin:10px 0; border-color:rgba(255,255,255,0.05);">', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # DataFrame table
    with st.expander("📋 Raw Metrics Table"):
        rows = []
        for mname, m in metrics_dict.items():
            rows.append({
                "Model": mname,
                "Precision@10": round(m["p"], 4),
                "Recall@10":    round(m["r"], 4),
                "NDCG@10":      round(m["n"], 4),
            })
        df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df.style.highlight_max(axis=0, color="#1a0008"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────
def main():
    d         = get_data()
    movies_df = d["movies"]

    # ── Sidebar ──
    render_sidebar(d)

    # ── Hero ──
    render_hero(len(movies_df))

    # ─────────────────────────────────────────────────────────────
    # SEARCH SECTION
    # ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(180deg, rgba(7,7,17,0) 0%, rgba(7,7,17,1) 100%);
                padding:24px 0 8px;">
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-hdr" style="margin-top:1.2rem;">
      <span style="font-size:1.5em;">🔍</span>
      <h2 class="section-hdr-title">Find Your Movie</h2>
    </div>
    """, unsafe_allow_html=True)

    render_history()

    # Search row
    s_col1, s_col2, s_col3 = st.columns([5, 1.2, 1.2])
    with s_col1:
        movie_titles = sorted(movies_df["title"].tolist())
        selected = st.selectbox(
            "Search a movie",
            options=[""] + movie_titles,
            index=0,
            placeholder="Type a movie name…",
            label_visibility="collapsed",
        )

    with s_col2:
        recommend_btn = st.button("🎬 Recommend", use_container_width=True)

    with s_col3:
        lucky_btn = st.button("🎲 Surprise Me", use_container_width=True)

    # Handle Surprise Me
    if lucky_btn:
        selected = random.choice(movie_titles)
        st.session_state.selected = selected

    if selected and selected != "":
        st.session_state.selected = selected

    # ─────────────────────────────────────────────────────────────
    # TABS: Recommendations | Trending | Metrics
    # ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "🔥 Trending Now", "📊 Model Metrics"])

    with tab1:
        if recommend_btn or lucky_btn or (st.session_state.selected and recommend_btn):
            if st.session_state.selected:
                mv_title  = st.session_state.selected
                mod_key   = st.session_state.model_key
                mood      = st.session_state.mood
                n_recs    = st.session_state.n_recs

                # Update history
                if mv_title not in st.session_state.history:
                    st.session_state.history.append(mv_title)

                # Section header
                model_label = next(k for k, v in MODEL_INFO.items() if v["key"] == mod_key)
                st.markdown(f"""
                <div class="section-hdr">
                  <span style="font-size:1.5em;">🎬</span>
                  <h2 class="section-hdr-title">Because you liked <em style="color:#E50914;">"{mv_title}"</em></h2>
                  <span class="section-badge">{model_label.split()[0]} {model_label.split()[1]}</span>
                </div>
                """, unsafe_allow_html=True)

                # Show selected movie details at top
                sel_id = movies_df[movies_df["title"] == mv_title]["movie_id"].values
                if len(sel_id):
                    sel_details = fetch_movie_details(int(sel_id[0]))
                    if sel_details:
                        sc1, sc2, sc3 = st.columns([1, 2, 3])
                        with sc1:
                            st.image(sel_details.get("poster", PLACEHOLDER), width=140)
                        with sc2:
                            st.markdown(f"""
                            <p style="font-family:'Cinzel',serif; font-size:1.4em; font-weight:700; color:#fff; margin:0 0 4px;">{mv_title}</p>
                            <p style="color:#FFD700; font-size:1em;">⭐ {sel_details.get('rating', 0):.1f}
                              <span style="color:rgba(200,200,220,0.45); font-size:0.8em;">
                                ({sel_details.get('votes', 0):,} votes) • {sel_details.get('year', '')} • {sel_details.get('runtime', 0)} min
                              </span>
                            </p>
                            <p style="font-size:0.8em; color:rgba(200,200,230,0.6); font-style:italic;">{sel_details.get('tagline', '')}</p>
                            <p style="font-size:0.82em; color:rgba(200,200,230,0.7);">🎬 {sel_details.get('director', 'N/A')}</p>
                            """, unsafe_allow_html=True)
                        with sc3:
                            overview_text = sel_details.get("overview", "")[:300]
                            st.markdown(
                                f'<p style="font-size:0.85em; color:rgba(210,210,240,0.75); line-height:1.7;">{overview_text}…</p>',
                                unsafe_allow_html=True,
                            )

                st.markdown('<div class="custom-divider" style="height:1px; background:linear-gradient(90deg,transparent,rgba(229,9,20,0.4),transparent); margin:16px 0 20px;"></div>', unsafe_allow_html=True)

                with st.spinner("🧠 Running AI recommendation engine…"):
                    recs = get_recommendations(mv_title, mod_key, mood, n_recs)

                if recs is not None and not recs.empty:
                    st.markdown(
                        f'<p style="font-size:0.82em; color:rgba(180,180,210,0.55); margin-bottom:12px;">'
                        f'Showing {len(recs)} results using <b style="color:#E50914;">{model_label}</b>'
                        f' with mood filter <b style="color:#E50914;">{mood}</b></p>',
                        unsafe_allow_html=True,
                    )
                    render_rec_grid(recs)
                else:
                    st.warning("No recommendations found. Try a different mood or model.")
            else:
                st.info("👆 Select a movie above and click **Recommend** to get AI-powered suggestions!")
        else:
            # Empty state
            st.markdown("""
            <div style="text-align:center; padding:60px 20px;">
              <p style="font-size:4em; margin:0;">🎬</p>
              <p style="font-family:'Cinzel',serif; font-size:1.3em; color:#fff; margin:10px 0 6px;">
                Discover Your Next Favourite Film
              </p>
              <p style="color:rgba(180,180,210,0.5); font-size:0.9em;">
                Search a movie you love, choose an AI model, and let CineAI do the magic.
              </p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        render_trending()

    with tab3:
        render_metrics(d["metrics"])

    # ─────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:30px 0 10px; margin-top:50px;
                border-top:1px solid rgba(229,9,20,0.12);">
      <p style="font-family:'Cinzel',serif; font-size:1em; font-weight:700; color:rgba(229,9,20,0.5); letter-spacing:3px;">CINEAI</p>
      <p style="color:rgba(160,160,190,0.35); font-size:0.72em; letter-spacing:1px;">
        Built with Streamlit • TMDB API • XGBoost • CatBoost • scikit-learn<br>
        Movie data via TMDB — not endorsed or certified by TMDB.
      </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
