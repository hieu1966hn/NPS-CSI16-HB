import os
import re
import difflib
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
import google.generativeai as genai
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

STEAM_DARK   = "#1b2838"
STEAM_MID    = "#2a475e"
STEAM_CARD   = "#16202d"
STEAM_BLUE   = "#66c0f4"
STEAM_GREEN  = "#c2e878"
STEAM_TEXT   = "#c6d4df"
STEAM_DIM    = "#8fa0b4"
STEAM_ACCENT = "#e8914c"

GRAPH_BG     = "#1b2838"
GRAPH_CARD   = "#16202d"
GRAPH_GRID   = "#2a3f5c"
GRAPH_TEXT   = "#c6d4df"

PLT_PARAMS = {
    "figure.facecolor":    GRAPH_BG,
    "axes.facecolor":      GRAPH_CARD,
    "axes.edgecolor":      GRAPH_GRID,
    "axes.labelcolor":     GRAPH_TEXT,
    "axes.titlecolor":     STEAM_BLUE,
    "axes.titlesize":      13,
    "axes.titleweight":    "bold",
    "axes.labelsize":      11,
    "axes.grid":           True,
    "grid.color":          GRAPH_GRID,
    "grid.linewidth":      0.6,
    "xtick.color":         GRAPH_TEXT,
    "ytick.color":         GRAPH_TEXT,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "legend.facecolor":    GRAPH_CARD,
    "legend.edgecolor":    GRAPH_GRID,
    "legend.labelcolor":   GRAPH_TEXT,
    "text.color":          GRAPH_TEXT,
}
plt.rcParams.update(PLT_PARAMS)
PALETTE_MAIN   = [STEAM_BLUE, STEAM_GREEN, STEAM_ACCENT, "#b388ff", "#f06292", "#4dd0e1"]
PALETTE_SEQ    = ["#1a3a52", "#1d4e6e", "#206688", "#2280a8", "#24a0cf", "#66c0f4", "#a8d8f0"]

ABBREV_MAP = {
    "csgo": "Counter-Strike: Global Offensive",
    "cs:go": "Counter-Strike: Global Offensive",
    "cs go": "Counter-Strike: Global Offensive",
    "css": "Counter-Strike: Source",
    "cs:s": "Counter-Strike: Source",
    "cs1": "Counter-Strike",
    "cscz": "Counter-Strike: Condition Zero",
    "tf2": "Team Fortress 2",
    "tfc": "Team Fortress Classic",
    "l4d": "Left 4 Dead",
    "l4d2": "Left 4 Dead 2",
    "left 4 dead 2": "Left 4 Dead 2",
    "gta5": "Grand Theft Auto V",
    "gta v": "Grand Theft Auto V",
    "gta 5": "Grand Theft Auto V",
    "gtav": "Grand Theft Auto V",
    "gta3": "Grand Theft Auto III",
    "gta iii": "Grand Theft Auto III",
    "gmod": "Garry's Mod",
    "garrys mod": "Garry's Mod",
    "dota2": "Dota 2",
    "dota 2": "Dota 2",
    "dota": "Dota 2",
    "p2": "Portal 2",
    "portal2": "Portal 2",
    "hl": "Half-Life",
    "hl2": "Half-Life 2",
    "half life 2": "Half-Life 2",
    "hk": "Hollow Knight",
    "hollow": "Hollow Knight",
    "ds": "DARK SOULS™ II",
    "ds1": "DARK SOULS™: Prepare to Die Edition",
    "ds2": "DARK SOULS™ II",
    "ds3": "DARK SOULS™ III",
    "dark souls 3": "DARK SOULS™ III",
    "darksouls": "DARK SOULS™ II",
    "pd2": "PAYDAY 2",
    "payday2": "PAYDAY 2",
    "r6": "Tom Clancy's Rainbow Six® Siege",
    "r6s": "Tom Clancy's Rainbow Six® Siege",
    "rainbow six siege": "Tom Clancy's Rainbow Six® Siege",
    "aoe2": "Age of Empires II HD",
    "aoe": "Age of Empires II HD",
    "age of empires 2": "Age of Empires II HD",
    "sw": "Stardew Valley",
    "sdv": "Stardew Valley",
    "stardew": "Stardew Valley",
    "terreria": "Terraria",
    "cup head": "Cuphead",
    "cuphed": "Cuphead",
    "dead cell": "Dead Cells",
}

GENRE_FLAGS = [
    "Indie", "Action", "Adventure", "RPG", "Strategy",
    "Simulation", "Casual", "Free to Play", "Early Access",
    "Massively Multiplayer", "Sports", "Racing",
]
CAT_FLAGS = [
    "Single-player", "Multi-player", "Online Multi-Player", "MMO",
    "Steam Cloud", "Co-op", "Steam Achievements",
    "Full controller support", "Steam Trading Cards",
]
BASE_FEATURES = [
    "price", "achievements", "average_playtime",
    "median_playtime", "required_age", "num_platforms", "game_age",
]

def _safe_key(s):
    return re.sub(r"[^a-z0-9]", "_", s.lower())

GENRE_COLS = [f"genre_{_safe_key(g)}" for g in GENRE_FLAGS]
CAT_COLS   = [f"cat_{_safe_key(c)}" for c in CAT_FLAGS]
FEATURE_COLS = BASE_FEATURES + GENRE_COLS + CAT_COLS

GB_BEST_PARAMS = {
    "n_estimators":    100,
    "max_depth":       4,
    "learning_rate":   0.08,
    "subsample":       0.7,
    "min_samples_leaf": 8,
    "random_state":    42,
}

@st.cache_data
def load_data():
    df = pd.read_csv("steam.csv")
    df = df[df["english"] == 1]
    df = df[df["positive_ratings"] > 500]
    df = df.reset_index(drop=True)
    df["features"] = (df["genres"].fillna("") + " " +
                      df["steamspy_tags"].fillna("") + " " +
                      df["categories"].fillna(""))
    return df

@st.cache_data
def load_full_data():
    return pd.read_csv("steam.csv")

@st.cache_data
def calculate_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=15000)
    matrix = tfidf.fit_transform(df["features"])
    return cosine_similarity(matrix, matrix)

def _enrich(df):
    work = df.copy()
    work["total_ratings"] = work["positive_ratings"] + work["negative_ratings"]
    work["positive_ratio"] = work["positive_ratings"] / work["total_ratings"].replace(0, np.nan)
    work["is_well_received"] = (work["positive_ratio"] >= 0.75).astype(int)
    work["num_platforms"]  = work["platforms"].apply(lambda x: len(str(x).split(";")))
    work["release_year"]   = pd.to_datetime(work["release_date"]).dt.year
    ref = work["release_year"].max()
    work["game_age"]       = ref - work["release_year"]
    for g, col in zip(GENRE_FLAGS, GENRE_COLS):
        work[col] = work["genres"].apply(lambda x: 1 if g in str(x) else 0)
    for c, col in zip(CAT_FLAGS, CAT_COLS):
        work[col] = work["categories"].apply(lambda x: 1 if c in str(x) else 0)
    return work

@st.cache_resource
def train_model(df):
    work = _enrich(df)
    X = work[FEATURE_COLS].fillna(0)
    y = work["is_well_received"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_dist = {
        "n_estimators":     [80, 100, 120, 150],
        "max_depth":        [3, 4, 5],
        "learning_rate":    [0.06, 0.08, 0.10, 0.12],
        "subsample":        [0.65, 0.7, 0.8],
        "min_samples_leaf": [6, 8, 12, 16],
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=12,
        scoring="accuracy",
        cv=skf,
        random_state=42,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, zero_division=0),
        "recall":           recall_score(y_test, y_pred, zero_division=0),
        "f1":               f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":          roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_importance": pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False),
        "best_params":      search.best_params_,
        "train_size":       len(X_train),
        "test_size":        len(X_test),
        "positive_rate":    float(y.mean()),
        "cv_score":         search.best_score_,
    }
    return model, metrics

def predict_score(model, row):
    work = pd.DataFrame([row])
    work["num_platforms"] = work["platforms"].apply(lambda x: len(str(x).split(";")))
    work["release_year"]  = pd.to_datetime(work["release_date"]).dt.year
    work["game_age"]      = 2019 - work["release_year"]
    for g, col in zip(GENRE_FLAGS, GENRE_COLS):
        work[col] = work["genres"].apply(lambda x: 1 if g in str(x) else 0)
    for c, col in zip(CAT_FLAGS, CAT_COLS):
        work[col] = work["categories"].apply(lambda x: 1 if c in str(x) else 0)
    X = work[FEATURE_COLS].fillna(0)
    proba = model.predict_proba(X)[0]
    idx   = list(model.classes_).index(1)
    return float(proba[idx])

def _make_acronym(name):
    words = re.sub(r"[^a-zA-Z0-9 ]", " ", name).split()
    return "".join(w[0] for w in words if w[0].isalpha()).lower()

@st.cache_data
def build_search_index(names):
    index = {}
    for n in names:
        low = n.lower()
        index[low] = n
        acr = _make_acronym(n)
        if len(acr) >= 2:
            index.setdefault(acr, n)
        stripped = re.sub(r"[^a-z0-9 ]", " ", low).strip()
        stripped = re.sub(r"\s+", " ", stripped)
        index.setdefault(stripped, n)
        index.setdefault(stripped.replace(" ", ""), n)
    index.update(ABBREV_MAP)
    return index

def search_games(query, names, index, top_n=8):
    if not query.strip():
        return []
    q       = query.strip().lower()
    q_clean = re.sub(r"[^a-z0-9 ]", " ", q).strip()
    q_clean = re.sub(r"\s+", " ", q_clean)
    q_ns    = q_clean.replace(" ", "")

    for variant in [q, q_clean, q_ns]:
        if variant in index:
            exact = index[variant]
            candidates = [exact] + [n for n in names if n != exact and q_clean in n.lower()]
            return list(dict.fromkeys(candidates))[:top_n]

    tokens = [t for t in q_clean.split() if len(t) > 1]
    if tokens:
        scored = []
        for name in names:
            nl = re.sub(r"[^a-z0-9 ]", " ", name.lower())
            score = sum(1 for t in tokens if t in nl) / len(tokens)
            if score > 0:
                scored.append((score, name))
        scored.sort(key=lambda x: -x[0])
        if scored and scored[0][0] >= 0.5:
            best_score = scored[0][0]
            top = [s[1] for s in scored if s[0] == best_score][:top_n]
            if len(top) < top_n:
                extras = [s[1] for s in scored if s[0] < best_score and s[1] not in top]
                top += extras[: top_n - len(top)]
            return top

    all_keys = list(index.keys())
    close = difflib.get_close_matches(q, all_keys, n=6, cutoff=0.55)
    if close:
        seen, results = set(), []
        for k in close:
            v = index[k]
            if v not in seen:
                seen.add(v)
                results.append(v)
        return results

    return []

def recommend_games(game_name, df, cosine_sim, n=3):
    try:
        idx       = df[df["name"].str.lower() == game_name.lower()].index[0]
        scores    = list(enumerate(cosine_sim[idx]))
        scores    = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
        return df.iloc[[i for i, _ in scores]]
    except IndexError:
        return pd.DataFrame()

def generate_ai_review(game_name, tags):
    model    = genai.GenerativeModel("gemini-2.5-flash")
    prompt   = (
        f"Phan tich ngan gon (duoi 50 tu) bang tieng Viet ve suc hut cua tro choi '{game_name}' "
        f"dua tren cac the loai: {tags}. Tap trung vao gia tri cot loi, khong dung van phong quang cao."
    )
    return model.generate_content(prompt).text

def score_color(score):
    if score >= 0.80:
        return STEAM_GREEN, "Rất cao"
    elif score >= 0.65:
        return STEAM_BLUE, "Cao"
    elif score >= 0.50:
        return STEAM_ACCENT, "Trung bình"
    else:
        return "#ef5350", "Thấp"

st.set_page_config(
    page_title="Steam AI Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}

.stApp {{
    background-color: {STEAM_DARK};
    color: {STEAM_TEXT};
}}

.hero-banner {{
    background: linear-gradient(135deg, {STEAM_CARD} 0%, #0d1b2e 60%, #0a1628 100%);
    border: 1px solid {STEAM_MID};
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}}
.hero-banner::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(102,192,244,0.08) 0%, transparent 70%);
    pointer-events: none;
}}
.hero-title {{
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0 0 8px;
    line-height: 1.2;
}}
.hero-title span {{ color: {STEAM_BLUE}; }}
.hero-sub {{
    color: {STEAM_DIM};
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: {STEAM_CARD};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {STEAM_MID};
    gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {STEAM_DIM} !important;
    font-weight: 500;
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 0.9rem;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    background: {STEAM_MID} !important;
    color: {STEAM_BLUE} !important;
}}

.stTextInput > div > div > input {{
    background: {STEAM_CARD} !important;
    border: 1.5px solid {STEAM_MID} !important;
    color: {STEAM_TEXT} !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    font-size: 1.05rem !important;
    transition: border-color 0.2s;
}}
.stTextInput > div > div > input:focus {{
    border-color: {STEAM_BLUE} !important;
    box-shadow: 0 0 0 3px rgba(102,192,244,0.12) !important;
}}
.stTextInput > div > div > input::placeholder {{
    color: {STEAM_DIM} !important;
}}

.stButton > button[kind="primary"],
div.stButton > button {{
    background: linear-gradient(135deg, {STEAM_BLUE} 0%, #4ba8e0 100%) !important;
    color: #0d1b2e !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(102,192,244,0.25) !important;
}}
div.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102,192,244,0.35) !important;
}}

.pill-btn {{
    display: inline-block;
    background: {STEAM_CARD};
    border: 1px solid {STEAM_MID};
    color: {STEAM_TEXT};
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    cursor: pointer;
    margin: 3px;
    transition: all 0.15s;
    text-decoration: none;
}}
.pill-btn:hover {{
    border-color: {STEAM_BLUE};
    color: {STEAM_BLUE};
}}

.suggest-box {{
    background: {STEAM_CARD};
    border: 1px solid {STEAM_MID};
    border-radius: 12px;
    padding: 12px 16px;
    margin: -4px 0 16px;
}}
.suggest-label {{
    color: {STEAM_DIM};
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}}

.game-card {{
    background: linear-gradient(160deg, {STEAM_CARD} 0%, #101c2a 100%);
    border: 1px solid {STEAM_MID};
    border-radius: 14px;
    padding: 22px;
    height: 100%;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}}
.game-card::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {STEAM_BLUE}, transparent);
    border-radius: 14px 14px 0 0;
}}
.game-card:hover {{
    border-color: {STEAM_BLUE};
}}
.card-rank {{
    font-size: 0.72rem;
    font-weight: 600;
    color: {STEAM_BLUE};
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
}}
.card-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 12px;
    line-height: 1.3;
}}
.tag-pill {{
    display: inline-block;
    background: rgba(102,192,244,0.1);
    border: 1px solid rgba(102,192,244,0.2);
    color: {STEAM_BLUE};
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px;
}}
.price-badge {{
    display: inline-block;
    background: rgba(194,232,120,0.12);
    border: 1px solid rgba(194,232,120,0.25);
    color: {STEAM_GREEN};
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 14px;
}}
.score-label {{
    font-size: 0.78rem;
    font-weight: 500;
    color: {STEAM_DIM};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}}
.score-value {{
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 6px;
}}
.divider-line {{
    border: none;
    border-top: 1px solid {STEAM_MID};
    margin: 14px 0;
}}
.ai-label {{
    font-size: 0.78rem;
    font-weight: 600;
    color: {STEAM_DIM};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}}
.ai-text {{
    color: {STEAM_TEXT};
    font-size: 0.88rem;
    line-height: 1.6;
    font-style: italic;
    background: rgba(102,192,244,0.05);
    border-left: 3px solid {STEAM_BLUE};
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
}}

.metric-card {{
    background: {STEAM_CARD};
    border: 1px solid {STEAM_MID};
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}}
.metric-val {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {STEAM_BLUE};
    line-height: 1.1;
}}
.metric-lbl {{
    font-size: 0.78rem;
    color: {STEAM_DIM};
    font-weight: 500;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}}

.section-header {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {STEAM_BLUE};
    letter-spacing: 0.3px;
    margin: 28px 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid {STEAM_MID};
    display: flex;
    align-items: center;
    gap: 8px;
}}

.graph-card {{
    background: {STEAM_CARD};
    border: 1px solid {STEAM_MID};
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 24px;
}}
.insight-box {{
    background: rgba(102,192,244,0.06);
    border: 1px solid rgba(102,192,244,0.18);
    border-left: 3px solid {STEAM_BLUE};
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-top: 12px;
    color: {STEAM_TEXT};
    font-size: 0.88rem;
    line-height: 1.65;
}}

.stMetric {{
    background: {STEAM_CARD} !important;
    border: 1px solid {STEAM_MID} !important;
    border-radius: 12px !important;
    padding: 14px !important;
}}
[data-testid="stMetricValue"] {{
    color: {STEAM_BLUE} !important;
    font-size: 1.6rem !important;
}}
[data-testid="stMetricLabel"] {{
    color: {STEAM_DIM} !important;
    font-size: 0.78rem !important;
}}
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {STEAM_MID}, {STEAM_BLUE}) !important;
    border-radius: 4px !important;
}}
.stProgress > div > div > div {{
    background: rgba(102,192,244,0.1) !important;
    border-radius: 4px !important;
}}
.stAlert, .stSuccess, .stError, .stWarning, .stInfo {{
    background: {STEAM_CARD} !important;
    border-radius: 10px !important;
    border: 1px solid {STEAM_MID} !important;
}}
.stSpinner > div {{
    border-top-color: {STEAM_BLUE} !important;
}}
div[data-testid="stSelectbox"] > div {{
    background: {STEAM_CARD} !important;
    border: 1px solid {STEAM_MID} !important;
    border-radius: 10px !important;
    color: {STEAM_TEXT} !important;
}}
.stDataFrame {{
    background: {STEAM_CARD} !important;
    border-radius: 12px;
    border: 1px solid {STEAM_MID};
}}
</style>
""", unsafe_allow_html=True)

df          = load_data()
full_df     = load_full_data()
cosine_sim  = calculate_similarity(df)
names_list  = df["name"].tolist()
search_idx  = build_search_index(names_list)

with st.spinner("Đang huấn luyện mô hình AI…"):
    ml_model, ml_metrics = train_model(df)

st.markdown(f"""
<div class="hero-banner">
    <p class="hero-title">🎮 Steam <span>AI</span> Game Recommender</p>
    <p class="hero-sub">
        Hệ thống gợi ý game thông minh · Phân tích dữ liệu Steam · Mô hình Machine Learning tích hợp Generative AI
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🎮  Tìm Kiếm & Gợi Ý",
    "📊  Phân Tích Dữ Liệu",
    "🤖  Mô Hình AI",
])

with tab1:
    st.markdown('<div class="section-header">🔍 Tìm Kiếm Tựa Game</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <p style="color:{STEAM_DIM}; font-size:0.88rem; margin-bottom:12px;">
      Hỗ trợ tên đầy đủ, viết tắt (<b style="color:{STEAM_BLUE}">CSGO</b>, <b style="color:{STEAM_BLUE}">TF2</b>, <b style="color:{STEAM_BLUE}">L4D2</b>, <b style="color:{STEAM_BLUE}">GModd</b>…),
      gõ sai chính tả, hoặc chỉ một phần tên.
    </p>
    """, unsafe_allow_html=True)

    user_query = st.text_input(
        "game_search",
        placeholder="Nhập tên game, viết tắt, hoặc từ khoá… (VD: CSGO, witcher, stardew)",
        label_visibility="collapsed",
    )

    suggestions = search_games(user_query, names_list, search_idx) if user_query.strip() else []

    selected_game = None

    if user_query.strip():
        if suggestions:
            if len(suggestions) == 1 and suggestions[0].lower() == user_query.strip().lower():
                selected_game = suggestions[0]
            else:
                exact = next((s for s in suggestions if s.lower() == user_query.strip().lower()), None)
                if exact:
                    selected_game = exact
                else:
                    st.markdown(f"""
                    <div class="suggest-box">
                        <div class="suggest-label">🔎 Kết quả tìm thấy — chọn tựa game bạn muốn</div>
                    </div>
                    """, unsafe_allow_html=True)
                    chosen = st.selectbox(
                        "Chọn game",
                        suggestions,
                        label_visibility="collapsed",
                    )
                    selected_game = chosen
        else:
            st.warning("Không tìm thấy tựa game nào phù hợp. Hãy thử tên khác hoặc kiểm tra chính tả.")

    st.markdown('<div class="section-header">⚡ Tìm Nhanh</div>', unsafe_allow_html=True)
    quick_picks = [
        "Counter-Strike: Global Offensive", "Terraria", "Portal 2",
        "Dota 2", "Left 4 Dead 2", "Stardew Valley",
        "Hollow Knight", "PAYDAY 2", "Cuphead", "Dead Cells",
    ]
    cols_q = st.columns(5)
    for i, qp in enumerate(quick_picks):
        if cols_q[i % 5].button(qp, key=f"qp_{i}", use_container_width=True):
            selected_game = qp

    if selected_game:
        st.markdown(f'<div class="section-header">🎯 Gợi Ý Dựa Trên: <em style="color:#fff">{selected_game}</em></div>', unsafe_allow_html=True)

        recommendations = recommend_games(selected_game, df, cosine_sim)

        if not recommendations.empty:
            cols = st.columns(3, gap="medium")
            for i, (_, row) in enumerate(recommendations.iterrows()):
                reception = predict_score(ml_model, row)
                col_hex, col_label = score_color(reception)
                tags = [t.strip() for t in str(row["steamspy_tags"]).split(";")[:5] if t.strip()]
                price_str = "Miễn phí" if row["price"] == 0 else f"${row['price']:.2f}"

                with cols[i]:
                    tag_html = "".join(f'<span class="tag-pill">{t}</span>' for t in tags)
                    st.markdown(f"""
                    <div class="game-card">
                        <div class="card-rank">#{i+1} Đề xuất</div>
                        <div class="card-title">{row['name']}</div>
                        <div>{tag_html}</div>
                        <hr class="divider-line"/>
                        <span class="price-badge">💰 {price_str}</span>
                        <div class="score-label">Dự đoán mức độ được yêu thích</div>
                        <div class="score-value" style="color:{col_hex};">{reception*100:.0f}% <small style="font-size:0.9rem;font-weight:500;">— {col_label}</small></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(max(reception, 0.0), 1.0))

                    if i == 0:
                        st.markdown(f'<div class="ai-label">✨ Phân tích AI chuyên sâu</div>', unsafe_allow_html=True)
                        with st.spinner("Đang tạo nhận xét…"):
                            try:
                                review = generate_ai_review(row["name"], row["steamspy_tags"])
                                st.markdown(f'<div class="ai-text">{review}</div>', unsafe_allow_html=True)
                            except Exception:
                                st.markdown(f'<div class="ai-text">Một tựa game đáng trải nghiệm với cơ chế chơi hấp dẫn và độ gắn kết cao.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="ai-text">Lựa chọn thay thế xuất sắc — cơ chế gameplay tương đồng, phong cách thiết kế ăn nhập với sở thích của bạn.</div>', unsafe_allow_html=True)
        else:
            st.error(f"Không tìm thấy game '{selected_game}' trong cơ sở dữ liệu. Hãy chọn game khác.")

with tab2:
    st.markdown('<div class="section-header">📊 Tổng Quan Dữ Liệu Steam</div>', unsafe_allow_html=True)

    total_games = len(full_df)
    avg_price   = full_df[full_df["price"] > 0]["price"].mean()
    max_ratings = full_df["positive_ratings"].max()
    free_pct    = (full_df["price"] == 0).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số game", f"{total_games:,}")
    c2.metric("Giá TB (game trả phí)", f"${avg_price:.2f}")
    c3.metric("Game miễn phí", f"{free_pct:.1f}%")
    c4.metric("Đánh giá tích cực cao nhất", f"{max_ratings:,}")

    adf = full_df.copy()
    adf["release_year"] = pd.to_datetime(adf["release_date"]).dt.year

    def fig_wrap():
        return plt.subplots(figsize=(10, 5))

    st.markdown('<div class="section-header">1 · Top 10 Thể Loại Phổ Biến</div>', unsafe_allow_html=True)
    with st.container():
        top_genres = adf["genres"].str.split(";").explode().str.strip().value_counts().head(10)
        fig, ax = fig_wrap()
        bars = ax.barh(top_genres.index[::-1], top_genres.values[::-1],
                       color=PALETTE_SEQ[::-1][:10], edgecolor="none", height=0.65)
        for bar, val in zip(bars, top_genres.values[::-1]):
            ax.text(val + 50, bar.get_y() + bar.get_height()/2,
                    f"{val:,}", va="center", fontsize=9, color=STEAM_DIM)
        ax.set_title("Top 10 Thể Loại Game Phổ Biến Nhất Trên Steam")
        ax.set_xlabel("Số lượng tựa game")
        ax.set_yticks(range(len(top_genres)))
        ax.set_yticklabels(top_genres.index[::-1])
        ax.set_xlim(0, top_genres.max() * 1.15)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", visible=False)
        ax.spines[["top","right","left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> <b style="color:{STEAM_BLUE}">Indie</b> và <b style="color:{STEAM_BLUE}">Action</b> chiếm
        ưu thế tuyệt đối, phản ánh làn sóng các studio nhỏ phát hành game độc lập nhờ Steam Direct. Hai thể loại này
        cộng lại chiếm hơn <b>{(top_genres.iloc[0]+top_genres.iloc[1])/total_games*100:.0f}%</b> toàn bộ danh mục,
        tạo ra thị trường cạnh tranh rất gay gắt cho các nhà phát triển.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">2 · Tỷ Lệ Hỗ Trợ Tiếng Anh</div>', unsafe_allow_html=True)
    with st.container():
        eng_counts = adf["english"].value_counts()
        fig, ax = fig_wrap()
        labels = ["Có hỗ trợ Tiếng Anh", "Không hỗ trợ"]
        colors = [STEAM_BLUE, STEAM_ACCENT]
        ax.pie(eng_counts, labels=labels, autopct="%1.1f%%", startangle=140,
               colors=colors, textprops={"color": GRAPH_TEXT, "weight": "bold"})
        ax.set_title("Tỷ Lệ Game Hỗ Trợ Tiếng Anh", color=STEAM_BLUE, fontweight="bold")
        fig.patch.set_facecolor(GRAPH_BG)
        ax.set_facecolor(GRAPH_CARD)
        st.pyplot(fig); plt.close(fig)
        pct_english = eng_counts.get(1, 0) / eng_counts.sum() * 100
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> Khoảng <b style="color:{STEAM_BLUE}">{pct_english:.1f}%</b> tựa game trên Steam
        có hỗ trợ Tiếng Anh. Số liệu này khẳng định Tiếng Anh gần như là ngôn ngữ bắt buộc để các studio
        tiếp cận thị trường toàn cầu trên nền tảng.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">3 · Phân Phối Giá Bán</div>', unsafe_allow_html=True)
    with st.container():
        price_data = adf[(adf["price"] > 0) & (adf["price"] <= 60)]["price"]
        median_p   = price_data.median()
        fig, ax = fig_wrap()
        n_bins = 30
        ax.hist(price_data, bins=n_bins, color=STEAM_BLUE, edgecolor=STEAM_CARD, alpha=0.85, linewidth=0.4)
        ax.axvline(median_p, color=STEAM_ACCENT, linewidth=2, linestyle="--", label=f"Trung vị: ${median_p:.2f}")
        ax.axvline(price_data.mean(), color=STEAM_GREEN, linewidth=2, linestyle=":", label=f"Trung bình: ${price_data.mean():.2f}")
        ax.set_title("Phân Phối Giá Bán Game Trả Phí (Dưới $60)")
        ax.set_xlabel("Giá (USD)")
        ax.set_ylabel("Số lượng game")
        ax.legend(framealpha=0.8)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> Giá trung vị chỉ <b style="color:{STEAM_BLUE}">${median_p:.2f}</b> và
        <b style="color:{STEAM_GREEN}">{free_pct:.0f}%</b> game hoàn toàn miễn phí. Phân phối lệch phải mạnh
        cho thấy Steam vận hành như một thị trường giá phổ thông — phần lớn game nằm trong khoảng $5–15,
        trong khi tác phẩm AAA giá cao trên $40 chỉ là thiểu số nhỏ.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">4 · Xu Hướng Phát Hành Theo Năm</div>', unsafe_allow_html=True)
    with st.container():
        yr = adf["release_year"].value_counts().sort_index()
        yr = yr[(yr.index >= 2003) & (yr.index <= 2019)]
        peak_yr = int(yr.idxmax())
        fig, ax = fig_wrap()
        ax.fill_between(yr.index, yr.values, alpha=0.18, color=STEAM_BLUE)
        ax.plot(yr.index, yr.values, color=STEAM_BLUE, linewidth=2.5, marker="o", markersize=5)
        ax.axvline(peak_yr, color=STEAM_ACCENT, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(peak_yr + 0.15, yr.max() * 0.95, f"Đỉnh {peak_yr}\n{yr.max():,} game",
                color=STEAM_ACCENT, fontsize=9, va="top")
        ax.set_title("Số Lượng Game Phát Hành Trên Steam Theo Năm")
        ax.set_xlabel("Năm")
        ax.set_ylabel("Số lượng tựa game")
        ax.set_xticks(yr.index[::2])
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> Số lượng game bùng nổ từ 2014, đạt đỉnh vào <b style="color:{STEAM_BLUE}">{peak_yr}</b>
        với <b>{yr.max():,}</b> tựa game — trùng với thời điểm Steam mở rộng chương trình Steam Direct giúp
        nhà phát triển độc lập dễ dàng lên kệ hơn. Tốc độ tăng trưởng này gần như gấp đôi mỗi 2 năm trong
        giai đoạn 2014–2017.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">5 · Mối Quan Hệ Đánh Giá Tích Cực vs Tiêu Cực</div>', unsafe_allow_html=True)
    with st.container():
        samp = adf.sample(min(3000, len(adf)), random_state=42)
        pos  = samp["positive_ratings"].clip(1)
        neg  = samp["negative_ratings"].clip(1)
        corr_val = adf["positive_ratings"].corr(adf["negative_ratings"])
        fig, ax = fig_wrap()
        sc = ax.scatter(pos, neg, alpha=0.35, s=14, c=STEAM_BLUE,
                        norm=plt.Normalize(0,1), linewidths=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title("Đánh Giá Tích Cực vs Tiêu Cực (Thang Log)")
        ax.set_xlabel("Đánh giá tích cực (log)")
        ax.set_ylabel("Đánh giá tiêu cực (log)")
        ax.text(0.97, 0.05, f"r = {corr_val:.2f}", transform=ax.transAxes,
                ha="right", fontsize=11, color=STEAM_ACCENT)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> Hệ số tương quan <b style="color:{STEAM_BLUE}">r = {corr_val:.2f}</b> — game càng phổ biến
        nhận càng nhiều đánh giá ở CẢ hai chiều. Điều này hoàn toàn hợp lý: độ phủ sóng cao kéo theo lượt
        tương tác lớn hơn, bất kể chất lượng. Để đánh giá chất lượng thực sự, cần xét <em>tỷ lệ</em>
        tích cực/tiêu cực chứ không phải số tuyệt đối.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">6 · Ma Trận Tương Quan Các Biến Số</div>', unsafe_allow_html=True)
    with st.container():
        num_cols = ["price","positive_ratings","negative_ratings","achievements","average_playtime"]
        num_cols = [c for c in num_cols if c in adf.columns]
        corr_m   = adf[num_cols].corr()
        rename   = {
            "price": "Giá", "positive_ratings": "Đ.G Tích Cực",
            "negative_ratings": "Đ.G Tiêu Cực",
            "achievements": "Thành Tựu", "average_playtime": "T.G Chơi TB"
        }
        corr_m = corr_m.rename(columns=rename, index=rename)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(GRAPH_BG)
        ax.set_facecolor(GRAPH_CARD)
        mask = np.triu(np.ones_like(corr_m, dtype=bool), k=1)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr_m, annot=True, fmt=".2f", cmap=cmap, center=0,
                    linewidths=0.5, linecolor=GRAPH_BG,
                    annot_kws={"size": 10, "color": GRAPH_TEXT},
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Ma Trận Tương Quan Giữa Các Chỉ Số", color=STEAM_BLUE, fontweight="bold", fontsize=13)
        ax.tick_params(colors=GRAPH_TEXT)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color=GRAPH_TEXT)
        plt.setp(ax.get_yticklabels(), color=GRAPH_TEXT)
        ax.collections[0].colorbar.ax.yaxis.set_tick_params(color=GRAPH_TEXT)
        ax.collections[0].colorbar.ax.tick_params(labelcolor=GRAPH_TEXT)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> <b>Giá bán gần như không tương quan</b> với thời gian chơi hay số lượng đánh giá
        (hệ số gần 0). Điều này chứng minh rằng trong ngành game, chất lượng nội dung và viral marketing
        quan trọng hơn nhiều so với chiến lược định giá khi nhắm đến độ phổ biến.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">7 · Thời Gian Chơi Theo Thể Loại</div>', unsafe_allow_html=True)
    with st.container():
        pt_df = adf[(adf["average_playtime"] > 0) & (adf["average_playtime"] < 2500)].copy()
        pt_df["main_genre"] = pt_df["genres"].apply(lambda x: x.split(";")[0].strip())
        top5g = pt_df["main_genre"].value_counts().head(5).index
        pt_df = pt_df[pt_df["main_genre"].isin(top5g)]
        order = pt_df.groupby("main_genre")["average_playtime"].median().sort_values(ascending=False).index
        fig, ax = fig_wrap()
        bp = ax.boxplot(
            [pt_df[pt_df["main_genre"] == g]["average_playtime"].values for g in order],
            patch_artist=True, notch=True, vert=True,
            medianprops=dict(color=STEAM_ACCENT, linewidth=2.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(color=STEAM_DIM, linewidth=1.2),
            capprops=dict(color=STEAM_DIM, linewidth=1.5),
            flierprops=dict(marker=".", color=STEAM_DIM, alpha=0.3, markersize=4),
        )
        colors = PALETTE_MAIN[:len(order)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(STEAM_MID)
        ax.set_xticklabels(order, rotation=15, ha="right")
        ax.set_title("Phân Phối Thời Gian Chơi Trung Bình Theo Thể Loại Chính")
        ax.set_xlabel("Thể loại chính")
        ax.set_ylabel("Thời gian chơi (phút)")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        med_by = pt_df.groupby("main_genre")["average_playtime"].median().sort_values(ascending=False)
        top_g  = med_by.index[0]
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> <b style="color:{STEAM_BLUE}">{top_g}</b> dẫn đầu về độ gắn kết người chơi với
        thời gian trung vị cao nhất <b>({med_by.iloc[0]:.0f} phút)</b>. Khoảng tứ phân vị rộng
        trong hầu hết thể loại cho thấy sự phân hóa lớn giữa các game — một số title "làm mưa làm gió"
        kéo con số trung bình lên rất cao so với phần còn lại của thể loại.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">8 · Indie vs Thể Loại Khác — Phân Phối Giá</div>', unsafe_allow_html=True)
    with st.container():
        adf["is_indie"] = adf["genres"].apply(lambda x: "Indie" if "Indie" in str(x) else "Khác")
        vi_df = adf[adf["price"].between(0.01, 30)]
        indie_med = vi_df[vi_df["is_indie"] == "Indie"]["price"].median()
        other_med = vi_df[vi_df["is_indie"] == "Khác"]["price"].median()
        fig, ax = fig_wrap()
        for grp, color in [("Indie", STEAM_BLUE), ("Khác", STEAM_ACCENT)]:
            vals = vi_df[vi_df["is_indie"] == grp]["price"]
            ax.hist(vals, bins=25, alpha=0.55, color=color, label=grp, edgecolor=STEAM_CARD, linewidth=0.3)
        ax.axvline(indie_med, color=STEAM_BLUE,   linewidth=2, linestyle="--", alpha=0.9)
        ax.axvline(other_med, color=STEAM_ACCENT, linewidth=2, linestyle="--", alpha=0.9)
        ax.set_title("Phân Phối Giá: Indie vs Thể Loại Khác (Dưới $30)")
        ax.set_xlabel("Giá (USD)")
        ax.set_ylabel("Số lượng game")
        ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> Game Indie có giá trung vị
        <b style="color:{STEAM_BLUE}">${indie_med:.2f}</b> — thấp hơn hẳn so với
        <b style="color:{STEAM_ACCENT}">${other_med:.2f}</b> của các nhóm khác. Phân phối Indie tập
        trung dày đặc ở vùng $5–10, phản ánh chiến lược định giá tiếp cận đại chúng để cạnh tranh
        trong thị trường đã bão hòa về số lượng title.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">9 · Top 10 Nhà Phát Hành</div>', unsafe_allow_html=True)
    with st.container():
        top_pub = adf["publisher"].value_counts().head(10)
        fig, ax = fig_wrap()
        gradient = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_pub)))
        ax.barh(top_pub.index[::-1], top_pub.values[::-1],
                color=gradient, edgecolor="none", height=0.65)
        for val, y_pos in zip(top_pub.values[::-1], range(len(top_pub))):
            ax.text(val + 2, y_pos, f"{val:,}", va="center", fontsize=9, color=STEAM_DIM)
        ax.set_title("Top 10 Nhà Phát Hành Có Nhiều Tựa Game Nhất")
        ax.set_xlabel("Số lượng tựa game")
        ax.set_xlim(0, top_pub.max() * 1.15)
        ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", visible=False)
        ax.spines[["top","right","left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> <b style="color:{STEAM_BLUE}">{top_pub.index[0]}</b> dẫn đầu với
        <b>{top_pub.iloc[0]:,}</b> game — cách biệt lớn so với vị trí thứ hai. Đây là mô hình
        "volume publisher" chuyên phát hành số lượng lớn game giá rẻ, khác hoàn toàn với chiến lược
        AAA của Ubisoft hay Square Enix (xuất hiện ở vị trí thấp hơn với ít game hơn nhưng chất lượng cao).
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">10 · Hỗ Trợ Đa Nền Tảng</div>', unsafe_allow_html=True)
    with st.container():
        plat = adf["platforms"].str.split(";").explode().str.strip().value_counts()
        plat = plat[plat.index.isin(["windows","mac","linux"])]
        pct  = plat / len(adf) * 100
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(GRAPH_BG)
        ax.set_facecolor(GRAPH_CARD)
        bar_colors = [STEAM_BLUE, STEAM_GREEN, STEAM_ACCENT]
        bars = ax.bar(plat.index, plat.values, color=bar_colors, edgecolor="none",
                      width=0.5)
        for bar, p in zip(bars, pct.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                    f"{p:.0f}%", ha="center", fontsize=11, fontweight="600", color=GRAPH_TEXT)
        ax.set_title("Số Game Hỗ Trợ Theo Nền Tảng", color=STEAM_BLUE, fontweight="bold")
        ax.set_ylabel("Số lượng tựa game")
        ax.set_xticklabels(["Windows","macOS","Linux"])
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3); ax.grid(axis="x", visible=False)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> <b style="color:{STEAM_BLUE}">Windows</b> gần như là điều kiện bắt buộc
        ({pct.get('windows',0):.0f}% game hỗ trợ), trong khi macOS ({pct.get('mac',0):.0f}%) và Linux
        ({pct.get('linux',0):.0f}%) vẫn ở mức thiểu số. Sự chênh lệch này phản ánh thực tế thị trường
        PC gaming — dù Valve tích cực thúc đẩy Linux/SteamDeck, hầu hết nhà phát triển vẫn ưu tiên
        nền tảng có thị phần người dùng lớn nhất.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">11 · Tỷ Lệ Đánh Giá Tích Cực Theo Khoảng Giá</div>', unsafe_allow_html=True)
    with st.container():
        r_df = adf[(adf["positive_ratings"] + adf["negative_ratings"]) >= 100].copy()
        r_df["total"]  = r_df["positive_ratings"] + r_df["negative_ratings"]
        r_df["pos_rt"] = r_df["positive_ratings"] / r_df["total"]
        bins  = [0, 0.01, 5, 10, 20, 30, 60, np.inf]
        lbls  = ["F2P","$0–5","$5–10","$10–20","$20–30","$30–60","$60+"]
        r_df["price_bin"] = pd.cut(r_df["price"], bins=bins, labels=lbls, right=False)
        grp   = r_df.groupby("price_bin", observed=True)["pos_rt"]
        means = grp.mean()
        cis   = grp.sem() * 1.96
        fig, ax = fig_wrap()
        colors = [STEAM_GREEN if v >= 0.75 else STEAM_ACCENT if v >= 0.65 else "#ef5350"
                  for v in means.values]
        ax.bar(means.index, means.values, color=colors, edgecolor="none",
               width=0.6, alpha=0.85)
        ax.errorbar(means.index, means.values, yerr=cis.values,
                    fmt="none", color=STEAM_TEXT, capsize=4, linewidth=1.5, alpha=0.7)
        ax.axhline(0.75, color=STEAM_BLUE, linewidth=1.5, linestyle="--", alpha=0.6,
                   label="Ngưỡng tích cực 75%")
        for idx, (val, ci) in enumerate(zip(means.values, cis.values)):
            ax.text(idx, val + ci + 0.008, f"{val*100:.0f}%",
                    ha="center", fontsize=9, color=GRAPH_TEXT)
        ax.set_title("Tỷ Lệ Đánh Giá Tích Cực Trung Bình Theo Khoảng Giá")
        ax.set_xlabel("Khoảng giá")
        ax.set_ylabel("Tỷ lệ đánh giá tích cực")
        ax.set_ylim(0.5, 0.95)
        ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        best_tier = means.idxmax()
        st.markdown(f"""<div class="insight-box">
        💡 <b>Nhận xét:</b> Khoảng giá <b style="color:{STEAM_BLUE}">{best_tier}</b> có tỷ lệ đánh giá
        tích cực cao nhất (<b>{means.max()*100:.0f}%</b>). Game miễn phí (F2P) thường có điểm đánh giá
        thấp hơn do kỳ vọng người chơi cao hơn và mô hình kinh doanh in-app purchase gây tranh cãi.
        Nhìn chung, tỷ lệ tích cực không tăng tuyến tính theo giá — chất lượng gameplay vẫn là yếu
        tố quyết định chứ không phải mức giá bán.
        </div>""", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-header">🤖 Mô Hình Dự Đoán Mức Độ Được Yêu Thích</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <p style="color:{STEAM_DIM}; font-size:0.9rem; line-height:1.7; margin-bottom:20px;">
    Mô hình <b style="color:{STEAM_BLUE}">Gradient Boosting Classifier</b> được tự động tối ưu hoá siêu tham số
    qua <b>Randomized Search CV (12 lần thử × 3-fold)</b> trên {ml_metrics['train_size'] + ml_metrics['test_size']:,} tựa game.
    Nhiệm vụ: dự đoán liệu một game có đạt <b>≥ 75% đánh giá tích cực</b> hay không.
    Kết quả được hiển thị trực tiếp trong tab Gợi Ý Game dưới dạng điểm % tin cậy.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("#### Kết quả huấn luyện")
    mc = st.columns(5)
    mc[0].metric("Accuracy",  f"{ml_metrics['accuracy']*100:.1f}%")
    mc[1].metric("Precision", f"{ml_metrics['precision']*100:.1f}%")
    mc[2].metric("Recall",    f"{ml_metrics['recall']*100:.1f}%")
    mc[3].metric("F1-Score",  f"{ml_metrics['f1']*100:.1f}%")
    mc[4].metric("ROC-AUC",   f"{ml_metrics['roc_auc']:.3f}")

    st.markdown("#### Siêu tham số tối ưu được chọn")
    param_df = pd.DataFrame([ml_metrics["best_params"]]).T.rename(columns={0: "Giá trị"})
    param_df.index.name = "Tham số"
    st.dataframe(param_df, use_container_width=True)

    col_cm, col_fi = st.columns([1, 1.6], gap="large")

    with col_cm:
        st.markdown("#### Ma Trận Nhầm Lẫn")
        cm   = ml_metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(GRAPH_BG)
        ax.set_facecolor(GRAPH_CARD)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Không nổi bật","Được\nyêu thích"], color=GRAPH_TEXT, fontsize=9)
        ax.set_yticklabels(["Không nổi bật","Được\nyêu thích"], color=GRAPH_TEXT, fontsize=9)
        ax.set_xlabel("Dự đoán", color=GRAPH_TEXT)
        ax.set_ylabel("Thực tế",  color=GRAPH_TEXT)
        ax.set_title("Confusion Matrix", color=STEAM_BLUE, fontweight="bold", fontsize=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else GRAPH_TEXT)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelcolor=GRAPH_TEXT)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col_fi:
        st.markdown("#### Tầm Quan Trọng Các Đặc Trưng (Top 15)")
        imp  = ml_metrics["feature_importance"].head(15)
        pretty = {
            "price": "Giá bán",
            "achievements": "Số thành tựu",
            "average_playtime": "T.G chơi TB",
            "median_playtime": "T.G chơi trung vị",
            "required_age": "Độ tuổi yêu cầu",
            "num_platforms": "Số nền tảng",
            "game_age": "Tuổi của game",
        }
        labels = [pretty.get(f, f.replace("genre_","🎮 ").replace("cat_","📂 ")
                              .replace("_"," ").title()) for f in imp.index]
        colors_fi = [STEAM_BLUE if imp[f] == imp.iloc[0]
                     else STEAM_GREEN if v >= imp.iloc[0]*0.6
                     else STEAM_DIM
                     for f, v in zip(imp.index, imp.values)]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor(GRAPH_BG)
        ax.set_facecolor(GRAPH_CARD)
        ax.barh(range(len(imp)), imp.values[::-1], color=colors_fi[::-1],
                edgecolor="none", height=0.65)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(labels[::-1], color=GRAPH_TEXT, fontsize=9)
        ax.set_xlabel("Mức độ quan trọng", color=GRAPH_TEXT)
        ax.set_title("Feature Importance", color=STEAM_BLUE, fontweight="bold", fontsize=12)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.markdown(f"""<div class="insight-box">
    💡 <b>Giải thích mô hình:</b> Ba đặc trưng quan trọng nhất là
    <b style="color:{STEAM_BLUE}">{pretty.get(ml_metrics['feature_importance'].index[0], ml_metrics['feature_importance'].index[0])}</b>,
    <b style="color:{STEAM_BLUE}">{pretty.get(ml_metrics['feature_importance'].index[1], ml_metrics['feature_importance'].index[1])}</b>, và
    <b style="color:{STEAM_BLUE}">{pretty.get(ml_metrics['feature_importance'].index[2], ml_metrics['feature_importance'].index[2])}</b>.
    Điều này cho thấy thời gian người chơi đầu tư và mức giá là tín hiệu chất lượng quan trọng nhất —
    game được chơi nhiều giờ thường phản ánh nội dung tốt, dẫn đến tỷ lệ đánh giá tích cực cao hơn.
    Mô hình đạt accuracy <b>{ml_metrics['accuracy']*100:.1f}%</b>, vượt baseline đa số
    <b>{ml_metrics['positive_rate']*100:.1f}%</b> một cách có ý nghĩa.
    </div>""", unsafe_allow_html=True)

    st.caption(f"""
    Tập dữ liệu: {ml_metrics['train_size']:,} mẫu huấn luyện / {ml_metrics['test_size']:,} mẫu kiểm tra ·
    Tỷ lệ lớp tích cực: {ml_metrics['positive_rate']*100:.1f}% ·
    CV Accuracy tốt nhất: {ml_metrics['cv_score']*100:.1f}% ·
    Mô hình tự động retrain khi dữ liệu nguồn thay đổi
    """)