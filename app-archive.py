# app.py — CFB PPA Monte Carlo Predictor + FPI Prior + Market Blend + EV + Bet Board + Projections Manager
# Run locally:  pip install streamlit pandas numpy requests supabase python-dateutil && streamlit run app.py

import os, io, json, time, re
import numpy as np
import pandas as pd
import streamlit as st
import requests
from typing import Optional, List, Dict, Tuple
from supabase import create_client, Client

# =============================== Defaults ===============================
DEFAULT_BASE = {
    "BASE_TEAM_POINTS": 27.0,      # used when CSV mu_pts is not present
    "RATING_SCALE_TO_POINTS": 7.5, # rating sd => points scaling (used for both legacy & ratings)
    "MIN_SD_POINTS": 10.0,
    "MAX_SD_POINTS": 24.0,
}

# Legacy model weights (used if ratings columns are missing)
DEFAULT_OFF_WEIGHTS = {
    "offense.ppa": 0.35,
    "offense.successRate": 0.20,
    "offense.explosiveness": 0.10,
    "offense.powerSuccess": 0.05,
    "offense.pointsPerOpportunity": 0.10,
    "offense.fieldPosition.averageStart": 0.05,
    "offense.standardDowns.successRate": 0.05,
    "offense.passingDowns.ppa": 0.05,
    "offense.rushingPlays.ppa": 0.025,
    "offense.passingPlays.ppa": 0.025,
}
DEFAULT_DEF_WEIGHTS = {
    "defense.ppa": -0.35,
    "defense.successRate": -0.20,
    "defense.explosiveness": -0.10,
    "defense.powerSuccess": -0.05,
    "defense.stuffRate": +0.075,
    "defense.pointsPerOpportunity": -0.10,
    "defense.fieldPosition.averageStart": -0.05,
    "defense.havoc.total": +0.075,
    "defense.rushingPlays.ppa": -0.025,
    "defense.passingPlays.ppa": -0.025,
}

# Required columns (legacy component features)
REQUIRED_COLUMNS = [
    "team",
    "offense.ppa","offense.successRate","offense.explosiveness",
    "offense.powerSuccess","offense.pointsPerOpportunity",
    "offense.fieldPosition.averageStart","offense.standardDowns.successRate",
    "offense.passingDowns.ppa","offense.rushingPlays.ppa","offense.passingPlays.ppa",
    "defense.ppa","defense.successRate","defense.explosiveness","defense.powerSuccess",
    "defense.stuffRate","defense.pointsPerOpportunity","defense.fieldPosition.averageStart",
    "defense.havoc.total","defense.rushingPlays.ppa","defense.passingPlays.ppa",
]

# Optional new columns coming from build_weekly_team_csv.py (ratings/volatility)
OPTIONAL_NEW_COLS = [
    "off_rating", "def_rating",   # opponent-adjusted ratings
    "mu_pts", "hfa_pts",          # league baseline & HFA estimate (reference)
    "vol_points", "games_played"  # team point volatility & GP
]

BET_LOG_COLS = [
    "id","timestamp","room","bettor","home","away","bet_type","pick","odds","stake",
    "model_spread","model_total","blended_spread","blended_total","recommendation",
    "ev_best","description","result"
]
SAVED_PROJ_COLS = [
    "id","timestamp","room","home","away","proj_home","proj_away","model_spread",
    "model_total","blended_spread","blended_total","winner","recommendation","ev_best",
    "weather_mult","hfa_points","n_sims","seed",
    # extra saved context
    "market_spread_home","market_total","market_weight",
    "spread_odds","total_odds","temp_f","wind_mph","precip","indoor","neutral",
]

LOCAL_SETTINGS_FILE = "model_settings.json"

# =============================== Supabase helpers ===============================
def _supabase_client() -> Optional[Client]:
    url = st.secrets.get("SB_URL")
    key = st.secrets.get("SB_SERVICE_KEY") or st.secrets.get("SB_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

def persistence_mode():
    return "supabase" if _supabase_client() is not None else "local_csv"

def _room() -> str:
    return st.session_state.get("room", st.secrets.get("ROOM", "main"))

# Settings persistence (Supabase JSON or local JSON)
def load_settings() -> Tuple[Dict, Dict, Dict]:
    sb = _supabase_client()
    if sb:
        try:
            res = sb.table("model_settings").select("*").eq("room", _room()).order("id", desc=True).limit(1).execute()
            if getattr(res, "error", None):
                st.info("Using default model settings (Supabase error).")
            else:
                rows = res.data or []
                if rows:
                    row = rows[0]
                    base = row.get("base_params") or {}
                    off = row.get("off_weights") or {}
                    deff = row.get("def_weights") or {}
                    return (
                        {**DEFAULT_BASE, **base},
                        {**DEFAULT_OFF_WEIGHTS, **off},
                        {**DEFAULT_DEF_WEIGHTS, **deff},
                    )
        except Exception:
            st.info("Using default model settings (Supabase settings table missing).")
    if os.path.exists(LOCAL_SETTINGS_FILE):
        try:
            with open(LOCAL_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = data.get("base_params", {})
            off = data.get("off_weights", {})
            deff = data.get("def_weights", {})
            return (
                {**DEFAULT_BASE, **base},
                {**DEFAULT_OFF_WEIGHTS, **off},
                {**DEFAULT_DEF_WEIGHTS, **deff},
            )
        except Exception:
            st.info("Using default model settings (local settings unreadable).")
    return DEFAULT_BASE.copy(), DEFAULT_OFF_WEIGHTS.copy(), DEFAULT_DEF_WEIGHTS.copy()

def save_settings(base_params: Dict, off_weights: Dict, def_weights: Dict) -> bool:
    sb = _supabase_client()
    payload = {
        "room": _room(),
        "base_params": base_params,
        "off_weights": off_weights,
        "def_weights": def_weights,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if sb:
        try:
            res = sb.table("model_settings").insert(payload).execute()
            if getattr(res, "error", None):
                st.warning(f"Supabase settings insert error: {res.error}")
            else:
                return True
        except Exception as e:
            st.warning(f"Supabase settings insert failed: {e}")
    try:
        with open(LOCAL_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save local settings: {e}")
        return False

# Generic row persistence (Supabase then CSV fallback)
def persist_row(row: dict, table: str) -> bool:
    mode = persistence_mode()
    row = dict(row); row.setdefault("room", _room())
    if mode == "supabase":
        try:
            sb = _supabase_client(); assert sb is not None
            res = sb.table(table).insert(row).execute()
            if getattr(res, "error", None):
                st.warning(f"Supabase insert error: {res.error}")
                return False
            return True
        except Exception as e:
            st.warning(f"Supabase insert failed, falling back to CSV: {e}")
    try:
        path = f"{table}.csv"
        df = pd.DataFrame([row])
        if os.path.exists(path): df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        st.error(f"Local CSV persist failed: {e}")
        return False

def update_row(table: str, row_id: int, fields: dict) -> bool:
    if persistence_mode() != "supabase":
        st.warning("Updates require Supabase (CSV fallback is append-only).")
        return False
    try:
        sb = _supabase_client(); assert sb is not None
        res = sb.table(table).update(fields).eq("id", row_id).execute()
        if getattr(res, "error", None):
            st.warning(f"Supabase update error: {res.error}")
            return False
        return True
    except Exception as e:
        st.warning(f"Supabase update failed: {e}")
        return False

def delete_row(table: str, row_id: int) -> bool:
    if persistence_mode() != "supabase":
        path = f"{table}.csv"
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "id" in df.columns:
                    df = df[df["id"] != row_id]
                    df.to_csv(path, index=False)
                    return True
            except Exception as e:
                st.warning(f"Local CSV delete failed: {e}")
        st.warning("Delete requires Supabase or an 'id' column in local CSV.")
        return False
    try:
        sb = _supabase_client(); assert sb is not None
        res = sb.table(table).delete().eq("id", row_id).execute()
        if getattr(res, "error", None):
            st.warning(f"Supabase delete error: {res.error}")
            return False
        return True
    except Exception as e:
        st.warning(f"Supabase delete failed: {e}")
        return False

def _empty_table(table: str) -> pd.DataFrame:
    if table == "bet_logs": return pd.DataFrame(columns=BET_LOG_COLS)
    if table == "saved_projections": return pd.DataFrame(columns=SAVED_PROJ_COLS)
    return pd.DataFrame()

def load_table(table: str) -> pd.DataFrame:
    df = None
    if persistence_mode() == "supabase":
        try:
            sb = _supabase_client(); assert sb is not None
            res = sb.table(table).select("*").eq("room", _room()).order("id", desc=True).limit(500).execute()
            if getattr(res, "error", None):
                st.warning(f"Supabase select error: {res.error}")
            else:
                data = res.data or []
                df = pd.DataFrame(data) if data else _empty_table(table)
        except Exception as e:
            st.warning(f"Supabase read failed, falling back to CSV: {e}")
    if df is None:
        path = f"{table}.csv"
        if os.path.exists(path):
            try: df = pd.read_csv(path)
            except Exception as e:
                st.warning(f"Failed to read {path}: {e}"); df = _empty_table(table)
        else:
            df = _empty_table(table)
    cols = BET_LOG_COLS if table == "bet_logs" else (SAVED_PROJ_COLS if table == "saved_projections" else [])
    for c in cols:
        if c not in df.columns: df[c] = pd.Series(dtype="object")
    return df

# =============================== Discord notify ===============================
def notify_discord(message: str):
    url = st.secrets.get("DISCORD_WEBHOOK_URL")
    if not url:
        return False, "Discord webhook not configured."
    try:
        resp = requests.post(url, json={"content": message}, timeout=8)
        return (200 <= resp.status_code < 300), f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)

# =============================== Modeling utils ===============================
def american_odds_profit_per_unit(odds: int) -> float:
    return 100 / abs(odds) if odds < 0 else odds / 100.0

def ev_from_p(p_win: float, odds: int, p_push: float = 0.0) -> float:
    profit = american_odds_profit_per_unit(odds)
    p_loss = max(0.0, 1.0 - p_win - p_push)
    return p_win * profit - p_loss

def bucket_confidence(ev: float, p: float) -> str:
    if ev >= 0.06 and p >= 0.60: return "STRONG"
    if ev >= 0.03 and p >= 0.56: return "BET"
    if ev >= 0.01 and p >= 0.53: return "LEAN"
    return "PASS"

def load_csv(uploaded):
    if "csv_bytes" in st.session_state:
        try: return pd.read_csv(io.BytesIO(st.session_state["csv_bytes"]))
        except Exception: st.session_state.pop("csv_bytes", None)
    if uploaded is not None:
        data = uploaded.getvalue()
        st.session_state["csv_bytes"] = data
        return pd.read_csv(io.BytesIO(data))
    for path in ("advancedStats.csv", "/mnt/data/advancedStats.csv"):
        if os.path.exists(path): return pd.read_csv(path)
    return pd.DataFrame()

# ---------- FPI helpers ----------
def _norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\W_]+", " ", s)                # keep words
    s = s.replace("&", " and ")
    s = s.replace("univ ", "university ").replace("univ", "university")
    s = s.replace("st ", " state ")              # lightly expand st -> state
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("(", "").replace(")", "")
    # collapse spaces
    return s.replace(" ", "")

_ALIAS = {
    # common ESPN ↔ CFBD mismatches
    "olemiss": "olemiss",          # CFBD uses 'Ole Miss'
    "louisianalafayette": "louisiana",
    "louisianamonroe": "louisianamonroe",
    "miamifl": "miamifl",
    "miamioh": "miamioh",
    "texasa&m": "texasam",
    "utsa": "utsa",
    "ucf": "ucf",
    "utsa": "utsa",
    "umass": "massachusetts",
    "uconn": "connecticut",
    "unlv": "unlv",
}

def load_fpi_csv(uploaded) -> pd.DataFrame:
    # Priority: uploaded -> local file -> none
    df = None
    if "fpi_bytes" in st.session_state:
        try:
            df = pd.read_csv(io.BytesIO(st.session_state["fpi_bytes"]))
        except Exception:
            st.session_state.pop("fpi_bytes", None)
    if df is None and uploaded is not None:
        data = uploaded.getvalue()
        st.session_state["fpi_bytes"] = data
        df = pd.read_csv(io.BytesIO(data))
    if df is None:
        for path in ("fpi_rankings.csv", "/mnt/data/fpi_rankings.csv"):
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
    if df is None: return pd.DataFrame()

    # Flexible column parsing
    cols = {c.lower(): c for c in df.columns}
    team_col = cols.get("team") or cols.get("school") or cols.get("name") or list(df.columns)[0]
    fpi_col = None
    for key in ["fpi","fpi rating","fpi_rating","fpi score"]:
        if key in cols: fpi_col = cols[key]; break
    if fpi_col is None:
        # Try to find a numeric column that looks like FPI
        num_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        fpi_col = num_candidates[0] if num_candidates else None
    rank_col = cols.get("rk") or cols.get("rank") or cols.get("fpi rank") or None
    conf_col = cols.get("conf") or cols.get("conference") or None

    out = pd.DataFrame()
    out["team_fpi_raw"] = df[team_col].astype(str)
    out["fpi"] = pd.to_numeric(df[fpi_col], errors="coerce") if fpi_col else np.nan
    if rank_col: out["fpi_rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    else: out["fpi_rank"] = np.nan
    if conf_col: out["conference"] = df[conf_col].astype(str)
    else: out["conference"] = ""

    # Normalized keys
    out["team_key"] = out["team_fpi_raw"].apply(_norm_name)
    out["team_key"] = out["team_key"].map(lambda x: _ALIAS.get(x, x))
    # Also a prefix dictionary for best-effort matching later
    return out

def _fpi_lookup_builder(fpi_df: pd.DataFrame):
    if fpi_df is None or fpi_df.empty:
        return lambda name: (np.nan, np.nan, "")
    # index by normalized key
    index = {}
    for _, r in fpi_df.iterrows():
        k = str(r["team_key"])
        if k not in index:
            index[k] = (r.get("fpi", np.nan), r.get("fpi_rank", np.nan), r.get("team_fpi_raw", ""))
    fpikeys = list(index.keys())
    def _lookup(name: str):
        key = _norm_name(name)
        key = _ALIAS.get(key, key)
        # exact
        if key in index:
            return index[key]
        # prefix match (school name prefix of "School Mascots")
        for fk in fpikeys:
            if fk.startswith(key) or key.startswith(fk):
                return index[fk]
        return (np.nan, np.nan, "")
    return _lookup

def coerce_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "team"] + [c for c in OPTIONAL_NEW_COLS if c not in ["team","precip"]]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    med = df[[c for c in REQUIRED_COLUMNS if c != "team" and c in df.columns]].median(numeric_only=True)
    for c in REQUIRED_COLUMNS:
        if c != "team" and c in df.columns:
            df[c] = df[c].fillna(med.get(c, 0.0))
    return df

def zscore_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            m = out[c].mean(); s = out[c].std(ddof=0)
            out[c+"__z"] = 0.0 if (s is None or s == 0 or not np.isfinite(s)) else (out[c] - m) / s
            out[c+"__z"] = out[c+"__z"].fillna(0.0)
    return out

def team_row(df: pd.DataFrame, team: str) -> pd.Series:
    row = df.loc[df["team"] == team]
    if row.empty: raise ValueError(f"Team '{team}' not found")
    return row.iloc[0]

def composite_rating(row: pd.Series, weights: dict) -> float:
    s = 0.0
    for col, w in weights.items():
        z = row.get(col + "__z", 0.0)
        if not np.isfinite(z): z = 0.0
        s += float(w) * float(z)
    return s

def weather_multiplier(temp_f: float, wind_mph: float, precip: str, indoor: bool) -> float:
    if indoor: return 1.0
    mult = 1.0
    if temp_f <= 25: mult *= 0.90
    elif temp_f <= 35: mult *= 0.95
    elif temp_f >= 95: mult *= 0.97
    elif temp_f >= 90: mult *= 0.985
    if wind_mph > 5: mult *= (1.0 - min(0.15, 0.01*(wind_mph-5)))
    mult *= {"None":1.00,"Light":0.97,"Moderate":0.93,"Heavy":0.88}.get(precip,1.0)
    return mult

def volatility_sd_dyn(off_row: pd.Series, def_row: pd.Series, min_sd: float, max_sd: float) -> float:
    off_expl = off_row.get("offense.explosiveness__z", 0.0) or 0.0
    def_havoc = def_row.get("defense.havoc.total__z", 0.0) or 0.0
    sd = min_sd + 4.0*max(0.0, off_expl) - 2.0*max(0.0, def_havoc)
    return float(np.clip(sd, min_sd, max_sd))

def safe_mu_sigma(mu, sd):
    if not np.isfinite(mu): mu = DEFAULT_BASE["BASE_TEAM_POINTS"]
    if not (np.isfinite(sd) and sd > 0): sd = 14.0
    return float(mu), float(sd)

def simulate_scores(mu_home, mu_away, sd_home, sd_away, n_sims, seed):
    rng = np.random.default_rng(int(seed))
    mu_home, sd_home = safe_mu_sigma(mu_home, sd_home)
    mu_away, sd_away = safe_mu_sigma(mu_away, sd_away)
    home = rng.normal(mu_home, sd_home, size=n_sims)
    away = rng.normal(mu_away, sd_away, size=n_sims)
    home = np.clip(home, 0, None); away = np.clip(away, 0, None)
    home = np.nan_to_num(home, nan=0.0); away = np.nan_to_num(away, nan=0.0)
    return np.rint(home).astype(int), np.rint(away).astype(int)

def cover_probs_and_ev(home_scores, away_scores, market_spread_home, market_total, spread_odds, total_odds):
    margin = home_scores - away_scores
    total = home_scores + away_scores
    k = market_spread_home
    k_is_int = abs(k - round(k)) < 1e-9
    if k_is_int:
        p_home_push = float(np.mean(margin == int(round(k))))
        p_home_cover = float(np.mean(margin > k))
        p_home_lose = max(0.0, 1.0 - p_home_cover - p_home_push)
    else:
        p_home_push = 0.0; p_home_cover = float(np.mean(margin > k)); p_home_lose = 1.0 - p_home_cover
    t = market_total
    t_is_int = abs(t - round(t)) < 1e-9
    if t_is_int:
        p_over_push = float(np.mean(total == int(round(t))))
        p_over = float(np.mean(total > t))
        p_under = max(0.0, 1.0 - p_over - p_over_push)
    else:
        p_over_push = 0.0; p_over = float(np.mean(total > t)); p_under = 1.0 - p_over
    return {
        "p_home_cover": p_home_cover, "p_home_push": p_home_push, "p_away_cover": p_home_lose,
        "ev_home_spread": ev_from_p(p_home_cover, spread_odds, p_home_push),
        "ev_away_spread": ev_from_p(p_home_lose,  spread_odds, p_home_push),
        "p_over": p_over, "p_over_push": p_over_push, "p_under": p_under,
        "ev_over": ev_from_p(p_over,  total_odds, p_over_push),
        "ev_under": ev_from_p(p_under, total_odds, p_over_push),
        "home_win": float(np.mean(margin > 0.0) + 0.5*np.mean(margin == 0.0)),
        "away_win": float(np.mean(margin < 0.0) + 0.5*np.mean(margin == 0.0)),
    }

def choose_recommendation(metrics, market_spread_home, market_total):
    candidates = [
        ("Home " + (f"{market_spread_home:+g}" if market_spread_home != 0 else "PK"),
         metrics["ev_home_spread"], metrics["p_home_cover"]),
        ("Away " + (f"{-market_spread_home:+g}" if market_spread_home != 0 else "PK"),
         metrics["ev_away_spread"], metrics["p_away_cover"]),
        (f"Over {market_total:g}", metrics["ev_over"], metrics["p_over"]),
        (f"Under {market_total:g}", metrics["ev_under"], metrics["p_under"]),
    ]
    label, ev, p = max(candidates, key=lambda x: x[1])
    return {"label": label, "ev": ev, "p": p, "confidence": bucket_confidence(ev, p)}

# =============================== Page & Styles ===============================
st.set_page_config(page_title="CFB Predictor — PPA + FPI", layout="wide")
st.markdown("""
<style>
.sticky-wrap { position: sticky; top: 0; z-index: 999; background: var(--background-color);
  padding: .5rem 0; border-bottom: 1px solid rgba(255,255,255,.08); }
.summary-card { border-radius: 14px; padding: 14px 18px; margin: 10px 0 18px 0;
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.12); }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
.summary-item { padding: 6px 10px; border-radius: 10px; background: rgba(255,255,255,.05); }
.summary-item .label { font-size: .8rem; opacity: .75; } .summary-item .value { font-weight: 700; font-size: 1.15rem; }
.winner { font-weight: 800; font-size: 1.25rem; }
.bet-col h4 { margin-top: .25rem; }
.bet-card, .proj-card { border: 1px solid rgba(255,255,255,.12); border-radius: 12px;
  padding: 10px 12px; margin-bottom: 8px; background: rgba(255,255,255,.04); }
.proj-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; }
.bet-pick { font-size: 1.05rem; font-weight: 700; }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:700; margin-left: 6px;}
.pill.pending{ background:rgba(255,255,255,.12); }
.pill.win{ background:rgba(0,200,0,.15); color:#9CFF9C;}
.pill.loss{ background:rgba(255,0,0,.18); color:#FF9D9D;}
.pill.push{ background:rgba(255,210,0,.18); color:#FFE38A;}
details > summary { font-size: 0.85rem; opacity: .8; }
.edit-row .stSelectbox, .edit-row .stButton { min-width: 140px; } .edit-row .stButton > button { width: 140px; }
.small-note { opacity:.7; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

st.title("🏈 CFB Predictor — PPA Monte Carlo + FPI Prior & Market Blend")

# =============================== Sidebar / data load ===============================
with st.sidebar:
    st.header("Data & Settings")
    uploaded = st.file_uploader("Upload PPA CSV", type=["csv"])
    df = load_csv(uploaded)
    if df.empty: st.info("Upload a CSV (or keep `advancedStats.csv` next to this app).")
    else: st.success(f"Loaded {len(df)} rows.")
    if st.button("Forget uploaded CSV"):
        st.session_state.pop("csv_bytes", None); st.rerun()

    st.write("---")
    st.subheader("FPI (optional)")
    fpi_upload = st.file_uploader("Upload FPI CSV", type=["csv"], key="fpi_up")
    fpi_df = load_fpi_csv(fpi_upload)
    if not fpi_df.empty:
        st.success(f"FPI loaded: {len(fpi_df)} teams")
    else:
        st.caption("FPI not loaded. (Drop in `fpi_rankings.csv` to auto-load.)")

    st.write("---")
    st.session_state["room"] = st.text_input("Room (namespace)", value=st.secrets.get("ROOM", "main"))
    st.caption(f"Persistence: **{persistence_mode()}** (room: `{_room()}`)")

# Always load settings so Tuning works without data
BASE, OFF_WEIGHTS, DEF_WEIGHTS = load_settings()

# =============================== Determine whether we have valid data for simulations ===============================
has_data = False
HAS_LEGACY = False
HAS_RATINGS = False

if df.empty:
    st.info("No CSV loaded — logging-only mode. You can still use **Bet Logger** and view **Saved Projections**.")
else:
    # --------- soften column expectations (rename a few common headers) ----------
    col_lower = {c.lower(): c for c in df.columns}
    # promote 'team' if it's named differently
    if "team" not in df.columns:
        for alt in ["Team", "school", "School", "team_name", "TeamName"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "team"})
                break

    # figure out which modes are available
    legacy_cols = [c for c in REQUIRED_COLUMNS if c != "team"]
    HAS_LEGACY = all(c in df.columns for c in legacy_cols)
    HAS_RATINGS = all(c in df.columns for c in ["off_rating", "def_rating"])

    # If we have neither legacy nor ratings, we can still run in FPI-only mode later
    if not HAS_LEGACY and not HAS_RATINGS:
        # we still need at least a 'team' column to make selections
        if "team" not in df.columns:
            st.warning("CSV has no 'team' column — logging-only mode enabled.")
        else:
            # minimal prep so the UI can populate teams
            df_z = df.copy()
            team_list = sorted(df_z["team"].astype(str).unique().tolist())
            has_data = True
    else:
        # Coerce + impute only for the legacy metrics we actually have
        df = coerce_and_impute(df)

        # Build z-scores for legacy component model if available
        if HAS_LEGACY:
            zcols = list(set(list(OFF_WEIGHTS.keys()) + list(DEF_WEIGHTS.keys())))
            df_z = zscore_columns(df, zcols)
        else:
            df_z = df.copy()

        # carry optional cols over verbatim
        for c in OPTIONAL_NEW_COLS:
            if c in df.columns and c not in df_z.columns:
                df_z[c] = df[c]

        team_list = sorted(df_z["team"].astype(str).unique().tolist())
        has_data = True


# Build FPI lookup (if loaded)
_fpi_lookup = _fpi_lookup_builder(fpi_df) if not fpi_df.empty else (lambda name: (np.nan, np.nan, ""))

# =============================== Sticky matchup ===============================
if has_data:
    st.markdown('<div class="sticky-wrap">', unsafe_allow_html=True)
    t1, t2, t3 = st.columns([1.2, 1.2, 0.9])
    with t1: home = st.selectbox("Home team", team_list, index=0, key="home_team_top")
    with t2: away = st.selectbox("Away team", team_list, index=1 if len(team_list) > 1 else 0, key="away_team_top")
    with t3:
        cA, cB = st.columns(2)
        with cB: neutral = st.toggle("Neutral", value=False)
    st.markdown('</div>', unsafe_allow_html=True)
    if home == away:
        st.warning("Pick two different teams.")
        st.stop()
else:
    st.info("Logging-only mode: upload a CSV to unlock matchup sims & projections. Bet Logger works below.")

# =============================== Tabs ===============================
tabs = st.tabs(["Adjustments", "Tuning", "Bet Logger", "Saved Projections"])

# ---------- Adjustments ----------
with tabs[0]:
    if not has_data:
        st.info("Upload a CSV to run simulations. Bet Logger is available without data.")
    else:
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.subheader("Weather & Context")
            indoor = st.checkbox("Indoors / Roof Closed", value=False)
            temp_f = st.slider("Temperature (°F)", -10, 110, 65, 1, disabled=indoor)
            wind_mph = st.slider("Wind (mph)", 0, 40, 5, 1, disabled=indoor)
            precip = st.select_slider("Precipitation", ["None","Light","Moderate","Heavy"], value="None", disabled=indoor)
            hfa = 0.0 if neutral else st.slider("Home Field Advantage (pts)", 0.0, 6.0, 2.5, 0.5)

            if "hfa_pts" in df_z.columns:
                try:
                    hfa_ref = float(pd.to_numeric(df_z["hfa_pts"], errors="coerce").dropna().median())
                    st.caption(f"CSV league HFA reference: {hfa_ref:.2f} pts")
                except Exception:
                    pass

        with right:
            st.subheader("Market & Simulation")
            st.caption("Spread inputs are **home-based** (negative = home favored).")
            market_spread_home = st.number_input("Market Spread (Home perspective)", value=-3.0, step=0.5, format="%.1f")
            market_total = st.number_input("Market Total", value=52.5, step=0.5, format="%.1f")
            spread_odds = st.number_input("Spread Price (American)", value=-110, step=5)
            total_odds = st.number_input("Total Price (American)", value=-110, step=5)
            market_weight = st.slider("Market Blend Weight (0 model → 1 market) — for totals & Model↔Market spread", 0.0, 1.0, 0.5, 0.05)
            n_sims = st.slider("Number of Simulations", 1000, 50000, 10000, 1000)
            seed = st.number_input("Random Seed", value=42, step=1)

            st.markdown("##### FPI Blending")
            auto_shrink = st.toggle("Auto-shrink to FPI early (by games played)", value=True)
            k_weeks = st.slider("FPI auto-shrink horizon (weeks)", 2, 8, 4, 1, disabled=not auto_shrink)
            w_fpi_model = st.slider("FPI ↔ Model weight", 0.0, 1.0, 0.35, 0.05)
            w_fpi_market = st.slider("FPI ↔ Market weight", 0.0, 1.0, 0.35, 0.05)

            st.markdown("##### All-Three Blend (weights sum to 1)")
            w_fpi_3 = st.slider("Weight — FPI", 0.0, 1.0, 0.33, 0.01)
            w_market_3 = st.slider("Weight — Market", 0.0, 1.0, 0.33, 0.01)
            w_model_3 = max(0.0, 1.0 - w_fpi_3 - w_market_3)
            st.caption(f"Implied Model weight: **{w_model_3:.2f}**")

# ---------- Tuning ----------
with tabs[1]:
    st.subheader("Model Tuning")
    t_left, t_right = st.columns([1, 1], gap="large")

    with t_left:
        st.markdown("##### Base Parameters")
        base_edit = {}
        base_edit["BASE_TEAM_POINTS"] = st.number_input("Baseline team points", value=float(DEFAULT_BASE["BASE_TEAM_POINTS"]), step=0.5)
        base_edit["RATING_SCALE_TO_POINTS"] = st.number_input("Scale: z/rating diff → points", value=float(DEFAULT_BASE["RATING_SCALE_TO_POINTS"]), step=0.25)
        base_edit["MIN_SD_POINTS"] = st.number_input("Min SD (points)", value=float(DEFAULT_BASE["MIN_SD_POINTS"]), step=0.5)
        base_edit["MAX_SD_POINTS"] = st.number_input("Max SD (points)", value=float(DEFAULT_BASE["MAX_SD_POINTS"]), step=0.5)

    def _weights_editor(title: str, weights: Dict) -> Dict:
        st.markdown(f"##### {title}")
        dfw = pd.DataFrame({"metric": list(weights.keys()), "weight": [float(v) for v in weights.values()]})
        edited = st.data_editor(dfw, hide_index=True, width="stretch")
        out = {}
        for _, r in edited.iterrows():
            try:
                out[str(r["metric"])] = float(r["weight"])
            except Exception:
                out[str(r["metric"])] = 0.0
        colA, colB, _ = st.columns(3)
        with colA:
            if st.button(f"Normalize {title.split()[0]} weights"):
                s = sum(abs(v) for v in out.values()) or 1.0
                for k in out: out[k] = out[k]/s
                st.session_state[f"norm_{title}"] = out; st.rerun()
        with colB:
            if st.button(f"Reset {title.split()[0]} to defaults"):
                return DEFAULT_OFF_WEIGHTS.copy() if "Off" in title else DEFAULT_DEF_WEIGHTS.copy()
        if f"norm_{title}" in st.session_state:
            out = st.session_state.pop(f"norm_{title}")
        return out

    with t_right:
        off_new = _weights_editor("Offense Weights", DEFAULT_OFF_WEIGHTS)
        def_new = _weights_editor("Defense Weights", DEFAULT_DEF_WEIGHTS)

    st.markdown("---")
    t1, _ = st.columns([1,3])
    with t1:
        if st.button("💾 Save Tuning"):
            ok = save_settings(base_edit, off_new, def_new)
            if ok: st.success("Saved model settings."); st.rerun()
            else: st.error("Could not save settings.")
    st.caption("Weights used only if ratings columns are missing. Defense weights may be negative where lower is better.")

# =============================== Model compute & Summary (only with data) ===============================
if has_data:
    home_row = team_row(df_z, home)
    away_row = team_row(df_z, away)

    # 1) Ratings vs legacy composites
    if 'HAS_RATINGS' in locals() and HAS_RATINGS:
        home_off = float(home_row.get("off_rating", 0.0))
        home_def = float(home_row.get("def_rating", 0.0))
        away_off = float(away_row.get("off_rating", 0.0))
        away_def = float(away_row.get("def_rating", 0.0))
        method_label = "Ratings"
    else:
        home_off = composite_rating(home_row, DEFAULT_OFF_WEIGHTS)
        home_def = composite_rating(home_row, DEFAULT_DEF_WEIGHTS)
        away_off = composite_rating(away_row, DEFAULT_OFF_WEIGHTS)
        away_def = composite_rating(away_row, DEFAULT_DEF_WEIGHTS)
        method_label = "Weighted components"

    # 2) Baseline scoring level & HFA
    mu_base_home = float(home_row.get("mu_pts", DEFAULT_BASE["BASE_TEAM_POINTS"]))
    mu_base_away = float(away_row.get("mu_pts", DEFAULT_BASE["BASE_TEAM_POINTS"]))
    mu_base = 0.5 * (mu_base_home + mu_base_away)
    pts_per_sd = float(DEFAULT_BASE["RATING_SCALE_TO_POINTS"])

    mu_home_raw = mu_base + pts_per_sd * (home_off - away_def)
    mu_away_raw = mu_base + pts_per_sd * (away_off - home_def)

    # Apply UI HFA (reference)
    hfa_pts_ui = 0.0 if neutral else float(hfa)
    mu_home_raw += hfa_pts_ui / 2.0
    mu_away_raw -= hfa_pts_ui / 2.0

    # 3) Weather impact
    w_mult = weather_multiplier(temp_f, wind_mph, precip, indoor)
    mu_home_model = mu_home_raw * w_mult
    mu_away_model = mu_away_raw * w_mult

    # 4) Volatility from CSV or fallback
    if "vol_points" in df_z.columns:
        sd_home = float(np.clip(home_row.get("vol_points", DEFAULT_BASE["MIN_SD_POINTS"]),
                                DEFAULT_BASE["MIN_SD_POINTS"], DEFAULT_BASE["MAX_SD_POINTS"]))
        sd_away = float(np.clip(away_row.get("vol_points", DEFAULT_BASE["MIN_SD_POINTS"]),
                                DEFAULT_BASE["MIN_SD_POINTS"], DEFAULT_BASE["MAX_SD_POINTS"]))
    else:
        sd_home = volatility_sd_dyn(home_row, away_row, DEFAULT_BASE["MIN_SD_POINTS"], DEFAULT_BASE["MAX_SD_POINTS"])
        sd_away = volatility_sd_dyn(away_row, home_row, DEFAULT_BASE["MIN_SD_POINTS"], DEFAULT_BASE["MAX_SD_POINTS"])

    if not indoor:
        sd_tighten = 1.0 - (1.0 - w_mult) * 0.5
        sd_home *= sd_tighten
        sd_away *= sd_tighten

    # ----------------- FPI pulls -----------------
    fpi_home, fpi_rank_h, fpi_name_h = _fpi_lookup(home)
    fpi_away, fpi_rank_a, fpi_name_a = _fpi_lookup(away)
    have_fpi_both = np.isfinite(fpi_home) and np.isfinite(fpi_away)

    # Display small FPI note if available
    if have_fpi_both:
        st.caption(f"FPI: {home} (rating {fpi_home:+.2f}, rank {int(fpi_rank_h) if np.isfinite(fpi_rank_h) else '—'}) vs "
                   f"{away} (rating {fpi_away:+.2f}, rank {int(fpi_rank_a) if np.isfinite(fpi_rank_a) else '—'})")

    # ----------------- Margins (6 variants) -----------------
    # Model (post-weather) margin:
    m_model = (mu_home_model - mu_away_model)
    # Market-implied expected margin (home perspective: -spread):
    m_market = -float(market_spread_home)
    # FPI neutral-field margin + UI HFA:
    m_fpi = ((fpi_home - fpi_away) + (0.0 if neutral else float(hfa))) if have_fpi_both else np.nan

    # Auto-shrink factor by games played
    gp_home = int(home_row.get("games_played", 0) or 0)
    gp_away = int(away_row.get("games_played", 0) or 0)
    shrink = 1.0
    if auto_shrink and have_fpi_both:
        shrink = max(0.0, min(1.0, 1.0 - min(gp_home, gp_away) / float(max(1, k_weeks))))
    w_fpi_model_eff = (w_fpi_model * shrink) if have_fpi_both else 0.0
    w_fpi_3_eff = (w_fpi_3 * shrink) if have_fpi_both else 0.0

    # Totals: model scoring level (post-weather)
    T_model = (mu_home_model + mu_away_model)
    total_model = T_model
    total_model_market = (1 - market_weight) * T_model + market_weight * float(market_total)

    # Build margins
    m_model_only = m_model
    m_model_market = (1 - market_weight) * m_model + market_weight * m_market

    m_fpi_only = m_fpi if np.isfinite(m_fpi) else np.nan
    m_fpi_model = (1 - w_fpi_model_eff) * m_model + (w_fpi_model_eff) * (m_fpi if np.isfinite(m_fpi) else m_model)
    m_fpi_market = (1 - w_fpi_market) * m_market + (w_fpi_market) * (m_fpi if np.isfinite(m_fpi) else m_market)

    m_all_three = (w_model_3 * m_model) + (w_fpi_3_eff * (m_fpi if np.isfinite(m_fpi) else 0.0)) + (w_market_3 * m_market)
    # If no FPI, renormalize two-way
    if not have_fpi_both:
        totw = w_model_3 + w_market_3
        m_all_three = (m_model * (w_model_3 / (totw or 1.0))) + (m_market * (w_market_3 / (totw or 1.0)))

    # Totals per variant
    total_fpi_only = total_model
    total_fpi_model = total_model
    total_fpi_market = total_model_market
    total_all_three = (1 - w_market_3) * total_model + (w_market_3) * float(market_total)

    # Lines (betting style per team: negative = favorite)
    def _line_from_margin(m): return (-m, +m) if np.isfinite(m) else (np.nan, np.nan)
    home_line_model, away_line_model = _line_from_margin(m_model_only)
    home_line_mktblend, away_line_mktblend = _line_from_margin(m_model_market)
    home_line_fpi, away_line_fpi = _line_from_margin(m_fpi_only)
    home_line_fpimodel, away_line_fpimodel = _line_from_margin(m_fpi_model)
    home_line_fpimkt, away_line_fpimkt = _line_from_margin(m_fpi_market)
    home_line_all, away_line_all = _line_from_margin(m_all_three)

    # Which margin powers the sim?
    st.markdown("##### Choose spread to power simulation / EV")
    choice = st.radio(
        "Margin source",
        ["All three", "Model only", "Model↔Market", "FPI only", "FPI↔Model", "FPI↔Market"],
        index=0, horizontal=True
    )
    if choice == "All three":
        margin_choice = m_all_three; total_choice = total_all_three
    elif choice == "Model only":
        margin_choice = m_model_only; total_choice = total_model
    elif choice == "Model↔Market":
        margin_choice = m_model_market; total_choice = total_model_market
    elif choice == "FPI only":
        margin_choice = m_fpi_only; total_choice = total_fpi_only
    elif choice == "FPI↔Model":
        margin_choice = m_fpi_model; total_choice = total_fpi_model
    else:  # FPI↔Market
        margin_choice = m_fpi_market; total_choice = total_fpi_market

    # Re-center means to the chosen margin while keeping scoring level (total_choice)
    mu_home = total_choice/2.0 + margin_choice/2.0
    mu_away = total_choice/2.0 - margin_choice/2.0

    # 5) Simulate with chosen margin/total
    home_scores, away_scores = simulate_scores(mu_home, mu_away, sd_home, sd_away, int(n_sims), int(seed))
    margins = home_scores - away_scores; totals = home_scores + away_scores

    # Derived model lines for reporting (from margins/T we used)
    model_spread = float(np.mean(margins))   # + => home ahead
    model_total  = float(np.mean(totals))
    model_home_line = -model_spread
    model_away_line = +model_spread

    metrics = cover_probs_and_ev(home_scores, away_scores, float(market_spread_home), float(market_total), int(spread_odds), int(total_odds))
    recommendation = choose_recommendation(metrics, float(market_spread_home), float(market_total))
    proj_home = float(np.mean(home_scores)); proj_away = float(np.mean(away_scores))
    winner = home if proj_home > proj_away else away

    # ----- Summary Card -----
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-grid">
      <div class="summary-item"><div class="label">Projected Score</div>
        <div class="value">{home}: {proj_home:.1f}</div></div>
      <div class="summary-item"><div class="label">&nbsp;</div>
        <div class="value">{away}: {proj_away:.1f}</div></div>
      <div class="summary-item"><div class="label">Projected Winner</div>
        <div class="value winner">{winner}</div></div>
      <div class="summary-item"><div class="label">Chosen Spread</div>
        <div class="value">{home}: {(-margin_choice):+.2f} &nbsp;/&nbsp; {away}: {(+margin_choice):+.2f}</div></div>
      <div class="summary-item"><div class="label">Chosen Total</div>
        <div class="value">{total_choice:.2f}</div></div>
      <div class="summary-item"><div class="label">Recommendation</div>
        <div class="value">{recommendation['confidence']} — {recommendation['label']}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ----- Quick metrics -----
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.subheader("Projected Score (Sim)")
        st.metric(home, f"{proj_home:.1f}"); st.metric(away, f"{proj_away:.1f}")
        st.caption(f"Median: {home} {np.median(home_scores):.0f} — {away} {np.median(away_scores):.0f}")
    with c2:
        st.subheader("Model Lines (from simulation)")
        st.metric(f"{home} spread", f"{model_home_line:+.2f}")
        st.metric(f"{away} spread", f"{model_away_line:+.2f}")
        st.caption(f"Betting-style (negative = favorite). Method: {method_label}.")
    with c3:
        st.subheader("Market Inputs")
        st.metric(f"Home line (market)", f"{float(market_spread_home):+g}")
        st.metric("Total (market)", f"{float(market_total):.2f}")
        st.caption(f"Blend sliders adjust comparisons below (totals use market weight when 'market' is present).")

    st.divider()
    a,b,c,d,e = st.columns(5)
    with a: st.metric(f"Home {market_spread_home:+g}", f"{metrics['p_home_cover']*100:.1f}%"); st.caption(f"EV @ {spread_odds}: {metrics['ev_home_spread']:.3f}/1u")
    with b: st.metric(f"Away {-market_spread_home:+g}", f"{metrics['p_away_cover']*100:.1f}%"); st.caption(f"EV @ {spread_odds}: {metrics['ev_away_spread']:.3f}/1u")
    with c: st.metric(f"Over {market_total:g}", f"{metrics['p_over']*100:.1f}%"); st.caption(f"EV @ {total_odds}: {metrics['ev_over']:.3f}/1u")
    with d: st.metric(f"Under {market_total:g}", f"{metrics['p_under']*100:.1f}%"); st.caption(f"EV @ {total_odds}: {metrics['ev_under']:.3f}/1u")
    with e:
        st.metric("Win Prob (Model)", f"{metrics['home_win']*100:.1f}% {home}")
        if have_fpi_both:
            st.caption(f"FPI rank: {home} #{int(fpi_rank_h) if np.isfinite(fpi_rank_h) else '—'} vs {away} #{int(fpi_rank_a) if np.isfinite(fpi_rank_a) else '—'}")

    # ----- Spreads Overview -----
    st.subheader("Spreads & Totals — Overview")
    grid = [
        ("Model only", home_line_model, away_line_model, total_model),
        ("Model↔Market", home_line_mktblend, away_line_mktblend, total_model_market),
        ("FPI only", home_line_fpi, away_line_fpi, total_fpi_only),
        ("FPI↔Model", home_line_fpimodel, away_line_fpimodel, total_fpi_model),
        ("FPI↔Market", home_line_fpimkt, away_line_fpimkt, total_fpi_market),
        ("All three", home_line_all, away_line_all, total_all_three),
    ]
    cols = st.columns(3, gap="large")
    for i, (label, hline, aline, tot) in enumerate(grid):
        with cols[i % 3]:
            st.markdown(
                f"<div class='summary-item'><div class='label'>{label}</div>"
                f"<div class='value'>{home}: {hline:+.2f} / {away}: {aline:+.2f}</div>"
                f"<div class='small-note'>Total: {tot:.2f}</div></div>",
                unsafe_allow_html=True
            )

    # Disagreement meter
    if have_fpi_both and np.isfinite(m_model) and np.isfinite(m_fpi):
        diff = abs(m_model - m_fpi)
        if diff >= 5.0:
            st.warning(f"Model vs FPI disagree by {diff:.1f} pts — proceed with caution.")

    st.subheader("Recommendation")
    st.write(f"**{recommendation['confidence']}** — {recommendation['label']} (EV {recommendation['ev']:.3f}/1u, P {recommendation['p']*100:.1f}%).")

# =============================== Bet Logger =========================== #
def _profit_units(row) -> float:
    stake = float(row.get("stake") or 1.0)
    odds = row.get("odds")
    try:
        odds = int(odds) if pd.notna(odds) else -110
    except Exception:
        odds = -110
    per_unit = american_odds_profit_per_unit(odds)
    res = (str(row.get("result") or "pending")).lower()
    if res == "win":  return stake * per_unit
    if res == "loss": return -stake
    return 0.0

def bettor_summary(df: pd.DataFrame, bettor: str):
    if df is None or df.empty or "bettor" not in df.columns:
        return 0, 0, 0, 0.0
    if "result" not in df.columns:
        df = df.copy(); df["result"] = "pending"
    d = df[(df["bettor"] == bettor) & (~df["result"].eq("deleted"))].copy()
    w = int((d["result"] == "win").sum())
    l = int((d["result"] == "loss").sum())
    p = int((d["result"] == "push").sum())
    vals = pd.to_numeric(d.apply(_profit_units, axis=1), errors="coerce")
    units = float(np.nansum(vals.to_numpy()))
    return w, l, p, units

with tabs[2]:
    top_cols = st.columns([1, 6])
    with top_cols[0]:
        if st.button("🔄 Refresh board"):
            st.rerun()

    st.markdown("### Bet Board")
    bet_df = load_table("bet_logs")

    # Board names (from secrets or fallback)
    default_bettors: List[str] = st.secrets.get("BETTORS", [])
    if not default_bettors and "bettor" in bet_df.columns and not bet_df.empty:
        default_bettors = [b for b in bet_df["bettor"].dropna().unique().tolist()][:3]
    while len(default_bettors) < 3:
        default_bettors.append(f"Bettor {len(default_bettors)+1}")

    with st.expander("Board Settings", expanded=False):
        colA, colB, colC = st.columns(3)
        with colA: b1 = st.text_input("Column 1", value=default_bettors[0])
        with colB: b2 = st.text_input("Column 2", value=default_bettors[1])
        with colC: b3 = st.text_input("Column 3", value=default_bettors[2])
        board_names = [b1 or "—", b2 or "—", b3 or "—"]

    fcol1, fcol2, _ = st.columns([1,1,2])
    with fcol1:
        show_status = st.selectbox("Show", ["All","Pending","Settled"], index=0)
    with fcol2:
        max_per_col = st.slider("Max per column", 5, 40, 20, 1)

    cols = st.columns(3, gap="large")
    valid_results = ["pending","win","loss","push"]

    for idx, who in enumerate(board_names):
        with cols[idx]:
            w_, l_, p_, u_ = bettor_summary(bet_df, who)
            st.markdown(
                f"<div class='bet-col'><h4>{who} — {w_}-{l_}-{p_} ({u_:+.2f}u)</h4></div>",
                unsafe_allow_html=True,
            )

            dfw = bet_df.copy()
            if "bettor" in dfw.columns:
                dfw = dfw[dfw["bettor"] == who]
            else:
                dfw = dfw.iloc[0:0]

            if "result" in dfw.columns:
                dfw = dfw[~dfw["result"].eq("deleted")]
            if show_status == "Pending" and "result" in dfw.columns:
                dfw = dfw[dfw["result"] == "pending"]
            elif show_status == "Settled" and "result" in dfw.columns:
                dfw = dfw[dfw["result"].isin(["win","loss","push"])]

            dfw = dfw.head(max_per_col)

            if dfw.empty:
                st.caption("No bets yet.")
            else:
                for _, r in dfw.iterrows():
                    odds_txt = f"{int(r['odds']):+d}" if pd.notna(r.get("odds")) else "—"
                    pill = r.get("result", "pending")
                    pill = pill if pill in valid_results else "pending"

                    desc_text = r.get("description")
                    note_html = ""
                    if isinstance(desc_text, str) and desc_text.strip():
                        note_html = f"<div class='bet-note'>{desc_text}</div>"

                    card_html = (
                        "<div class='bet-card'>"
                        f"<div class='bet-pick'>{r.get('pick', '')}</div>"
                        f"<div class='bet-type'>{r.get('bet_type', '')} "
                        f"<span class='pill {pill}'>{pill.title()}</span></div>"
                        f"<div class='bet-odds'>Odds: {odds_txt} • {r.get('timestamp', '')}</div>"
                        f"{note_html}"
                        "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    with st.expander("Edit", expanded=False):
                        with st.container():
                            edit_cols = st.columns([4, 3, 2, 0.2], gap="small")
                            new_res = edit_cols[0].selectbox(
                                "Result", valid_results,
                                index=valid_results.index(pill),
                                label_visibility="collapsed",
                                key=f"res_{int(r.get('id', 0))}"
                            )
                            rid = int(r.get("id", 0))
                            if edit_cols[1].button("Save", key=f"save_{rid}"):
                                if rid and update_row("bet_logs", rid, {"result": new_res}):
                                    st.success("Updated."); st.rerun()
                            if edit_cols[2].button("Delete", key=f"del_{rid}"):
                                if rid and delete_row("bet_logs", rid):
                                    st.success("Deleted."); st.rerun()

    st.markdown("---")
    st.markdown("### Log a Bet")

    with st.form("bet_form"):
        bettor = st.selectbox("Bettor", board_names, index=0)
        sportsbook = st.text_input("Sportsbook", value="—")
        bet_type = st.selectbox("Bet Type", ["Spread","Total","Moneyline"], index=0)

        if has_data:
            home_name, away_name = home, away
        else:
            c1, c2 = st.columns(2)
            with c1: home_name = st.text_input("Home team", "")
            with c2: away_name = st.text_input("Away team", "")
            if not (home_name and away_name):
                st.caption("Enter both team names to include them in the pick/Discord message.")

        if bet_type == "Spread":
            side = st.selectbox("Side", [f"Home ({home_name or 'Home'})", f"Away ({away_name or 'Away'})"])
            line = st.number_input("Line (home-based; -3.5 = Home -3.5)", value=0.0, step=0.5, format="%.1f")
            odds_str = st.text_input("Price (American, optional)", value="")
            pick = f"{side} {line:+g}"
        elif bet_type == "Total":
            ou = st.selectbox("Over / Under", ["Over","Under"])
            line = st.number_input("Total", value=50.0, step=0.5, format="%.1f")
            odds_str = st.text_input("Price (American, optional)", value="")
            pick = f"{ou} {line:g}"
        else:
            side = st.selectbox("Side", [f"Home ({home_name or 'Home'})", f"Away ({away_name or 'Away'})"])
            odds_str = st.text_input("Price (American, optional)", value="")
            pick = f"{side} ML {odds_str or ''}".strip()

        try:
            odds_val = int(odds_str) if odds_str.strip() else None
        except ValueError:
            odds_val = None

        stake = st.number_input("Stake (units or dollars)", value=1.0, step=0.5, min_value=0.0)
        description = st.text_area("Description / Note (optional)", height=60)
        notify = st.checkbox("Notify Discord", value=True)
        submit = st.form_submit_button("Save Bet")

    if submit:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        ms = round(model_spread, 2) if has_data else None
        mt = round(model_total, 2) if has_data else None
        bs = round(margin_choice, 2) if has_data else None
        bt = round(total_choice, 2) if has_data else None
        rec_txt = (f"{recommendation['confidence']} — {recommendation['label']}") if has_data else ""
        ev_best = round(recommendation["ev"], 3) if has_data else None

        row = {
            "timestamp": ts,
            "bettor": bettor,
            "home": home_name or "",
            "away": away_name or "",
            "bet_type": bet_type,
            "pick": pick,
            "odds": odds_val,
            "stake": float(stake),
            "model_spread": ms,
            "model_total": mt,
            "blended_spread": bs,
            "blended_total": bt,
            "recommendation": rec_txt,
            "ev_best": ev_best,
            "description": description,
            "result": "pending",
            "room": _room(),
        }
        if persist_row(row, "bet_logs"):
            st.success("Bet saved.")

        if notify:
            odds_txt = f"{int(odds_val):+d}" if odds_val is not None else "—"
            sep = "────────────────────────────────"
            note_line = f"Note: {description}\n" if (description and str(description).strip()) else ""

            lines = [
                sep,
                f"🚨 **{bettor}'s New Bet** 🚨",
                f"Game: {home_name or 'Home'} vs {away_name or 'Away'}",
                "",
                f"{bet_type}: {pick} @ {odds_txt}",
                f"Stake: {stake}u",
            ]
            if has_data:
                lines += [
                    "",
                    f"Chosen: {home_name or 'Home'} {(-margin_choice):+.2f} / {away_name or 'Away'} {(+margin_choice):+.2f} (total {total_choice:.2f})",
                    f"Reco: {recommendation['confidence']} — {recommendation['label']}",
                ]
            if note_line:
                lines += ["", note_line.strip()]
            lines.append(sep)

            ok, detail = notify_discord("\n".join(lines))
            st.info(f"Discord: {'sent' if ok else 'not sent'} ({detail})")

        st.rerun()

# =============================== Saved Projections (no loading) ===============================
with tabs[3]:
    st.markdown("### Save / Manage Projections")

    if has_data:
        if st.button("💾 Save Current Projection Summary"):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            row = {
                "timestamp": ts, "home": home, "away": away, "room": _room(),
                "proj_home": round(proj_home, 1), "proj_away": round(proj_away, 1),
                "model_spread": round(model_spread, 2), "model_total": round(model_total, 2),
                "blended_spread": round(margin_choice, 2), "blended_total": round(total_choice, 2),
                "winner": winner, "recommendation": f"{recommendation['confidence']} — {recommendation['label']}",
                "ev_best": round(recommendation["ev"], 3),
                "weather_mult": round(weather_multiplier(temp_f, wind_mph, precip, indoor), 3),
                "hfa_points": float(0.0 if neutral else hfa), "n_sims": int(n_sims), "seed": int(seed),
                # extra saved context
                "market_spread_home": float(market_spread_home),
                "market_total": float(market_total),
                "market_weight": float(market_weight),
                "spread_odds": int(spread_odds),
                "total_odds": int(total_odds),
                "temp_f": float(temp_f),
                "wind_mph": float(wind_mph),
                "precip": precip,
                "indoor": bool(indoor),
                "neutral": bool(neutral),
            }
            if persist_row(row, "saved_projections"):
                st.success("Projection saved.")
            st.rerun()
    else:
        st.info("Upload a CSV to save new projections. You can still view and manage previously saved items below.")

    proj_df = load_table("saved_projections")

    # Controls
    f1, f2, f3 = st.columns([2,1,1])
    with f1: filter_txt = st.text_input("Filter by team (contains)", value="")
    with f2: max_items = st.slider("Max items", 5, 100, 24, 1)
    with f3: sort_newest = st.toggle("Newest first", value=True)

    if not proj_df.empty:
        pdf = proj_df.copy()
        if filter_txt.strip():
            m = pdf["home"].fillna("").str.contains(filter_txt, case=False) | pdf["away"].fillna("").str.contains(filter_txt, case=False)
            pdf = pdf[m]
        pdf["ts_ord"] = pd.to_datetime(pdf["timestamp"], errors="coerce")
        pdf = pdf.sort_values("ts_ord", ascending=not sort_newest, na_position="last").drop(columns=["ts_ord"])
        pdf = pdf.head(max_items)

        st.markdown('<div class="proj-grid">', unsafe_allow_html=True)
        for _, r in pdf.iterrows():
            rid = int(r.get("id", 0))
            home_name = r.get('home','?'); away_name = r.get('away','?')
            blended_spread_r = float(r.get('blended_spread', 0) or 0.0)
            blended_total_r  = float(r.get('blended_total', 0) or 0.0)

            card = f"""
            <div class="proj-card">
              <div style="font-weight:700;font-size:1.05rem;">{home_name} vs {away_name}</div>
              <div style="opacity:.7;font-size:.85rem;">{r.get('timestamp','')}</div>
              <div style="margin-top:.35rem;">
                <div><b>Projected</b> — {home_name}: {r.get('proj_home','?')}, {away_name}: {r.get('proj_away','?')}</div>
                <div><b>Model</b> — {home_name}: {-(float(r.get('model_spread',0) or 0.0)):+.2f} / {away_name}: {(float(r.get('model_spread',0) or 0.0)):+.2f} (total {float(r.get('model_total',0) or 0.0):.2f})</div>
                <div><b>Chosen</b> — {home_name}: {(-blended_spread_r):+.2f} / {away_name}: {(+blended_spread_r):+.2f} (total {blended_total_r:.2f})</div>
                <div><b>Winner</b> — {r.get('winner','?')}</div>
                <div><b>Rec</b> — {r.get('recommendation','')}</div>
              </div>
            </div>
            """
            st.markdown(card, unsafe_allow_html=True)

            cDel, _ = st.columns([0.2, 2.8])
            with cDel:
                if st.button("Delete", key=f"del_proj_{rid}"):
                    if rid and delete_row("saved_projections", rid):
                        st.success("Deleted."); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        del_cols = st.columns([1,2])
        with del_cols[0]:
            confirm = st.checkbox("Confirm delete filtered")
        with del_cols[1]:
            if st.button("🗑️ Delete filtered"):
                if confirm:
                    ids = [int(i) for i in pdf["id"].dropna().tolist()]
                    ok_all = True
                    for rid in ids:
                        if not delete_row("saved_projections", rid): ok_all = False
                    st.success("Deleted filtered." if ok_all else "Deleted with some errors."); st.rerun()
                else:
                    st.warning("Check confirm to proceed.")

        st.dataframe(pdf, width="stretch")
        st.download_button(
            "⬇️ Download filtered (CSV)",
            data=pdf.to_csv(index=False).encode("utf-8"),
            file_name="saved_projections_filtered.csv",
            mime="text/csv",
        )
    else:
        st.caption("No saved projections yet.")
