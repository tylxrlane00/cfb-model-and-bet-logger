# app.py
# Streamlit GUI for CFB predictor (advanced PPA schema only)
# This build:
# - FIX: spread signal sign / EV selection (uses Φ((model-line)/σ))
# - CHANGE: PPD floor now applied AFTER weather/situational multipliers
# - TUNE: EFF_TO_PPD_K nudged to 0.07 (gentler efficiency→PPD slope)
# - Session-state persistence, sigma sliders, signals-basis toggle
# - Probability display clamp [1%, 99%]
# - Market blend KPIs (Final) vs Model KPIs (no blend)

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import os, re, math, json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ========== CONSTANTS / CONFIG (overridable by config.json) ==========

DEFAULTS = dict(
    # Variance knobs (Normal approx)
    SPREAD_SIGMA_PTS = 13.0,
    TOTALS_SIGMA_PTS = 11.0,

    # Game-scale knobs
    DEFAULT_HFA_POINTS = 2.5,
    BASE_POINTS_PER_DRIVE = 2.28,
    PLAYS_PER_DRIVE = 5.8,
    DEFAULT_TEMPO = 65.8,   # per-team baseline tempo proxy (plays)

    # Efficiency->PPD curve (nudged from 0.11 → 0.07)
    EFF_TO_PPD_K = 0.07,
    MIN_PPD = 0.80,         # floor is applied AFTER multipliers

    # Weather effects
    WX_WIND_TEMPO_PER_MPH = -0.0015,
    WX_WIND_PPD_PER_MPH = -0.008,
    WX_RAIN_TEMPO = -0.035,
    WX_SNOW_TEMPO = -0.05,
    WX_RAIN_PPD = -0.06,
    WX_SNOW_PPD = -0.10,
    WX_COLD_PPD = -0.03,
    WX_HOT_PPD = -0.02,

    # Market + EV
    MARKET_BLEND = {"use": True, "spread_weight": 0.50, "total_weight": 0.50},
    DEFAULT_BOOK_ODDS = -110,
    EV_THRESHOLDS = {"strong": 0.05, "lean": 0.02},

    # Off/Def feature weights (tune freely)
    OFF_WEIGHTS = {
        "ppa": 0.35,
        "success": 0.20,
        "explosiveness": 0.10,
        "ppo": 0.10,
        "power": 0.05,
        "pass_ppa": 0.05,
        "rush_ppa": 0.05,
        "std_sr": 0.10,
    },
    DEF_WEIGHTS = {
        "ppa": 0.35,              # lower is better (invert)
        "success": 0.20,          # lower is better (invert)
        "explosiveness": 0.10,    # lower is better (invert)
        "ppo": 0.10,              # lower is better (invert)
        "power": 0.05,            # lower is better (invert)
        "rush_ppa": 0.07,         # lower is better (invert)
        "pass_ppa": 0.07,         # lower is better (invert)
        "havoc": 0.06,            # higher is better (keep sign)
    },
)

def load_config(path="config.json") -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

_cfg = load_config("config.json")

# Back-compat for flat configs
for key in ["DEFAULT_HFA_POINTS","SPREAD_SIGMA_PTS","TOTALS_SIGMA_PTS","MARKET_BLEND",
            "EV_THRESHOLDS","ADV_OFF_WEIGHTS","ADV_DEF_WEIGHTS","MIN_PPD",
            "BASE_POINTS_PER_DRIVE","EFF_TO_PPD_K"]:
    if key in _cfg:
        if key in ["ADV_OFF_WEIGHTS"]: DEFAULTS["OFF_WEIGHTS"] = _cfg[key]
        elif key in ["ADV_DEF_WEIGHTS"]: DEFAULTS["DEF_WEIGHTS"] = _cfg[key]
        else: DEFAULTS[key] = _cfg[key]

# Also accept nested "constants"
for k, v in _cfg.get("constants", {}).items():
    DEFAULTS[k] = v

# Canonicalize weight names (accepts synonyms)
def _canonicalize_weights(raw: Optional[dict], defaults: dict, kind: str) -> dict:
    if raw is None: raw = {}
    w = dict(defaults)
    if kind == "off":
        syn = {
            "ppa": ["ppa","off_ppa"],
            "success": ["success","successRate"],
            "explosiveness": ["explosiveness","explosive"],
            "power": ["power","powerSuccess"],
            "ppo": ["ppo","pointsPerOpportunity"],
            "std_sr": ["std_sr","sd_success","standard_downs","standardDowns.successRate"],
            "rush_ppa": ["rush_ppa","rushing_ppa","offense.rushingPlays.ppa","rush"],
            "pass_ppa": ["pass_ppa","passing_ppa","offense.passingPlays.ppa","pass"],
        }
    else:
        syn = {
            "ppa": ["ppa","def_ppa"],
            "success": ["success","successRate"],
            "explosiveness": ["explosiveness","explosive"],
            "power": ["power","powerSuccess"],
            "ppo": ["ppo","pointsPerOpportunity"],
            "rush_ppa": ["rush_ppa","rushing_ppa","defense.rushingPlays.ppa","rush"],
            "pass_ppa": ["pass_ppa","passing_ppa","defense.passingPlays.ppa","pass"],
            "havoc": ["havoc","havoc.total"],
        }
    for canon, aliases in syn.items():
        for a in aliases:
            if a in raw:
                w[canon] = float(raw[a]); break
    for k in defaults.keys():
        if k not in w: w[k] = defaults[k]
    return w

raw_off = _cfg.get("ADV_OFF_WEIGHTS", _cfg.get("weights", {}).get("off"))
raw_def = _cfg.get("ADV_DEF_WEIGHTS", _cfg.get("weights", {}).get("def"))
OFF_WEIGHTS = _canonicalize_weights(raw_off, DEFAULTS["OFF_WEIGHTS"], kind="off")
DEF_WEIGHTS = _canonicalize_weights(raw_def, DEFAULTS["DEF_WEIGHTS"], kind="def")

MARKET_BLEND = _cfg.get("MARKET_BLEND", _cfg.get("market_blend", DEFAULTS["MARKET_BLEND"]))
EV_THRESHOLDS = _cfg.get("EV_THRESHOLDS", _cfg.get("ev_thresholds", DEFAULTS["EV_THRESHOLDS"]))

SPREAD_SIGMA_PTS = float(DEFAULTS["SPREAD_SIGMA_PTS"])
TOTALS_SIGMA_PTS = float(DEFAULTS["TOTALS_SIGMA_PTS"])
DEFAULT_HFA_POINTS = float(DEFAULTS["DEFAULT_HFA_POINTS"])
BASE_POINTS_PER_DRIVE = float(DEFAULTS["BASE_POINTS_PER_DRIVE"])
PLAYS_PER_DRIVE = float(DEFAULTS["PLAYS_PER_DRIVE"])
DEFAULT_TEMPO = float(DEFAULTS["DEFAULT_TEMPO"])
EFF_TO_PPD_K = float(DEFAULTS["EFF_TO_PPD_K"])
MIN_PPD = float(DEFAULTS["MIN_PPD"])

WX_WIND_TEMPO_PER_MPH = float(DEFAULTS["WX_WIND_TEMPO_PER_MPH"])
WX_WIND_PPD_PER_MPH   = float(DEFAULTS["WX_WIND_PPD_PER_MPH"])
WX_RAIN_TEMPO = float(DEFAULTS["WX_RAIN_TEMPO"])
WX_SNOW_TEMPO = float(DEFAULTS["WX_SNOW_TEMPO"])
WX_RAIN_PPD = float(DEFAULTS["WX_RAIN_PPD"])
WX_SNOW_PPD = float(DEFAULTS["WX_SNOW_PPD"])
WX_COLD_PPD = float(DEFAULTS["WX_COLD_PPD"])
WX_HOT_PPD = float(DEFAULTS["WX_HOT_PPD"])

DEFAULT_BOOK_ODDS = int(DEFAULTS["DEFAULT_BOOK_ODDS"])

# ========== DATA TYPES ==========

@dataclass
class AdvTeam:
    name: str
    off_ppa: float; off_sr: float; off_expl: float; off_power: float; off_ppo: float
    off_fp_start: float; off_std_sr: float; off_pd_ppa: float; off_rush_ppa: float; off_pass_ppa: float
    def_ppa: float; def_sr: float; def_expl: float; def_power: float; def_stuff: float
    def_ppo: float; def_fp_start: float; def_havoc: float; def_rush_ppa: float; def_pass_ppa: float

@dataclass
class GameWeather:
    is_indoor: bool = False
    temp_f: float = 70.0
    wind_mph: float = 5.0
    precip: str = "none"  # none|light|rain|snow

# ========== UTILS ==========

def z(v: float, m: float, s: float) -> float:
    s = s if (s and s > 1e-12) else 1.0
    return (v - m) / s

def clamp(x, lo, hi): return max(lo, min(hi, x))

def american_to_decimal(odds: float) -> float:
    if odds >= 100:  return 1.0 + odds / 100.0
    if odds <= -100: return 1.0 + 100.0 / abs(odds)
    raise ValueError("American odds should be ≤ -100 or ≥ +100")

def break_even_prob(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)

def expected_value_pct(p: float, odds: float) -> float:
    dec = american_to_decimal(odds)
    return p * (dec - 1.0) - (1.0 - p)

def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _fmt_pct(p: float) -> str:
    p = max(0.010, min(0.990, float(p)))   # display clamp [1%, 99%]
    return f"{p*100:.1f}%"

# ========== CSV INGEST (advanced schema) ==========

ADV_COLS = [
    "team",
    "offense.ppa", "offense.successRate", "offense.explosiveness", "offense.powerSuccess",
    "offense.pointsPerOpportunity", "offense.fieldPosition.averageStart",
    "offense.standardDowns.successRate", "offense.passingDowns.ppa",
    "offense.rushingPlays.ppa", "offense.passingPlays.ppa",
    "defense.ppa", "defense.successRate", "defense.explosiveness", "defense.powerSuccess",
    "defense.stuffRate", "defense.pointsPerOpportunity", "defense.fieldPosition.averageStart",
    "defense.havoc.total", "defense.rushingPlays.ppa", "defense.passingPlays.ppa"
]

def load_csv_df(upload_or_path) -> pd.DataFrame:
    if isinstance(upload_or_path, str):
        df = pd.read_csv(upload_or_path)
    else:
        df = pd.read_csv(upload_or_path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in ADV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df

def mk_team(row: pd.Series) -> AdvTeam:
    g = lambda c, d=0.0: float(row.get(c, d))
    return AdvTeam(
        name=str(row["team"]),
        off_ppa=g("offense.ppa"),
        off_sr=g("offense.successRate"),
        off_expl=g("offense.explosiveness"),
        off_power=g("offense.powerSuccess"),
        off_ppo=g("offense.pointsPerOpportunity"),
        off_fp_start=g("offense.fieldPosition.averageStart"),
        off_std_sr=g("offense.standardDowns.successRate"),
        off_pd_ppa=g("offense.passingDowns.ppa"),
        off_rush_ppa=g("offense.rushingPlays.ppa"),
        off_pass_ppa=g("offense.passingPlays.ppa"),
        def_ppa=g("defense.ppa"),
        def_sr=g("defense.successRate"),
        def_expl=g("defense.explosiveness"),
        def_power=g("defense.powerSuccess"),
        def_stuff=g("defense.stuffRate"),
        def_ppo=g("defense.pointsPerOpportunity"),
        def_fp_start=g("defense.fieldPosition.averageStart"),
        def_havoc=g("defense.havoc.total"),
        def_rush_ppa=g("defense.rushingPlays.ppa"),
        def_pass_ppa=g("defense.passingPlays.ppa"),
    )

def league_baselines(df: pd.DataFrame) -> Dict[str, Tuple[float,float]]:
    out = {}
    for c in ADV_COLS:
        if c == "team": continue
        s = df[c].astype(float)
        out[c] = (float(np.nanmean(s)), float(np.nanstd(s, ddof=0)))
    return out

# ========== MODEL (advanced PPA) ==========

def offense_index(t: AdvTeam, base: Dict[str,Tuple[float,float]]) -> float:
    z_ppa   = z(t.off_ppa,   *base["offense.ppa"])
    z_sr    = z(t.off_sr,    *base["offense.successRate"])
    z_expl  = z(t.off_expl,  *base["offense.explosiveness"])
    z_power = z(t.off_power, *base["offense.powerSuccess"])
    z_ppo   = z(t.off_ppo,   *base["offense.pointsPerOpportunity"])
    z_std   = z(t.off_std_sr,*base["offense.standardDowns.successRate"])
    z_rppa  = z(t.off_rush_ppa,*base["offense.rushingPlays.ppa"])
    z_pppa  = z(t.off_pass_ppa,*base["offense.passingPlays.ppa"])
    w = OFF_WEIGHTS
    return (
        w["ppa"]*z_ppa + w["success"]*z_sr + w["explosiveness"]*z_expl +
        w["ppo"]*z_ppo + w["power"]*z_power + w["std_sr"]*z_std +
        w["rush_ppa"]*z_rppa + w["pass_ppa"]*z_pppa
    )

def defense_index(t: AdvTeam, base: Dict[str,Tuple[float,float]]) -> float:
    inv  = lambda val, mean,std: -z(val, mean, std)   # lower/better -> higher after inversion
    keep = lambda val, mean,std:  z(val, mean, std)   # higher/better -> higher (havoc)
    z_ppa   = inv(t.def_ppa,   *base["defense.ppa"])
    z_sr    = inv(t.def_sr,    *base["defense.successRate"])
    z_expl  = inv(t.def_expl,  *base["defense.explosiveness"])
    z_power = inv(t.def_power, *base["defense.powerSuccess"])
    z_ppo   = inv(t.def_ppo,   *base["defense.pointsPerOpportunity"])
    z_rppa  = inv(t.def_rush_ppa,*base["defense.rushingPlays.ppa"])
    z_pppa  = inv(t.def_pass_ppa,*base["defense.passingPlays.ppa"])
    z_havoc = keep(t.def_havoc,*base["defense.havoc.total"])
    w = DEF_WEIGHTS
    return (
        w["ppa"]*z_ppa + w["success"]*z_sr + w["explosiveness"]*z_expl +
        w["ppo"]*z_ppo + w["power"]*z_power +
        w["rush_ppa"]*z_rppa + w["pass_ppa"]*z_pppa +
        w["havoc"]*z_havoc
    )

def weather_tempo_multiplier(wx: GameWeather) -> float:
    if wx.is_indoor: return 1.0
    mult = 1.0 + WX_WIND_TEMPO_PER_MPH * max(0.0, wx.wind_mph)
    if wx.precip in ("rain","snow"):
        mult *= (1.0 + (WX_RAIN_TEMPO if wx.precip=="rain" else WX_SNOW_TEMPO))
    return clamp(mult, 0.85, 1.10)

def weather_ppd_multiplier(wx: GameWeather) -> float:
    if wx.is_indoor: return 1.0
    mult = 1.0 + WX_WIND_PPD_PER_MPH * max(0.0, wx.wind_mph)
    if wx.precip == "rain": mult *= (1.0 + WX_RAIN_PPD)
    if wx.precip == "snow": mult *= (1.0 + WX_SNOW_PPD)
    if wx.temp_f < 40: mult *= (1.0 + WX_COLD_PPD)
    elif wx.temp_f > 90: mult *= (1.0 + WX_HOT_PPD)
    return clamp(mult, 0.70, 1.05)

def situational_mult(off_sr: float, def_sr: float, off_ppo: float, def_ppo: float,
                     base: Dict[str,Tuple[float,float]]) -> float:
    m_sr  = 1.0 + 0.06 * ( z(off_sr, *base["offense.successRate"]) - z(def_sr, *base["defense.successRate"]) )
    m_ppo = 1.0 + 0.04 * ( z(off_ppo,*base["offense.pointsPerOpportunity"]) - z(def_ppo,*base["defense.pointsPerOpportunity"]) )
    return clamp(m_sr, 0.90, 1.10) * clamp(m_ppo, 0.95, 1.07)

def ppd_raw(off_idx: float, def_idx: float, base_ppd: float = None, k: float = None) -> float:
    """Return *raw* PPD before multipliers; floor is applied later."""
    if base_ppd is None: base_ppd = BASE_POINTS_PER_DRIVE
    if k is None: k = EFF_TO_PPD_K
    gap = off_idx - def_idx
    return base_ppd * math.exp(k * gap)

def predict_game(home: AdvTeam, away: AdvTeam, wx: GameWeather,
                 base: Dict[str,Tuple[float,float]],
                 hfa_points: float,
                 vegas_spread: Optional[float],
                 vegas_total: Optional[float],
                 use_market: bool, spread_w: float, total_w: float) -> Dict[str,float]:

    h_off = offense_index(home, base); a_off = offense_index(away, base)
    h_def = defense_index(home, base);  a_def = defense_index(away, base)

    # Drives / tempo
    tempo_mult = weather_tempo_multiplier(wx)
    plays = tempo_mult * (DEFAULT_TEMPO + DEFAULT_TEMPO) * 0.96
    total_drives = max(10.0, plays / PLAYS_PER_DRIVE)
    den = max(1e-6, 2*DEFAULT_TEMPO)
    drives_home = total_drives * (DEFAULT_TEMPO / den)
    drives_away = total_drives - drives_home

    # Base PPDs (apply multipliers, then floor)
    wx_ppd = weather_ppd_multiplier(wx)

    h_mult = wx_ppd * situational_mult(home.off_sr, away.def_sr, home.off_ppo, away.def_ppo, base)
    a_mult = wx_ppd * situational_mult(away.off_sr, home.def_sr, away.off_ppo, home.def_ppo, base)

    h_ppd_eff = max(MIN_PPD, ppd_raw(h_off, a_def) * h_mult)
    a_ppd_eff = max(MIN_PPD, ppd_raw(a_off, h_def) * a_mult)

    h_pts = max(0.0, h_ppd_eff * drives_home)
    a_pts = max(0.0, a_ppd_eff * drives_away)

    model_total = h_pts + a_pts
    model_spread = (h_pts - a_pts) + hfa_points  # Home − Away

    # Final (market blend)
    final_spread = model_spread
    final_total = model_total
    if use_market and (vegas_spread is not None or vegas_total is not None):
        if vegas_spread is not None:
            final_spread = (1.0 - spread_w) * model_spread + spread_w * vegas_spread
        if vegas_total is not None:
            final_total = (1.0 - total_w) * model_total + total_w * vegas_total

    # Convert spread/total → points
    model_home_pts = max(0.0, (model_total + model_spread)/2.0)
    model_away_pts = max(0.0, model_total - model_home_pts)
    final_home_pts = max(0.0, (final_total + final_spread)/2.0)
    final_away_pts = max(0.0, final_total - final_home_pts)

    p_home_model = phi(model_spread / SPREAD_SIGMA_PTS)
    p_home_final = phi(final_spread / SPREAD_SIGMA_PTS)

    return {
        "home_team": home.name, "away_team": away.name,
        "model_spread": round(model_spread, 1), "model_total": round(model_total, 1),
        "model_home_pts": round(model_home_pts, 1), "model_away_pts": round(model_away_pts, 1),
        "p_home_model": p_home_model,
        "final_spread": round(final_spread, 1), "final_total": round(final_total, 1),
        "final_home_pts": round(final_home_pts, 1), "final_away_pts": round(final_away_pts, 1),
        "p_home_final": p_home_final,
        "drives_home": round(drives_home, 1), "drives_away": round(drives_away, 1),
        "h_ppd": round(h_ppd_eff, 3), "a_ppd": round(a_ppd_eff, 3),
        "tempo_mult": round(tempo_mult, 3), "wx_ppd_mult": round(wx_ppd, 3),
    }

# ========== SIGNALS ==========

def spread_signal(model_spread: float, vegas_line: float, odds: int = DEFAULT_BOOK_ODDS):
    """
    vegas_line is Home−Away in points (positive = home favorite).
    Home covers if margin M > L. With M ~ N(model_spread, σ^2):
      P(home covers) = Φ((model_spread - vegas_line) / σ)
      P(away covers) = 1 - above
    Choose side with higher EV at the given odds.
    """
    sigma = SPREAD_SIGMA_PTS
    p_home_cover = phi((model_spread - vegas_line) / sigma)
    p_away_cover = 1.0 - p_home_cover

    ev_home = expected_value_pct(p_home_cover, odds)
    ev_away = expected_value_pct(p_away_cover, odds)

    # Edge direction by model vs line
    edge_val = model_spread - vegas_line
    edge_pts = round(abs(edge_val), 1)

    if ev_home >= ev_away:
        side = "HOME"; p_reco = p_home_cover; ev = ev_home
    else:
        side = "AWAY"; p_reco = p_away_cover; ev = ev_away

    tier = "PASS"
    if ev > 0:
        if ev >= EV_THRESHOLDS["strong"]: tier = "STRONG"
        elif ev >= EV_THRESHOLDS["lean"]: tier = "LEAN"

    return dict(
        vegas_line=vegas_line,
        model_spread=round(model_spread,1),
        edge_pts=edge_pts,
        side=side,
        p_cover_reco=p_reco,
        ev_pct=ev,
        tier=tier
    )

def total_signal(model_total: float, vegas_total: float, odds: int = DEFAULT_BOOK_ODDS):
    edge = model_total - vegas_total
    side = "OVER" if edge > 0 else "UNDER"
    zedge = edge / TOTALS_SIGMA_PTS
    p_over = phi(zedge)
    p_hit_reco = p_over if side == "OVER" else (1.0 - p_over)
    ev = expected_value_pct(p_hit_reco, odds)
    tier = "PASS"
    if ev > 0:
        if ev >= EV_THRESHOLDS["strong"]: tier = "STRONG"
        elif ev >= EV_THRESHOLDS["lean"]: tier = "LEAN"

    return dict(
        vegas_total=vegas_total,
        model_total=round(model_total,1),
        edge_pts=round(abs(edge),1),
        side=side,
        p_hit_reco=p_hit_reco,
        ev_pct=ev,
        tier=tier
    )

# ========== STREAMLIT UI ==========

st.set_page_config(page_title="CFB Predictor — Advanced (PPA)", page_icon="🏈", layout="wide")
st.title("🏈 College Football Predictor — Advanced (PPA)")

# Persist last prediction
if "pred" not in st.session_state: st.session_state.pred = None

with st.sidebar:
    st.header("Upload weekly advanced CSV")
    up = st.file_uploader("Drag and drop file here", type=["csv"])
    csv_path = st.text_input("...or path to CSV (optional)")
    df = None
    if up is not None:
        df = load_csv_df(up); st.success(f"Loaded CSV with {len(df)} rows")
    elif csv_path.strip():
        try: df = load_csv_df(csv_path.strip()); st.success(f"Loaded CSV with {len(df)} rows")
        except Exception as e: st.error(f"Failed to load CSV: {e}")
    if df is not None: st.info("Schema detected: **advanced (PPA)**")

    st.markdown("---")
    st.header("Game Settings")
    hfa_points = st.slider("Home Field Advantage (pts)", 0.0, 10.0, float(DEFAULT_HFA_POINTS), 0.5)

    st.markdown("---")
    st.header("Market Blend")
    use_market = st.checkbox("Blend with Vegas numbers", bool(MARKET_BLEND.get("use", True)))
    vegas_spread_raw = st.text_input("Vegas spread input (e.g., 'Home -3', 'Away +6.5', 'pk', or +3.5)", "")
    vegas_total_raw = st.text_input("Vegas total (e.g., 51.5)", "")
    spread_w = st.slider("Weight toward Vegas spread", 0.0, 1.0, float(MARKET_BLEND.get("spread_weight", 0.5)), 0.05)
    total_w  = st.slider("Weight toward Vegas total", 0.0, 1.0, float(MARKET_BLEND.get("total_weight", 0.5)), 0.05)
    st.caption("Weights are applied only when the corresponding Vegas number is supplied.")

    st.markdown("---")
    st.header("Model Variance (advanced)")
    SPREAD_SIGMA_PTS = st.slider("Spread sigma (pts)", 10.0, 20.0, float(SPREAD_SIGMA_PTS), 0.5)
    TOTALS_SIGMA_PTS = st.slider("Totals sigma (pts)", 10.0, 24.0, float(TOTALS_SIGMA_PTS), 0.5)
    st.caption("Tip: CFB often calibrates around σspread≈14–16 and σtotal≈16–20.")

    st.markdown("---")
    st.header("Moneyline Odds (optional)")
    odds_home_raw = st.text_input("Home ML (e.g., -150 or +120)", "")
    odds_away_raw = st.text_input("Away ML (e.g., +200 or -180)", "")

def parse_spread(raw: str, home_name: str, away_name: str) -> Optional[float]:
    s = raw.strip().lower()
    if not s: return None
    if s in {"pk","pick","pickem","pick'em","0","0.0"}: return 0.0
    m = re.search(r'([+-]?\d+(?:\.\d+)?)', s)
    if not m: return None
    num = float(m.group(1))
    def has(text, word):
        w = word.strip().lower()
        if " " in w: return w in text
        return re.search(r'\b'+re.escape(w)+r'\b', text) is not None
    is_home = has(s,"home") or has(s,home_name)
    is_away = has(s,"away") or has(s,away_name)
    if not (is_home or is_away): return num
    if is_home:   return +abs(num) if num < 0 else -abs(num)  # "Home -3" => +3 ; "Home +3" => -3
    else:         return +abs(num) if num > 0 else -abs(num)  # "Away +3" => +3 ; "Away -3" => -3

def parse_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s: return None
    try: return float(s)
    except: return None

# Team selectors
colA, colB = st.columns(2)
with colA:
    st.subheader("Home Team")
    if df is not None:
        home_name = st.selectbox("Select home team", sorted(df["team"].astype(str).unique()))
        hrow = df[df["team"].astype(str) == home_name].iloc[0]
        home = mk_team(hrow)
    else:
        st.warning("Upload an advanced CSV to pick teams.")
        st.stop()
with colB:
    st.subheader("Away Team")
    away_name = st.selectbox("Select away team", sorted(df["team"].astype(str).unique()))
    arow = df[df["team"].astype(str) == away_name].iloc[0]
    away = mk_team(arow)

st.markdown("---")
st.subheader("Weather / Venue")
wx_cols = st.columns(4)
with wx_cols[0]: is_indoor = st.checkbox("Indoors / Roof", False)
with wx_cols[1]: temp_f = st.number_input("Temp (°F)", 0.0, 120.0, 70.0, 1.0)
with wx_cols[2]: wind_mph = st.number_input("Wind (mph)", 0.0, 50.0, 5.0, 0.5)
with wx_cols[3]: precip = st.selectbox("Precip", ["none", "light", "rain", "snow"])
wx = GameWeather(is_indoor=is_indoor, temp_f=temp_f, wind_mph=wind_mph, precip=precip)

# Run + persist
if st.button("Run Prediction"):
    try:
        base = league_baselines(df)
        vegas_spread = parse_spread(vegas_spread_raw, home.name, away.name) if use_market else None
        vegas_total  = parse_float(vegas_total_raw) if use_market else None
        st.session_state.oh = parse_float(odds_home_raw)
        st.session_state.oa = parse_float(odds_away_raw)
        st.session_state.pred = predict_game(
            home=home, away=away, wx=wx, base=base, hfa_points=float(hfa_points),
            vegas_spread=vegas_spread, vegas_total=vegas_total,
            use_market=use_market, spread_w=float(spread_w), total_w=float(total_w)
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.pred = None

pred = st.session_state.pred
oh = st.session_state.get("oh"); oa = st.session_state.get("oa")

if pred is not None:
    # ---------- HEADLINE ----------
    st.success(f"Projected (Final line): {pred['away_team']} {pred['final_away_pts']} @ {pred['home_team']} {pred['final_home_pts']}")

    f1,f2,f3,f4 = st.columns(4)
    f1.metric("Home Win Prob — Final", _fmt_pct(pred['p_home_final']))
    f2.metric("Final spread (Home−Away)", f"{pred['final_spread']}")
    f3.metric("Final total", f"{pred['final_total']}")
    f4.metric("Away Win Prob — Final", _fmt_pct(1.0 - pred['p_home_final']))

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Home Win Prob — Model", _fmt_pct(pred['p_home_model']))
    m2.metric("Model spread (Home−Away)", f"{pred['model_spread']}")
    m3.metric("Model total", f"{pred['model_total']}")
    m4.metric("Away Win Prob — Model", _fmt_pct(1.0 - pred['p_home_model']))

    with st.expander("Diagnostics"):
        d1, d2, d3 = st.columns(3)
        d1.write(f"Model spread (H−A): **{pred['model_spread']}**")
        d1.write(f"Model total: **{pred['model_total']}**")
        d1.write(f"Drives — Home: **{pred['drives_home']}**, Away: **{pred['drives_away']}**")
        d2.write(f"PPD — Home: **{pred['h_ppd']}**, Away: **{pred['a_ppd']}**")
        d2.write(f"WX multipliers — Tempo: **{pred['tempo_mult']}**, PPD: **{pred['wx_ppd_mult']}**")
        d3.write(f"Model pts: **{pred['home_team']} {pred['model_home_pts']}**, **{pred['away_team']} {pred['model_away_pts']}**")

        N = 2000
        margin_draws = np.random.normal(loc=float(pred["model_spread"]), scale=SPREAD_SIGMA_PTS, size=N)
        total_draws  = np.random.normal(loc=float(pred["model_total"]),  scale=TOTALS_SIGMA_PTS,  size=N)
        home_draws = np.maximum(0.0, (total_draws + margin_draws)/2.0)
        away_draws = np.maximum(0.0, total_draws - home_draws)
        p_mc = float(np.mean(home_draws > away_draws))
        d3.write(f"MC p(home wins) ~ **{p_mc*100:.1f}%** (for reference)")

        import matplotlib.pyplot as plt
        fig1 = plt.figure(); plt.hist(margin_draws, bins=40); plt.title("Simulated Margin (Home − Away)")
        st.pyplot(fig1)
        fig2 = plt.figure(); plt.hist(total_draws, bins=40); plt.title("Simulated Total Points")
        st.pyplot(fig2)

    # ---------- Signals basis ----------
    signals_basis = st.radio("Signals use:", ["Model lines", "Final (blended) lines"], horizontal=True, index=0)
    use_final_for_signals = (signals_basis == "Final (blended) lines")

    # ---------- Spread Signal ----------
    vegas_spread = parse_spread(vegas_spread_raw, pred["home_team"], pred["away_team"]) if use_market else None
    if vegas_spread is not None:
        st.subheader("Spread Signal (Model vs Vegas)")
        is_home_fav = vegas_spread > 0
        st.caption(f"Vegas says: **{pred['home_team'] if is_home_fav else pred['away_team']} {abs(vegas_spread):.1f}** (Home−Away {vegas_spread:+.1f})")

        basis_spread = float(pred["final_spread"] if use_final_for_signals else pred["model_spread"])
        st.caption(f"Signals based on **{'Final' if use_final_for_signals else 'Model'}** line (Home−Away {basis_spread:+.1f}).")

        s_sig = spread_signal(model_spread=basis_spread, vegas_line=float(vegas_spread), odds=DEFAULT_BOOK_ODDS)
        c1,c2,c3 = st.columns(3)
        c1.metric("Edge (pts) toward", f"{s_sig['side']} {s_sig['edge_pts']}")
        c2.metric("P(covers) — recommended", _fmt_pct(s_sig['p_cover_reco']))
        c3.metric("EV @ -110 — recommended", f"{s_sig['ev_pct']*100:.1f}%")
        st.info(f"Recommendation: **{s_sig['tier']}** → Bet **{s_sig['side']}**")

    # ---------- Total Signal ----------
    vegas_total = parse_float(vegas_total_raw) if use_market else None
    if vegas_total is not None:
        st.subheader("Total Signal (Model vs Vegas)")
        basis_total = float(pred["final_total"] if use_final_for_signals else pred["model_total"])
        st.caption(f"Signals based on **{'Final' if use_final_for_signals else 'Model'}** total ({basis_total:.1f}).")
        t_sig = total_signal(model_total=basis_total, vegas_total=float(vegas_total), odds=DEFAULT_BOOK_ODDS)
        c1,c2,c3 = st.columns(3)
        c1.metric("Edge (pts) toward", f"{t_sig['side']} {t_sig['edge_pts']}")
        c2.metric(f"P({t_sig['side']}) — recommended", _fmt_pct(t_sig['p_hit_reco']))
        c3.metric("EV @ -110 — recommended", f"{t_sig['ev_pct']*100:.1f}%")
        st.info(f"Recommendation: **{t_sig['tier']}** → Bet **{t_sig['side']}**")

    # ---------- Moneyline ----------
    if (oh is not None) or (oa is not None):
        st.subheader("Moneyline (Model)")
        if oh is not None:
            be = break_even_prob(oh); ev = expected_value_pct(pred["p_home_model"], oh)
            st.write(f"**Home {pred['home_team']}** — Model p: **{_fmt_pct(pred['p_home_model'])}**, Odds: **{oh}**, Break-even: **{be*100:.1f}%**, EV: **{ev*100:.1f}%**")
        if oa is not None:
            p_away = 1.0 - pred["p_home_model"]; be = break_even_prob(oa); ev = expected_value_pct(p_away, oa)
            st.write(f"**Away {pred['away_team']}** — Model p: **{_fmt_pct(p_away)}**, Odds: **{oa}**, Break-even: **{be*100:.1f}%**, EV: **{ev*100:.1f}%**")

    # ---------- Download ----------
    out = {
        "home_team": pred["home_team"], "away_team": pred["away_team"],
        "final_home_pts": pred["final_home_pts"], "final_away_pts": pred["final_away_pts"],
        "final_spread": pred["final_spread"], "final_total": pred["final_total"],
        "p_home_final": round(pred["p_home_final"], 4),
        "model_home_pts": pred["model_home_pts"], "model_away_pts": pred["model_away_pts"],
        "model_spread": pred["model_spread"], "model_total": pred["model_total"],
        "p_home_model": round(pred["p_home_model"], 4),
    }
    out_df = pd.DataFrame([out])
    st.download_button(
        "Download this prediction as CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cfb_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
