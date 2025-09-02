# app.py — CFB Predictor (Advanced PPA)
# Blended build: v4_fixed + recent GUI upgrades
# - Dual KPI rows (Model vs Final) + signals-basis toggle
# - Session-state persistence (results don't clear when toggling)
# - Neutral-site toggle (zeros HFA)
# - Batch mode (schedule CSV)
# - Robust spread parser (sign fixed; Home−Away convention)
# - Variance sliders (σspread, σtotal) affect probabilities only
# - PPD floor moved up (default 0.90) to avoid ultra-low totals
# - EFF_TO_PPD_K softened (default 0.09) to reduce extremes
# - Config-driven overrides via config.json (optional)

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from datetime import datetime
import math, os, re, json

import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# Config & defaults
# ==============================

def load_config(path: str = "config.json") -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

DEFAULTS = dict(
    # Weather impact coefficients
    WX_WIND_PPD_PER_MPH = -0.008,
    WX_WIND_TEMPO_PER_MPH = -0.0015,
    WX_RAIN_PPD = -0.06,
    WX_SNOW_PPD = -0.10,
    WX_RAIN_TEMPO = -0.035,
    WX_SNOW_TEMPO = -0.05,
    WX_COLD_PPD = -0.03,
    WX_HOT_PPD  = -0.02,

    # Betting/simulation defaults
    DEFAULT_HFA_POINTS = 2.5,
    SPREAD_SIGMA_PTS   = 14.0,   # UI adjustable
    TOTALS_SIGMA_PTS   = 18.0,   # UI adjustable
    DEFAULT_BOOK_ODDS  = -110,
    EV_THRESHOLDS = {"strong": 0.05, "lean": 0.02},

    # Points-per-drive model
    BASE_POINTS_PER_DRIVE = 2.28,
    EFF_TO_PPD_K = 0.09,         # softened from 0.11
    MIN_PPD = 0.90,              # raised from 0.80

    # Tempo/drives proxy
    AVG_TEMPO = 65.8,
    PLAYS_PER_DRIVE = 5.8,

    # Turnover estimation
    MARGIN_PER_NET_TO  = 3.2,
    TO_TAKE_K   = 0.15,
    TO_GIVE_K   = 0.10,
    TO_PRESSURE = 0.10,
    TO_CLAMP    = 2.5,

    # Market blend defaults
    MARKET_BLEND = {"use": True, "spread_weight": 0.35, "total_weight": 0.35},

    # Advanced index weights (z-score weights)
    ADV_OFF_WEIGHTS = {
        "ppa": 0.40,
        "success": 0.22,
        "explosive": 0.12,
        "power": 0.06,
        "ppo": 0.12,
        "sd_success": 0.08,
    },
    ADV_DEF_WEIGHTS = {
        "ppa": 0.42,          # lower is better (invert)
        "success": 0.20,      # lower is better (invert)
        "explosive": 0.14,    # lower is better (invert)
        "power": 0.06,        # lower is better (invert)
        "ppo": 0.10,          # lower is better (invert)
        "havoc": 0.06,        # higher is better
        "stuff": 0.02,
    },

    # Situational multipliers
    POWER_MULT_CAP = (0.90, 1.10),
    POWER_MULT_COEF = 0.05,
    PPO_MULT_CAP = (0.88, 1.15),
    PPO_MULT_EXP = 0.50,
    RPP_MULT_CAP = (0.90, 1.10),
    RPP_MULT_COEF = 0.04,
    FP_MULT_CAP = (0.97, 1.03),
    FP_MULT_COEF = 0.01,
    TEMPO_SUCCESS_COEF = 0.06,
    TEMPO_EXPLOSIVE_COEF = -0.02,

    LEAGUE_STATIC = {
        "AVG_PPO": 4.0,  "STD_PPO": 0.7,
        "AVG_FP_START": 70.0, "STD_FP_START": 3.0,
    },
)

_cfg = load_config("config.json")
# Allow flat & nested overrides
for k, v in _cfg.get("constants", {}).items():
    DEFAULTS[k] = v
for key in ["MARKET_BLEND","EV_THRESHOLDS","ADV_OFF_WEIGHTS","ADV_DEF_WEIGHTS","LEAGUE_STATIC"]:
    if key in _cfg: DEFAULTS[key] = {**DEFAULTS[key], **_cfg[key]}
# Flat top-level fallbacks
for key in ["DEFAULT_HFA_POINTS","SPREAD_SIGMA_PTS","TOTALS_SIGMA_PTS",
            "BASE_POINTS_PER_DRIVE","EFF_TO_PPD_K","MIN_PPD",
            "AVG_TEMPO","PLAYS_PER_DRIVE","MARGIN_PER_NET_TO",
            "TO_TAKE_K","TO_GIVE_K","TO_PRESSURE","TO_CLAMP",
            "WX_WIND_PPD_PER_MPH","WX_WIND_TEMPO_PER_MPH","WX_RAIN_PPD","WX_SNOW_PPD",
            "WX_RAIN_TEMPO","WX_SNOW_TEMPO","WX_COLD_PPD","WX_HOT_PPD",
            "DEFAULT_BOOK_ODDS"]:
    if key in _cfg: DEFAULTS[key] = _cfg[key]

# Pull out constants
WX_WIND_PPD_PER_MPH   = float(DEFAULTS["WX_WIND_PPD_PER_MPH"])
WX_WIND_TEMPO_PER_MPH = float(DEFAULTS["WX_WIND_TEMPO_PER_MPH"])
WX_RAIN_PPD = float(DEFAULTS["WX_RAIN_PPD"])
WX_SNOW_PPD = float(DEFAULTS["WX_SNOW_PPD"])
WX_RAIN_TEMPO = float(DEFAULTS["WX_RAIN_TEMPO"])
WX_SNOW_TEMPO = float(DEFAULTS["WX_SNOW_TEMPO"])
WX_COLD_PPD = float(DEFAULTS["WX_COLD_PPD"])
WX_HOT_PPD  = float(DEFAULTS["WX_HOT_PPD"])

DEFAULT_HFA_POINTS = float(DEFAULTS["DEFAULT_HFA_POINTS"])
SPREAD_SIGMA_PTS   = float(DEFAULTS["SPREAD_SIGMA_PTS"])
TOTALS_SIGMA_PTS   = float(DEFAULTS["TOTALS_SIGMA_PTS"])
DEFAULT_BOOK_ODDS  = int(DEFAULTS["DEFAULT_BOOK_ODDS"])
EV_THRESHOLDS      = DEFAULTS["EV_THRESHOLDS"]

BASE_POINTS_PER_DRIVE = float(DEFAULTS["BASE_POINTS_PER_DRIVE"])
EFF_TO_PPD_K = float(DEFAULTS["EFF_TO_PPD_K"])
MIN_PPD = float(DEFAULTS["MIN_PPD"])

AVG_TEMPO = float(DEFAULTS["AVG_TEMPO"])
PLAYS_PER_DRIVE = float(DEFAULTS["PLAYS_PER_DRIVE"])

MARGIN_PER_NET_TO = float(DEFAULTS["MARGIN_PER_NET_TO"])
TO_TAKE_K   = float(DEFAULTS["TO_TAKE_K"])
TO_GIVE_K   = float(DEFAULTS["TO_GIVE_K"])
TO_PRESSURE = float(DEFAULTS["TO_PRESSURE"])
TO_CLAMP    = float(DEFAULTS["TO_CLAMP"])

MARKET_BLEND = DEFAULTS["MARKET_BLEND"]
ADV_OFF_WEIGHTS = DEFAULTS["ADV_OFF_WEIGHTS"]
ADV_DEF_WEIGHTS = DEFAULTS["ADV_DEF_WEIGHTS"]
POWER_MULT_CAP = tuple(DEFAULTS["POWER_MULT_CAP"])
POWER_MULT_COEF = float(DEFAULTS["POWER_MULT_COEF"])
PPO_MULT_CAP = tuple(DEFAULTS["PPO_MULT_CAP"])
PPO_MULT_EXP = float(DEFAULTS["PPO_MULT_EXP"])
RPP_MULT_CAP = tuple(DEFAULTS["RPP_MULT_CAP"])
RPP_MULT_COEF = float(DEFAULTS["RPP_MULT_COEF"])
FP_MULT_CAP = tuple(DEFAULTS["FP_MULT_CAP"])
FP_MULT_COEF = float(DEFAULTS["FP_MULT_COEF"])
TEMPO_SUCCESS_COEF = float(DEFAULTS["TEMPO_SUCCESS_COEF"])
TEMPO_EXPLOSIVE_COEF = float(DEFAULTS["TEMPO_EXPLOSIVE_COEF"])
LEAGUE_STATIC = DEFAULTS["LEAGUE_STATIC"]

# ==============================
# Data model & helpers
# ==============================

ADV_COLS = [
    "team",
    "offense.ppa","offense.successRate","offense.explosiveness","offense.powerSuccess",
    "offense.pointsPerOpportunity","offense.fieldPosition.averageStart",
    "offense.standardDowns.successRate","offense.passingDowns.ppa",
    "offense.rushingPlays.ppa","offense.passingPlays.ppa",
    "defense.ppa","defense.successRate","defense.explosiveness","defense.powerSuccess",
    "defense.stuffRate","defense.pointsPerOpportunity","defense.fieldPosition.averageStart",
    "defense.havoc.total","defense.rushingPlays.ppa","defense.passingPlays.ppa"
]

@dataclass
class Team:
    name: str
    off_ppa: float
    def_ppa: float
    off_sr: Optional[float] = None
    off_expl: Optional[float] = None
    off_power: Optional[float] = None
    off_ppo: Optional[float] = None
    off_fp_start: Optional[float] = None
    off_sd_sr: Optional[float] = None
    off_pd_ppa: Optional[float] = None
    off_rush_ppa: Optional[float] = None
    off_pass_ppa: Optional[float] = None

    def_sr: Optional[float] = None
    def_expl: Optional[float] = None
    def_power: Optional[float] = None
    def_stuff: Optional[float] = None
    def_ppo: Optional[float] = None
    def_fp_start: Optional[float] = None
    def_havoc: Optional[float] = None
    def_rush_ppa: Optional[float] = None
    def_pass_ppa: Optional[float] = None

@dataclass
class Weather:
    is_indoor: bool = False
    temp_f: float = 70.0
    wind_mph: float = 5.0
    precip: str = "none"   # none|light|rain|snow

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def load_df(src) -> pd.DataFrame:
    df = pd.read_csv(src)
    if "team" not in df.columns:
        raise ValueError("Advanced CSV must include a 'team' column.")
    # Ensure all ADV_COLS exist
    for c in ADV_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def compute_baselines(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    bl: Dict[str, Dict[str, float]] = {}
    for c in [c for c in df.columns if c != "team"]:
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().any():
            m = float(series.mean(skipna=True))
            s = float(series.std(ddof=0, skipna=True))
            if s < 1e-9: s = 1.0
        else:
            m, s = 0.0, 1.0
        bl[c] = {"mean": m, "std": s}
    # fallbacks
    bl.setdefault("offense.pointsPerOpportunity", {"mean": LEAGUE_STATIC["AVG_PPO"], "std": LEAGUE_STATIC["STD_PPO"]})
    bl.setdefault("defense.pointsPerOpportunity", {"mean": LEAGUE_STATIC["AVG_PPO"], "std": LEAGUE_STATIC["STD_PPO"]})
    bl.setdefault("offense.fieldPosition.averageStart", {"mean": LEAGUE_STATIC["AVG_FP_START"], "std": LEAGUE_STATIC["STD_FP_START"]})
    bl.setdefault("defense.fieldPosition.averageStart", {"mean": LEAGUE_STATIC["AVG_FP_START"], "std": LEAGUE_STATIC["STD_FP_START"]})
    return bl

def zcol(v: Optional[float], bl: Dict[str, Dict[str, float]], col: str) -> float:
    if v is None or pd.isna(v): return 0.0
    m = bl.get(col, {}).get("mean", 0.0)
    s = bl.get(col, {}).get("std", 1.0)
    if s < 1e-9: s = 1.0
    return (float(v) - m) / s

# ==============================
# Indices & multipliers
# ==============================

def offense_index(t: Team, bl) -> float:
    w = ADV_OFF_WEIGHTS
    return (
        w["ppa"]      * zcol(t.off_ppa, bl, "offense.ppa") +
        w["success"]  * zcol(t.off_sr, bl, "offense.successRate") +
        w["explosive"]* zcol(t.off_expl, bl, "offense.explosiveness") +
        w["power"]    * zcol(t.off_power, bl, "offense.powerSuccess") +
        w["ppo"]      * zcol(t.off_ppo, bl, "offense.pointsPerOpportunity") +
        w["sd_success"]*zcol(t.off_sd_sr, bl, "offense.standardDowns.successRate")
    )

def defense_index(t: Team, bl) -> float:
    w = ADV_DEF_WEIGHTS
    return (
        w["ppa"]      * (-zcol(t.def_ppa, bl, "defense.ppa")) +
        w["success"]  * (-zcol(t.def_sr, bl, "defense.successRate")) +
        w["explosive"]* (-zcol(t.def_expl, bl, "defense.explosiveness")) +
        w["power"]    * (-zcol(t.def_power, bl, "defense.powerSuccess")) +
        w["ppo"]      * (-zcol(t.def_ppo, bl, "defense.pointsPerOpportunity")) +
        w["havoc"]    * ( zcol(t.def_havoc, bl, "defense.havoc.total")) +
        w.get("stuff",0.0) * ( zcol(t.def_stuff, bl, "defense.stuffRate"))
    )

def tempo_proxy(team: Team, bl) -> float:
    base = AVG_TEMPO
    adj = 1.0 + TEMPO_SUCCESS_COEF * zcol(team.off_sr, bl, "offense.successRate") \
              + TEMPO_EXPLOSIVE_COEF * zcol(team.off_expl, bl, "offense.explosiveness")
    return clamp(base * adj, 55.0, 80.0)

def weather_ppd_multiplier(wx: Weather) -> float:
    if wx.is_indoor: return 1.0
    mult = 1.0
    mult *= max(0.70, 1.0 + WX_WIND_PPD_PER_MPH * max(0.0, wx.wind_mph))
    if wx.precip == "rain": mult *= (1.0 + WX_RAIN_PPD)
    elif wx.precip == "snow": mult *= (1.0 + WX_SNOW_PPD)
    if wx.temp_f < 40: mult *= (1.0 + WX_COLD_PPD)
    elif wx.temp_f > 90: mult *= (1.0 + WX_HOT_PPD)
    return mult

def combined_plays(home: Team, away: Team, wx: Weather, bl) -> float:
    th = tempo_proxy(home, bl)
    ta = tempo_proxy(away, bl)
    raw = 0.96 * (th + ta)
    if not wx.is_indoor:
        wind_factor = 1.0 + WX_WIND_TEMPO_PER_MPH * max(0.0, wx.wind_mph)
        raw *= max(0.85, wind_factor)
        if wx.precip in ("rain","snow"):
            raw *= (1.0 + (WX_RAIN_TEMPO if wx.precip=="rain" else WX_SNOW_TEMPO))
    return raw

def situational_ppd_multiplier(off: Team, opp_def: Team, bl) -> float:
    pow_off = zcol(off.off_power, bl, "offense.powerSuccess")
    pow_def = zcol(opp_def.def_power, bl, "defense.powerSuccess")
    m_power = 1.0 + POWER_MULT_COEF * (pow_off - (-pow_def))
    m_power = clamp(m_power, *POWER_MULT_CAP)

    off_ppo = off.off_ppo if off.off_ppo is not None else LEAGUE_STATIC["AVG_PPO"]
    def_ppo = opp_def.def_ppo if opp_def.def_ppo is not None else LEAGUE_STATIC["AVG_PPO"]
    ppo_factor = (off_ppo / LEAGUE_STATIC["AVG_PPO"]) * (LEAGUE_STATIC["AVG_PPO"] / max(1e-6, def_ppo)) ** PPO_MULT_EXP
    m_ppo = clamp(ppo_factor, *PPO_MULT_CAP)

    rush_diff_z = 0.0; pass_diff_z = 0.0
    if off.off_rush_ppa is not None and opp_def.def_rush_ppa is not None:
        rush_diff_z = zcol(off.off_rush_ppa, bl, "offense.rushingPlays.ppa") - zcol(opp_def.def_rush_ppa, bl, "defense.rushingPlays.ppa")
    if off.off_pass_ppa is not None and opp_def.def_pass_ppa is not None:
        pass_diff_z = zcol(off.off_pass_ppa, bl, "offense.passingPlays.ppa") - zcol(opp_def.def_pass_ppa, bl, "defense.passingPlays.ppa")
    m_rpp = 1.0 + RPP_MULT_COEF * (0.5 * (rush_diff_z + pass_diff_z))
    m_rpp = clamp(m_rpp, *RPP_MULT_CAP)

    fp_off = zcol(off.off_fp_start, bl, "offense.fieldPosition.averageStart")
    fp_def = -zcol(opp_def.def_fp_start, bl, "defense.fieldPosition.averageStart")
    m_fp = clamp(1.0 + FP_MULT_COEF * (fp_off + fp_def), *FP_MULT_CAP)

    return m_power * m_ppo * m_rpp * m_fp

def expected_net_turnovers(home: Team, away: Team, bl) -> float:
    h_hav = zcol(home.def_havoc, bl, "defense.havoc.total")
    a_hav = zcol(away.def_havoc, bl, "defense.havoc.total")
    h_off = zcol(home.off_sr, bl, "offense.successRate")
    a_off = zcol(away.off_sr, bl, "offense.successRate")

    take_h = 1.45 + TO_TAKE_K * h_hav
    take_a = 1.45 + TO_TAKE_K * a_hav
    give_h = 1.45 - TO_GIVE_K * h_off + TO_PRESSURE * a_hav
    give_a = 1.45 - TO_GIVE_K * a_off + TO_PRESSURE * h_hav

    net = (take_h + give_a) - (give_h + take_a)
    return clamp(net, -TO_CLAMP, TO_CLAMP)

def ppd(off_idx: float, def_idx: float) -> float:
    gap = off_idx - def_idx
    val = BASE_POINTS_PER_DRIVE * math.exp(EFF_TO_PPD_K * gap)
    return max(MIN_PPD, val)

# ==============================
# Prediction pipeline
# ==============================

def build_team(row: pd.Series) -> Team:
    g = lambda k: (None if k not in row or pd.isna(row[k]) else float(row[k]))
    return Team(
        name=str(row["team"]),
        off_ppa=float(row["offense.ppa"]),
        def_ppa=float(row["defense.ppa"]),
        off_sr=g("offense.successRate"),
        off_expl=g("offense.explosiveness"),
        off_power=g("offense.powerSuccess"),
        off_ppo=g("offense.pointsPerOpportunity"),
        off_fp_start=g("offense.fieldPosition.averageStart"),
        off_sd_sr=g("offense.standardDowns.successRate"),
        off_pd_ppa=g("offense.passingDowns.ppa"),
        off_rush_ppa=g("offense.rushingPlays.ppa"),
        off_pass_ppa=g("offense.passingPlays.ppa"),
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

def predict_game(home: Team, away: Team, wx: Weather, bl,
                 hfa_points: float,
                 vegas_spread: Optional[float],
                 vegas_total: Optional[float],
                 use_market: bool, spread_w: float, total_w: float) -> Dict[str, float]:

    h_off = offense_index(home, bl); a_off = offense_index(away, bl)
    h_def = defense_index(home, bl);  a_def = defense_index(away, bl)

    plays = combined_plays(home, away, wx, bl)
    th = tempo_proxy(home, bl); ta = tempo_proxy(away, bl)
    total_drives = max(10.0, plays / PLAYS_PER_DRIVE)
    den = max(1e-6, th + ta)
    drives_home = total_drives * (th / den)
    drives_away = total_drives - drives_home

    wx_mult = weather_ppd_multiplier(wx)
    h_ppd = ppd(h_off, a_def) * wx_mult * situational_ppd_multiplier(home, away, bl)
    a_ppd = ppd(a_off, h_def) * wx_mult * situational_ppd_multiplier(away, home, bl)

    h_pts = max(0.0, h_ppd * drives_home)
    a_pts = max(0.0, a_ppd * drives_away)

    model_total = h_pts + a_pts
    model_spread = (h_pts - a_pts) + hfa_points  # Home − Away

    # turnovers → margin
    net_to = expected_net_turnovers(home, away, bl)
    model_spread += MARGIN_PER_NET_TO * net_to

    # Convert to team totals (model)
    model_home_pts = max(0.0, (model_total + model_spread)/2.0)
    model_away_pts = max(0.0, model_total - model_home_pts)

    # Final (market blend)
    final_spread = model_spread
    final_total  = model_total
    if use_market and (vegas_spread is not None or vegas_total is not None):
        if vegas_spread is not None:
            final_spread = (1.0 - spread_w) * model_spread + spread_w * vegas_spread
        if vegas_total is not None:
            final_total  = (1.0 - total_w) * model_total  + total_w  * vegas_total

    final_home_pts = max(0.0, (final_total + final_spread)/2.0)
    final_away_pts = max(0.0, final_total - final_home_pts)

    # Win probs from margins (Normal approx)
    def phi(x: float) -> float: return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
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
        "h_ppd": round(h_ppd, 3), "a_ppd": round(a_ppd, 3),
        "plays_est": round(plays, 1), "hfa_points": hfa_points,
    }

# ==============================
# Betting helpers (use global sigmas)
# ==============================

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

def spread_signal(model_spread: float, vegas_line: float, odds: int, sigma_pts: float) -> Dict:
    # Home covers if margin > line  ⇒ P = Phi((model - line)/sigma)
    edge = model_spread - vegas_line
    p_home_cover = phi(edge / sigma_pts)
    be = break_even_prob(odds)
    ev = expected_value_pct(p_home_cover, odds)
    side = "HOME" if p_home_cover > be else "AWAY"
    tier = "PASS"
    if ev > 0:
        if ev >= EV_THRESHOLDS["strong"]: tier = "STRONG"
        elif ev >= EV_THRESHOLDS["lean"]: tier = "LEAN"
    return dict(
        vegas_line=float(vegas_line),
        model_spread=round(model_spread,1),
        edge_pts=round(abs(edge),1),
        p_cover_reco=p_home_cover if side=="HOME" else (1.0 - p_home_cover),
        ev_pct=ev, side=side, tier=tier
    )

def total_signal(model_total: float, vegas_total: float, odds: int, sigma_pts: float) -> Dict:
    edge = model_total - vegas_total
    p_over = phi(edge / sigma_pts)
    be = break_even_prob(odds)
    p_hit = p_over if edge > 0 else (1.0 - p_over)
    side = "OVER" if edge > 0 else "UNDER"
    ev = expected_value_pct(p_hit, odds)
    tier = "PASS"
    if ev > 0:
        if ev >= EV_THRESHOLDS["strong"]: tier = "STRONG"
        elif ev >= EV_THRESHOLDS["lean"]: tier = "LEAN"
    return dict(
        vegas_total=float(vegas_total),
        model_total=round(model_total,1),
        edge_pts=round(abs(edge),1),
        p_hit_reco=p_hit, ev_pct=ev, side=side, tier=tier
    )

# ==============================
# Parsing helpers
# ==============================

def parse_spread_flex(raw: str, home_name: str, away_name: str) -> Optional[float]:
    """
    Return spread in Home−Away units. Examples:
      'Home -3.5' -> +3.5, 'Home +3' -> -3, 'Away +7' -> +7, 'Away -2' -> -2
      'pk'/'0' -> 0.0; raw '+3.5' or '-4' treated as Home−Away directly
    """
    s = (raw or "").strip().lower()
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
    if is_home and is_away: return None
    if not (is_home or is_away): return num  # already Home−Away
    if is_home:
        # "Home -3" means home favored by 3 ⇒ Home−Away = +3
        return +abs(num) if num < 0 else -abs(num)
    else:
        # "Away +3" means away is +3 dog ⇒ home favored by 3 ⇒ Home−Away = +3
        return +abs(num) if num > 0 else -abs(num)

def parse_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s: return None
    try: return float(s)
    except Exception: return None

def fmt_pct(p: float) -> str:
    p = max(0.010, min(0.990, float(p)))
    return f"{p*100:.1f}%"

# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="CFB Predictor — Advanced (PPA)", page_icon="🏈", layout="wide")
st.title("🏈 College Football Predictor — Advanced (PPA)")

# Persist last prediction + vegas odds in session
if "pred" not in st.session_state: st.session_state.pred = None
if "oh" not in st.session_state: st.session_state.oh = None
if "oa" not in st.session_state: st.session_state.oa = None
if "vs_spread" not in st.session_state: st.session_state.vs_spread = None
if "vs_total" not in st.session_state: st.session_state.vs_total = None

with st.sidebar:
    st.header("Data Source")
    up = st.file_uploader("Upload weekly advanced CSV", type=["csv"])
    csv_path = st.text_input("...or path to CSV (optional)")
    df = None
    if up is not None:
        df = load_df(up); st.success(f"Loaded CSV with {len(df)} rows")
    elif csv_path.strip():
        df = load_df(csv_path.strip()); st.success(f"Loaded CSV with {len(df)} rows")
    else:
        st.info("Upload an advanced CSV to proceed.")
    if df is not None: st.caption("Schema: **advanced (PPA)** — one row per team")

    st.markdown("---")
    st.header("Game Settings")
    base_hfa = st.slider("Home Field Advantage (pts)", 0.0, 10.0, float(DEFAULT_HFA_POINTS), 0.5)
    neutral = st.checkbox("Neutral site (zero HFA)", False)

    st.markdown("---")
    st.header("Market Blend")
    use_market = st.checkbox("Blend with Vegas numbers", bool(MARKET_BLEND.get("use", True)))
    vegas_spread_raw = st.text_input("Vegas spread (e.g., 'Home -3.5', 'Away +4', 'pk', or raw +/- like +3.5)", "")
    vegas_total_raw  = st.text_input("Vegas total (e.g., 54.5)", "")
    spread_w = st.slider("Weight toward Vegas spread", 0.0, 1.0, float(MARKET_BLEND.get("spread_weight", 0.35)), 0.05)
    total_w  = st.slider("Weight toward Vegas total",  0.0, 1.0, float(MARKET_BLEND.get("total_weight", 0.35)), 0.05)
    st.caption("Weights are used only if the corresponding Vegas number is provided.")

    st.markdown("---")
    st.header("Model Variance")
    SPREAD_SIGMA_PTS = st.slider("Spread σ (pts)", 10.0, 20.0, float(SPREAD_SIGMA_PTS), 0.5)
    TOTALS_SIGMA_PTS = st.slider("Total σ (pts)", 12.0, 24.0, float(TOTALS_SIGMA_PTS), 0.5)
    st.caption("Tip: CFB commonly calibrates around σspread≈14–16 and σtotal≈16–20.")

    st.markdown("---")
    st.header("Moneyline Odds (optional)")
    odds_home_raw = st.text_input("Home ML (e.g., -150 or +120)", "")
    odds_away_raw = st.text_input("Away ML (e.g., +200 or -180)", "")

if df is None:
    st.stop()

# Tabs
single_tab, batch_tab = st.tabs(["Single Game", "Batch Mode"])

with single_tab:
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Home Team")
        home_name = st.selectbox("Select home team", sorted(df["team"].astype(str).unique()))
        hrow = df[df["team"].astype(str) == home_name].iloc[0]
        home = build_team(hrow)
    with colB:
        st.subheader("Away Team")
        away_name = st.selectbox("Select away team", sorted(df["team"].astype(str).unique()))
        arow = df[df["team"].astype(str) == away_name].iloc[0]
        away = build_team(arow)

    st.markdown("---")
    st.subheader("Weather / Venue")
    wx_cols = st.columns(4)
    with wx_cols[0]: is_indoor = st.checkbox("Indoors / Roof", False)
    with wx_cols[1]: temp_f = st.number_input("Temp (°F)", 0.0, 120.0, 70.0, 1.0)
    with wx_cols[2]: wind_mph = st.number_input("Wind (mph)", 0.0, 50.0, 5.0, 0.5)
    with wx_cols[3]: precip = st.selectbox("Precip", ["none", "light", "rain", "snow"])
    wx = Weather(is_indoor=is_indoor, temp_f=temp_f, wind_mph=wind_mph, precip=precip)

    if st.button("Run Prediction"):
        try:
            BL = compute_baselines(df)
            vs = parse_spread_flex(vegas_spread_raw, home.name, away.name) if use_market else None
            vt = parse_float(vegas_total_raw) if use_market else None
            st.session_state.vs_spread = vs
            st.session_state.vs_total = vt
            st.session_state.oh = parse_float(odds_home_raw)
            st.session_state.oa = parse_float(odds_away_raw)
            hfa_points = 0.0 if neutral else float(base_hfa)

            st.session_state.pred = predict_game(
                home, away, wx, BL,
                hfa_points=hfa_points,
                vegas_spread=vs, vegas_total=vt,
                use_market=use_market, spread_w=float(spread_w), total_w=float(total_w)
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.pred = None

    pred = st.session_state.pred
    vs = st.session_state.vs_spread
    vt = st.session_state.vs_total
    oh = st.session_state.oh
    oa = st.session_state.oa

    if pred is not None:
        # Headline
        st.success(f"Projected (Final line): {pred['away_team']} {pred['final_away_pts']} @ {pred['home_team']} {pred['final_home_pts']}")

        # Final vs Model KPI rows
        f1,f2,f3,f4 = st.columns(4)
        f1.metric("Home Win Prob — Final", fmt_pct(pred['p_home_final']))
        f2.metric("Final spread (Home−Away)", f"{pred['final_spread']}")
        f3.metric("Final total", f"{pred['final_total']}")
        f4.metric("Away Win Prob — Final", fmt_pct(1.0 - pred['p_home_final']))

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Home Win Prob — Model", fmt_pct(pred['p_home_model']))
        m2.metric("Model spread (Home−Away)", f"{pred['model_spread']}")
        m3.metric("Model total", f"{pred['model_total']}")
        m4.metric("Away Win Prob — Model", fmt_pct(1.0 - pred['p_home_model']))

        with st.expander("Diagnostics"):
            d1, d2, d3 = st.columns(3)
            d1.write(f"Drives — Home: **{pred['drives_home']}**, Away: **{pred['drives_away']}**")
            d1.write(f"Plays (est): **{pred['plays_est']}**  |  HFA: **{pred['hfa_points']}**")
            d2.write(f"PPD — Home: **{pred['h_ppd']}**, Away: **{pred['a_ppd']}**")

            # quick MC (for reference only)
            N = 2000
            margin_draws = np.random.normal(float(pred["model_spread"]), SPREAD_SIGMA_PTS, size=N)
            total_draws  = np.random.normal(float(pred["model_total"]),  TOTALS_SIGMA_PTS,  size=N)
            p_mc = float(np.mean(margin_draws > 0))
            d3.write(f"MC p(home wins) ~ **{p_mc*100:.1f}%** (reference)")

            import matplotlib.pyplot as plt
            fig1 = plt.figure(); plt.hist(margin_draws, bins=40); plt.title("Simulated Margin (Home − Away)")
            st.pyplot(fig1)
            fig2 = plt.figure(); plt.hist(total_draws, bins=40); plt.title("Simulated Total Points")
            st.pyplot(fig2)

        # Signals basis
        basis_choice = st.radio("Signals use:", ["Model lines", "Final (blended) lines"], horizontal=True, index=0)
        use_final_for_signals = (basis_choice == "Final (blended) lines")

        # Spread signal
        if use_market and (vs is not None):
            st.subheader("Spread Signal (vs Vegas)")
            basis_spread = float(pred["final_spread"] if use_final_for_signals else pred["model_spread"])
            # friendly text for vegas line
            if vs > 0: vegas_txt = f"{pred['home_team']} -{abs(vs):.1f} (i.e., {pred['away_team']} +{abs(vs):.1f})"
            elif vs < 0: vegas_txt = f"{pred['away_team']} -{abs(vs):.1f} (i.e., {pred['home_team']} +{abs(vs):.1f})"
            else: vegas_txt = "Pick'em"
            st.caption(f"Vegas says: **{vegas_txt}**  |  Signals based on **{'Final' if use_final_for_signals else 'Model'}** line ({basis_spread:+.1f}).")

            s_sig = spread_signal(basis_spread, float(vs), DEFAULT_BOOK_ODDS, SPREAD_SIGMA_PTS)
            c1,c2,c3 = st.columns(3)
            c1.metric("Edge (pts) toward", f"{s_sig['side']} {s_sig['edge_pts']}")
            c2.metric("P(covers) — recommended", fmt_pct(s_sig['p_cover_reco']))
            c3.metric("EV @ -110 — recommended", f"{s_sig['ev_pct']*100:.1f}%")
            st.info(f"Recommendation: **{s_sig['tier']}** → Bet **{s_sig['side']}**")

        # Total signal
        if use_market and (vt is not None):
            st.subheader("Total Signal (vs Vegas)")
            basis_total = float(pred["final_total"] if use_final_for_signals else pred["model_total"])
            st.caption(f"Signals based on **{'Final' if use_final_for_signals else 'Model'}** total ({basis_total:.1f}).")
            t_sig = total_signal(basis_total, float(vt), DEFAULT_BOOK_ODDS, TOTALS_SIGMA_PTS)
            c1,c2,c3 = st.columns(3)
            c1.metric("Edge (pts) toward", f"{t_sig['side']} {t_sig['edge_pts']}")
            c2.metric(f"P({t_sig['side']}) — recommended", fmt_pct(t_sig['p_hit_reco']))
            c3.metric("EV @ -110 — recommended", f"{t_sig['ev_pct']*100:.1f}%")
            st.info(f"Recommendation: **{t_sig['tier']}** → Bet **{t_sig['side']}**")

        # Moneyline
        if (oh is not None) or (oa is not None):
            st.subheader("Moneyline (Model)")
            if oh is not None:
                be = break_even_prob(oh); ev = expected_value_pct(pred["p_home_model"], oh)
                st.write(f"**Home {pred['home_team']}** — Model p: **{fmt_pct(pred['p_home_model'])}**, Odds: **{oh}**, Break-even: **{be*100:.1f}%**, EV: **{ev*100:.1f}%**")
            if oa is not None:
                p_away = 1.0 - pred["p_home_model"]; be = break_even_prob(oa); ev = expected_value_pct(p_away, oa)
                st.write(f"**Away {pred['away_team']}** — Model p: **{fmt_pct(p_away)}**, Odds: **{oa}**, Break-even: **{be*100:.1f}%**, EV: **{ev*100:.1f}%**")

        # Download
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

with batch_tab:
    st.subheader("Batch mode: score many games")
    st.caption("Upload a schedule CSV with columns: home_team, away_team, optional: vegas_spread, vegas_total, neutral_site (True/False). Team names must match the advanced CSV.")
    sched_up = st.file_uploader("Upload schedule CSV", type=["csv"], key="sched")
    if sched_up is not None:
        try:
            sched = pd.read_csv(sched_up)
            for req in ["home_team","away_team"]:
                if req not in sched.columns:
                    st.error(f"Schedule missing required column: {req}")
                    st.stop()
            BL = compute_baselines(df)
            tmap = {str(r.team).strip(): build_team(r) for _, r in df.iterrows()}

            def parse_spread_simple(s) -> Optional[float]:
                if pd.isna(s): return None
                # treat as already Home−Away if team names aren't present
                t = str(s).strip()
                if t.lower() in {"pk","pick","pickem","pick'em","0","0.0"}: return 0.0
                m = re.search(r"([+-]?\d+(?:\.\d+)?)", t)
                return float(m.group(1)) if m else None

            rows = []
            for _, g in sched.iterrows():
                ht = str(g["home_team"]).strip(); at = str(g["away_team"]).strip()
                if ht not in tmap or at not in tmap:
                    rows.append({"home_team": ht, "away_team": at, "error": "Unknown team name"}); continue
                home = tmap[ht]; away = tmap[at]
                wx = Weather(is_indoor=False, temp_f=70.0, wind_mph=5.0, precip="none")

                vs_spread = parse_spread_simple(g.get("vegas_spread")) if use_market else None
                vs_total  = None
                if use_market and ("vegas_total" in g) and not pd.isna(g.get("vegas_total")):
                    try: vs_total = float(g.get("vegas_total"))
                    except: vs_total = None
                hfa_pts = 0.0 if bool(g.get("neutral_site", False)) else float(base_hfa)

                pred = predict_game(
                    home, away, wx, BL,
                    hfa_points=hfa_pts,
                    vegas_spread=vs_spread, vegas_total=vs_total,
                    use_market=use_market, spread_w=float(spread_w), total_w=float(total_w)
                )
                rows.append({**pred, "neutral_site": bool(g.get("neutral_site", False)),
                             "vegas_spread": vs_spread, "vegas_total": vs_total})

            out_df = pd.DataFrame(rows)
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Download batch predictions CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name=f"cfb_batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Batch error: {e}")
