# cfb_predictor_gui_v4_fixed.py — Clean Rebuild (indent-safe)
# Streamlit GUI for CFB predictor using weekly advanced CSV (one row per team)
#
# Key features:
#   • Neutral-site toggle (zeros HFA)
#   • Batch mode (upload schedule CSV, export results)
#   • Config-driven weights with safe fallbacks (config.json optional)
#   • Diagnostics + Monte Carlo for win prob
#   • Market blend (optional) for spread/total
#
# This build fixes indentation/try-except structure issues reported by users.

from dataclasses import dataclass
from typing import Optional, Dict
import math
import os
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# Config & global defaults
# ==============================

def load_config(path: str = "config.json") -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# Weather impact coefficients
WX_WIND_PPD_PER_MPH = -0.008
WX_WIND_TEMPO_PER_MPH = -0.0015
WX_RAIN_PPD = -0.06
WX_SNOW_PPD = -0.10
WX_RAIN_TEMPO = -0.035
WX_SNOW_TEMPO = -0.05
WX_COLD_PPD = -0.03
WX_HOT_PPD  = -0.02

# Betting/simulation defaults
DEFAULT_HFA_POINTS = 2.5
MARGIN_PER_NET_TO  = 3.2
SPREAD_SIGMA_PTS   = 13.0
TOTALS_SIGMA_PTS   = 11.0
DEFAULT_BOOK_ODDS  = -110
EV_THRESHOLDS = {"strong": 0.05, "lean": 0.02}

# Points-per-drive model
BASE_POINTS_PER_DRIVE = 2.28
EFF_TO_PPD_K = 0.11

# Tempo/drives proxy
AVG_TEMPO = 65.8
PLAYS_PER_DRIVE = 5.8

# Turnover estimation
TO_TAKE_K   = 0.15
TO_GIVE_K   = 0.10
TO_PRESSURE = 0.10
TO_CLAMP    = 2.5

# Market blend defaults
MARKET_BLEND = {"use": True, "spread_weight": 0.25, "total_weight": 0.25}

# Advanced index weights (z-score weights)
ADV_OFF_WEIGHTS = {
    "ppa": 0.45,
    "success": 0.25,
    "explosive": 0.15,
    "power": 0.05,
    "ppo": 0.07,
    "sd_success": 0.03,
}

ADV_DEF_WEIGHTS = {
    "ppa": 0.50,          # lower is better (invert)
    "success": 0.20,      # lower is better (invert)
    "explosive": 0.15,    # lower is better (invert)
    "power": 0.05,        # lower is better (invert)
    "ppo": 0.07,          # lower is better (invert)
    "havoc": 0.03,        # higher is better
    "stuff": 0.00,
}

# Situational multipliers
POWER_MULT_CAP = (0.90, 1.10)
POWER_MULT_COEF = 0.05
PPO_MULT_CAP = (0.85, 1.15)
PPO_MULT_EXP = 0.50
RPP_MULT_CAP = (0.90, 1.10)
RPP_MULT_COEF = 0.04
FP_MULT_CAP = (0.97, 1.03)
FP_MULT_COEF = 0.01
TEMPO_SUCCESS_COEF = 0.06
TEMPO_EXPLOSIVE_COEF = -0.02

LEAGUE_STATIC = {
    "AVG_PPO": 4.0,  "STD_PPO": 0.7,
    "AVG_FP_START": 70.0, "STD_FP_START": 3.0,
}

# Apply config overrides if present
_cfg = load_config("config.json")
DEFAULT_HFA_POINTS = _cfg.get("DEFAULT_HFA_POINTS", DEFAULT_HFA_POINTS)
MARKET_BLEND.update(_cfg.get("MARKET_BLEND", {}))
EV_THRESHOLDS.update(_cfg.get("EV_THRESHOLDS", {}))
ADV_OFF_WEIGHTS.update(_cfg.get("ADV_OFF_WEIGHTS", {}))
ADV_DEF_WEIGHTS.update(_cfg.get("ADV_DEF_WEIGHTS", {}))
LEAGUE_STATIC.update(_cfg.get("LEAGUE_STATIC", {}))

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


def load_df(src) -> pd.DataFrame:
    df = pd.read_csv(src)
    if "team" not in df.columns:
        raise ValueError("Advanced CSV must include a 'team' column.")
    for c in ADV_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def compute_baselines(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    bl: Dict[str, Dict[str, float]] = {}
    cols_for_stats = [c for c in df.columns if c != "team"]
    for c in cols_for_stats:
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().any():
            m = float(series.mean(skipna=True))
            s = float(series.std(ddof=0, skipna=True))
            if s < 1e-9:
                s = 1.0
        else:
            m, s = 0.0, 1.0
        bl[c] = {"mean": m, "std": s}
    bl.setdefault("offense.pointsPerOpportunity", {"mean": LEAGUE_STATIC["AVG_PPO"], "std": LEAGUE_STATIC["STD_PPO"]})
    bl.setdefault("defense.pointsPerOpportunity", {"mean": LEAGUE_STATIC["AVG_PPO"], "std": LEAGUE_STATIC["STD_PPO"]})
    bl.setdefault("offense.fieldPosition.averageStart", {"mean": LEAGUE_STATIC["AVG_FP_START"], "std": LEAGUE_STATIC["STD_FP_START"]})
    bl.setdefault("defense.fieldPosition.averageStart", {"mean": LEAGUE_STATIC["AVG_FP_START"], "std": LEAGUE_STATIC["STD_FP_START"]})
    return bl


def zcol(v: Optional[float], bl: Dict[str, Dict[str, float]], col: str) -> float:
    if v is None or pd.isna(v):
        return 0.0
    m = bl.get(col, {}).get("mean", 0.0)
    s = bl.get(col, {}).get("std", 1.0)
    if s < 1e-9:
        s = 1.0
    return (float(v) - m) / s


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ==============================
# Indices & multipliers
# ==============================

def offense_index(t: Team, bl) -> float:
    return (
        ADV_OFF_WEIGHTS["ppa"]      * zcol(t.off_ppa, bl, "offense.ppa") +
        ADV_OFF_WEIGHTS["success"]  * zcol(t.off_sr, bl, "offense.successRate") +
        ADV_OFF_WEIGHTS["explosive"]* zcol(t.off_expl, bl, "offense.explosiveness") +
        ADV_OFF_WEIGHTS["power"]    * zcol(t.off_power, bl, "offense.powerSuccess") +
        ADV_OFF_WEIGHTS["ppo"]      * zcol(t.off_ppo, bl, "offense.pointsPerOpportunity") +
        ADV_OFF_WEIGHTS["sd_success"]*zcol(t.off_sd_sr, bl, "offense.standardDowns.successRate")
    )


def defense_index(t: Team, bl) -> float:
    return (
        ADV_DEF_WEIGHTS["ppa"]      * (-zcol(t.def_ppa, bl, "defense.ppa")) +
        ADV_DEF_WEIGHTS["success"]  * (-zcol(t.def_sr, bl, "defense.successRate")) +
        ADV_DEF_WEIGHTS["explosive"]* (-zcol(t.def_expl, bl, "defense.explosiveness")) +
        ADV_DEF_WEIGHTS["power"]    * (-zcol(t.def_power, bl, "defense.powerSuccess")) +
        ADV_DEF_WEIGHTS["ppo"]      * (-zcol(t.def_ppo, bl, "defense.pointsPerOpportunity")) +
        ADV_DEF_WEIGHTS["havoc"]    * ( zcol(t.def_havoc, bl, "defense.havoc.total")) +
        ADV_DEF_WEIGHTS.get("stuff",0.0) * ( zcol(t.def_stuff, bl, "defense.stuffRate"))
    )


def tempo_proxy(team: Team, bl) -> float:
    base = AVG_TEMPO
    adj = 1.0 + TEMPO_SUCCESS_COEF * zcol(team.off_sr, bl, "offense.successRate") \
              + TEMPO_EXPLOSIVE_COEF * zcol(team.off_expl, bl, "offense.explosiveness")
    tempo = base * adj
    return clamp(tempo, 55.0, 80.0)


def weather_ppd_multiplier(wx: Weather) -> float:
    if wx.is_indoor:
        return 1.0
    mult = 1.0
    mult *= max(0.70, 1.0 + WX_WIND_PPD_PER_MPH * max(0.0, wx.wind_mph))
    if wx.precip == "rain":
        mult *= (1.0 + WX_RAIN_PPD)
    elif wx.precip == "snow":
        mult *= (1.0 + WX_SNOW_PPD)
    if wx.temp_f < 40:
        mult *= (1.0 + WX_COLD_PPD)
    elif wx.temp_f > 90:
        mult *= (1.0 + WX_HOT_PPD)
    return mult


def combined_plays(home: Team, away: Team, wx: Weather, bl) -> float:
    th = tempo_proxy(home, bl)
    ta = tempo_proxy(away, bl)
    raw = 0.96 * (th + ta)
    if not wx.is_indoor:
        wind_factor = 1.0 + WX_WIND_TEMPO_PER_MPH * max(0.0, wx.wind_mph)
        raw *= max(0.85, wind_factor)
        if wx.precip in ("rain", "snow"):
            raw *= (1.0 + (WX_RAIN_TEMPO if wx.precip == "rain" else WX_SNOW_TEMPO))
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

    rush_diff_z = 0.0
    pass_diff_z = 0.0
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


def points_per_drive(off_idx: float, def_idx: float) -> float:
    gap = off_idx - def_idx
    return BASE_POINTS_PER_DRIVE * math.exp(EFF_TO_PPD_K * gap)

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


def predict_game(home: Team, away: Team, wx: Weather, bl, vegas_spread=None, vegas_total=None,
                 use_market_blend=True, spread_weight=MARKET_BLEND["spread_weight"],
                 total_weight=MARKET_BLEND["total_weight"], hfa_points=DEFAULT_HFA_POINTS):
    # Indices
    home_off = offense_index(home, bl)
    home_def = defense_index(home, bl)
    away_off = offense_index(away, bl)
    away_def = defense_index(away, bl)

    # Plays & drives
    th = tempo_proxy(home, bl)
    ta = tempo_proxy(away, bl)
    plays = combined_plays(home, away, wx, bl)
    total_drives = max(10.0, plays / PLAYS_PER_DRIVE)
    den = max(1e-6, th + ta)
    drives_home = total_drives * (th / den)
    drives_away = total_drives - drives_home

    # Base PPD
    home_ppd = points_per_drive(home_off, away_def)
    away_ppd = points_per_drive(away_off, home_def)

    # Situational + weather
    wx_mult = weather_ppd_multiplier(wx)
    home_ppd *= situational_ppd_multiplier(home, away, bl) * wx_mult
    away_ppd *= situational_ppd_multiplier(away, home, bl) * wx_mult

    # Raw points
    home_pts = home_ppd * drives_home
    away_pts = away_ppd * drives_away

    # Spread & total (model-only)
    model_total  = home_pts + away_pts
    model_spread = (home_pts - away_pts) + hfa_points

    # Net turnovers
    net_to = expected_net_turnovers(home, away, bl)
    model_spread += MARGIN_PER_NET_TO * net_to

    # Convert to team totals
    home_pts_adj = (model_total + model_spread) / 2.0
    away_pts_adj = model_total - home_pts_adj

    # Optional market blend
    final_spread = model_spread
    final_total  = model_total
    if use_market_blend and (vegas_spread is not None or vegas_total is not None):
        if vegas_spread is not None:
            final_spread = (1 - spread_weight) * model_spread + spread_weight * float(vegas_spread)
        if vegas_total is not None:
            final_total  = (1 - total_weight) * model_total  + total_weight  * float(vegas_total)
        home_pts_adj = (final_total + final_spread) / 2.0
        away_pts_adj = final_total - home_pts_adj

    home_pts_adj = max(0.0, home_pts_adj)
    away_pts_adj = max(0.0, away_pts_adj)

    return {
        "home_team": home.name,
        "away_team": away.name,
        "home_points": round(home_pts_adj, 1),
        "away_points": round(away_pts_adj, 1),
        "spread_home_minus_away": round(final_spread, 1),
        "total_points": round(final_total, 1),
        "model_spread_no_market": round(model_spread, 1),
        "model_total_no_market": round(model_total, 1),
        "vegas_spread": vegas_spread if vegas_spread is not None else "",
        "vegas_total": vegas_total if vegas_total is not None else "",
        "hfa_points": hfa_points,
        "combined_plays_est": round(plays, 1),
        "home_ppd": round(home_ppd, 3),
        "away_ppd": round(away_ppd, 3),
        "drives_home": round(drives_home, 1),
        "drives_away": round(drives_away, 1),
        "net_turnovers_home_adv": round(net_to, 2),
        "home_off_idx": round(home_off, 3),
        "home_def_idx": round(home_def, 3),
        "away_off_idx": round(away_off, 3),
        "away_def_idx": round(away_def, 3),
    }

# ==============================
# Betting helpers
# ==============================

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def american_to_decimal(odds: float) -> float:
    if odds >= 100:
        return 1.0 + odds / 100.0
    if odds <= -100:
        return 1.0 + 100.0 / abs(odds)
    raise ValueError("American odds must be ≤ -100 or ≥ +100")


def break_even_prob(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)


def expected_value_pct(p: float, odds: float) -> float:
    dec = american_to_decimal(odds)
    return p * (dec - 1.0) - (1.0 - p)


def tier_from_ev(ev_pct: float) -> str:
    if ev_pct >= EV_THRESHOLDS["strong"]:
        return "STRONG"
    if ev_pct >= EV_THRESHOLDS["lean"]:
        return "LEAN"
    return "PASS"


def spread_signal(model_spread: float, line_home_minus_away: float, odds: float = DEFAULT_BOOK_ODDS) -> Dict:
    edge = model_spread - float(line_home_minus_away)
    p_home_covers = _phi(edge / SPREAD_SIGMA_PTS)
    be = break_even_prob(odds)
    ev = expected_value_pct(p_home_covers, odds)
    side = "HOME" if p_home_covers > be else "AWAY"
    sig = tier_from_ev(abs(ev)) if ev > 0 else "PASS"
    return {
        "market_line": float(line_home_minus_away),
        "model_spread": round(model_spread, 2),
        "edge_pts": round(edge, 2),
        "p_home_covers": round(p_home_covers, 3),
        "odds": odds,
        "break_even": round(be, 3),
        "ev_pct": round(ev, 3),
        "side": side,
        "signal": sig,
    }


def total_signal(model_total: float, line_total: float, odds: float = DEFAULT_BOOK_ODDS) -> Dict:
    edge = model_total - float(line_total)
    p_over = _phi(edge / TOTALS_SIGMA_PTS)
    be = break_even_prob(odds)
    ev = expected_value_pct(p_over, odds)
    side = "OVER" if p_over > be else "UNDER"
    sig = tier_from_ev(abs(ev)) if ev > 0 else "PASS"
    return {
        "market_total": float(line_total),
        "model_total": round(model_total, 2),
        "edge_pts": round(edge, 2),
        "p_over": round(p_over, 3),
        "odds": odds,
        "break_even": round(be, 3),
        "ev_pct": round(ev, 3),
        "side": side,
        "signal": sig,
    }


def moneyline_signal(model_home_win_prob: float, odds_home: Optional[float] = None, odds_away: Optional[float] = None) -> Dict:
    out: Dict[str, Dict] = {}
    if odds_home is not None:
        be_h = break_even_prob(odds_home)
        ev_h = expected_value_pct(model_home_win_prob, odds_home)
        out["HOME"] = {
            "model_p": round(model_home_win_prob, 3),
            "odds": odds_home,
            "break_even": round(be_h, 3),
            "ev_pct": round(ev_h, 3),
            "signal": tier_from_ev(abs(ev_h)) if ev_h > 0 else "PASS",
        }
    if odds_away is not None:
        p_away = 1.0 - model_home_win_prob
        be_a = break_even_prob(odds_away)
        ev_a = expected_value_pct(p_away, odds_away)
        out["AWAY"] = {
            "model_p": round(p_away, 3),
            "odds": odds_away,
            "break_even": round(be_a, 3),
            "ev_pct": round(ev_a, 3),
            "signal": tier_from_ev(abs(ev_a)) if ev_a > 0 else "PASS",
        }
    return out

# ==============================
# UI
# ==============================

st.set_page_config(page_title="CFB Predictor v4 (Advanced PPA)", page_icon="🏈", layout="wide")
st.title("🏈 College Football Predictor — Advanced (PPA) v4")

with st.sidebar:
    st.header("Data Source")
    up = st.file_uploader("Upload weekly advanced CSV", type=["csv"])
    csv_path = st.text_input("...or path to CSV (optional)")
    df = None
    if up is not None:
        df = load_df(up)
        st.success(f"Loaded CSV with {len(df)} rows")
    elif csv_path.strip():
        df = load_df(csv_path.strip())
        st.success(f"Loaded CSV with {len(df)} rows")

    st.info("Schema detected: **advanced (PPA)** — one row per team")

    st.markdown("---")
    st.header("Game Settings")
    hfa_points_base = st.slider("Home Field Advantage (pts)", 0.0, 10.0, float(DEFAULT_HFA_POINTS), 0.5)
    neutral_site = st.checkbox("Neutral site (zero HFA)", False)

    st.markdown("---")
    st.header("Market Blend")
    use_market = st.checkbox("Blend with Vegas numbers", bool(MARKET_BLEND.get("use", True)))
    vegas_spread = st.text_input("Vegas spread (e.g., 'Home -3.5', 'Away +4', 'pk', or raw +/- like +3.5)", "")
    vegas_total  = st.text_input("Vegas total (e.g., 54.5)", "")
    spread_w = st.slider("Weight toward Vegas spread", 0.0, 1.0, float(MARKET_BLEND.get("spread_weight", 0.25)), 0.05)
    total_w  = st.slider("Weight toward Vegas total",  0.0, 1.0, float(MARKET_BLEND.get("total_weight", 0.25)), 0.05)
    st.caption("Weights are used only if the corresponding Vegas number is provided.")

    st.markdown("---")
    st.header("Moneyline Odds (optional)")
    odds_home = st.text_input("Home ML (e.g., -150 or +120)", "")
    odds_away = st.text_input("Away ML (e.g., +200 or -180)", "")

if df is None:
    st.stop()

# Tabs for single game vs batch mode
single_tab, batch_tab = st.tabs(["Single Game", "Batch Mode"])

with single_tab:
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Home Team")
        home_team_name = st.selectbox("Select home team", sorted(df["team"].astype(str).unique()))
        home_row = df[df["team"].astype(str) == home_team_name].iloc[0]
        home = build_team(home_row)
    with colB:
        st.subheader("Away Team")
        away_team_name = st.selectbox("Select away team", sorted(df["team"].astype(str).unique()))
        away_row = df[df["team"].astype(str) == away_team_name].iloc[0]
        away = build_team(away_row)

    st.markdown("---")
    st.subheader("Weather / Venue")
    wx_cols = st.columns(4)
    with wx_cols[0]:
        is_indoor = st.checkbox("Indoors / Roof", False)
    with wx_cols[1]:
        temp_f = st.number_input("Temp (°F)", 0.0, 120.0, 70.0, 1.0)
    with wx_cols[2]:
        wind_mph = st.number_input("Wind (mph)", 0.0, 50.0, 5.0, 0.5)
    with wx_cols[3]:
        precip = st.selectbox("Precip", ["none", "light", "rain", "snow"]) 
    wx = Weather(is_indoor=is_indoor, temp_f=temp_f, wind_mph=wind_mph, precip=precip)

    # Helpers
    def parse_spread(raw: str, home_name: str, away_name: str) -> Optional[float]:
        raw = raw.strip()
        if not raw:
            return None
        rl = raw.lower()
        if rl in {"pk", "pick", "pickem", "pick'em", "0", "0.0"}:
            return 0.0
        m = re.search(r"([+-]?\d+(?:\.\d+)?)", rl)
        if not m:
            return None
        num = float(m.group(1))

        def has_word(text: str, word: str) -> bool:
            w = word.strip().lower()
            if " " in w:
                return w in text
            return re.search(r"\b" + re.escape(w) + r"\b", text) is not None

        is_home = has_word(rl, "home") or has_word(rl, home_name.lower())
        is_away = has_word(rl, "away") or has_word(rl, away_name.lower())
        if is_home and is_away:
            return None
        if not (is_home or is_away):
            return num  # treat as already Home−Away
        if is_home:
            return num if num > 0 else -abs(num)
        return -num if num > 0 else abs(num)

    def parse_float(s: str) -> Optional[float]:
        s = s.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    run = st.button("Run Prediction")

    if run:
        try:
            BL = compute_baselines(df)

            vs_spread = parse_spread(vegas_spread, home.name, away.name) if use_market else None
            vs_total  = parse_float(vegas_total) if use_market else None
            oh = parse_float(odds_home)
            oa = parse_float(odds_away)
            hfa_points = 0.0 if neutral_site else hfa_points_base

            pred = predict_game(
                home, away, wx, BL,
                vegas_spread=vs_spread, vegas_total=vs_total,
                use_market_blend=use_market, spread_weight=spread_w, total_weight=total_w,
                hfa_points=hfa_points,
            )

            # Monte Carlo
            n_sims = 1000
            home_scores: list[float] = []
            away_scores: list[float] = []
            for _ in range(n_sims):
                hp = np.random.normal(pred["home_ppd"], 0.015) * pred["drives_home"]
                ap = np.random.normal(pred["away_ppd"], 0.015) * pred["drives_away"]
                hp += np.random.normal(0, 0.15) + 0.15 * pred["hfa_points"]
                ap += np.random.normal(0, 0.15)
                home_scores.append(max(0.0, hp))
                away_scores.append(max(0.0, ap))

            home_scores = np.array(home_scores)
            away_scores = np.array(away_scores)
            win_prob = float(np.mean(home_scores > away_scores))
            pred.update(
                {
                    "home_win_prob": round(win_prob, 3),
                    "sim_home_pts": round(float(np.mean(home_scores)), 1),
                    "sim_away_pts": round(float(np.mean(away_scores)), 1),
                    "sim_home_pts_50th": round(float(np.percentile(home_scores, 50)), 1),
                    "sim_away_pts_50th": round(float(np.percentile(away_scores, 50)), 1),
                }
            )

            # Signals
            sig_spread = None
            sig_total = None
            sig_ml = None
            if vs_spread is not None:
                sig_spread = spread_signal(pred["model_spread_no_market"], vs_spread, DEFAULT_BOOK_ODDS)
            if vs_total is not None:
                sig_total = total_signal(pred["model_total_no_market"], vs_total, DEFAULT_BOOK_ODDS)
            if (oh is not None) or (oa is not None):
                sig_ml = moneyline_signal(pred["home_win_prob"], oh, oa)

            # ---- DISPLAY ----
            st.success(
                f"Projected: {pred['away_team']} {pred['away_points']} @ {pred['home_team']} {pred['home_points']}"
            )
            k1, k2, k3 = st.columns(3)
            k1.metric("Home Win Prob", f"{pred['home_win_prob']*100:.1f}%")
            k2.metric("Spread (Home−Away)", f"{pred['spread_home_minus_away']}")
            k3.metric("Total Points", f"{pred['total_points']}")

            with st.expander("Diagnostics"):
                d1, d2, d3 = st.columns(3)
                d1.write(f"Combined plays est: **{pred['combined_plays_est']}**")
                d1.write(f"Drives — Home: **{pred['drives_home']}**, Away: **{pred['drives_away']}**")
                d2.write(f"PPD — Home: **{pred['home_ppd']}**, Away: **{pred['away_ppd']}**")
                d2.write(f"Expected net TO (home+): **{pred['net_turnovers_home_adv']}**")
                d3.write(
                    f"Indices — Home off/def: **{pred['home_off_idx']} / {pred['home_def_idx']}**, "
                    f"Away off/def: **{pred['away_off_idx']} / {pred['away_def_idx']}**"
                )

                # snapshot of advanced mismatches
                def mk_row(name: str, off_v, def_v):
                    return {"Metric": name, "Home Off / Away Def": f"{off_v} / {def_v}"}

                snap = []
                snap.append(mk_row("PPA (off/def)", f"{home.off_ppa:.3f}", f"{away.def_ppa:.3f}"))
                snap.append(mk_row("SuccessRate", f"{home.off_sr}", f"{away.def_sr}"))
                snap.append(mk_row("Explosiveness", f"{home.off_expl}", f"{away.def_expl}"))
                snap.append(mk_row("PowerSuccess", f"{home.off_power}", f"{away.def_power}"))
                snap.append(mk_row("PPO", f"{home.off_ppo}", f"{away.def_ppo}"))
                snap.append(mk_row("Rush PPA", f"{home.off_rush_ppa}", f"{away.def_rush_ppa}"))
                snap.append(mk_row("Pass PPA", f"{home.off_pass_ppa}", f"{away.def_pass_ppa}"))
                st.caption("Home offense vs Away defense (key advanced metrics)")
                st.table(pd.DataFrame(snap))

                import matplotlib.pyplot as plt
                fig1 = plt.figure()
                plt.hist(home_scores - away_scores, bins=40)
                plt.title("Simulated Margin (Home - Away)")
                st.pyplot(fig1)

                fig2 = plt.figure()
                plt.hist(home_scores + away_scores, bins=40)
                plt.title("Simulated Total Points")
                st.pyplot(fig2)

            def vegas_line_text(spread_val: Optional[float], home_label: str, away_label: str) -> str:
                if spread_val is None:
                    return "n/a"
                if spread_val > 0:
                    return f"{home_label} -{abs(spread_val):.1f} (i.e., {away_label} +{abs(spread_val):.1f})"
                if spread_val < 0:
                    return f"{away_label} -{abs(spread_val):.1f} (i.e., {home_label} +{abs(spread_val):.1f})"
                return "Pick'em"

            if sig_spread:
                st.subheader("Spread Signal")
                st.write(f"**Vegas says:** {vegas_line_text(vs_spread, home.name, away.name)}")
                model_line_txt = (
                    f"{home.name} -{abs(sig_spread['model_spread']):.1f}"
                    if sig_spread["model_spread"] > 0
                    else f"{away.name} -{abs(sig_spread['model_spread']):.1f}"
                )
                st.write(f"**Model says:** {model_line_txt}")
                p_cover = (
                    sig_spread["p_home_covers"] if sig_spread["side"] == "HOME" else (1.0 - sig_spread["p_home_covers"])
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Edge (pts) toward", f"{sig_spread['side']} {abs(sig_spread['edge_pts']):.1f}")
                c2.metric("P(Covers)", f"{p_cover * 100:.1f}%")
                c3.metric("EV @ -110", f"{sig_spread['ev_pct'] * 100:.1f}%")
                st.info(f"Recommendation: **{sig_spread['signal']}** → Bet **{sig_spread['side']}**")

            if sig_total:
                st.subheader("Total Signal")
                edge_dir = "OVER" if sig_total["side"] == "OVER" else "UNDER"
                p_hit = sig_total["p_over"] if edge_dir == "OVER" else (1.0 - sig_total["p_over"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Edge (pts) toward", f"{edge_dir} {abs(sig_total['edge_pts']):.1f}")
                c2.metric(f"P({edge_dir})", f"{p_hit * 100:.1f}%")
                c3.metric("EV @ -110", f"{sig_total['ev_pct'] * 100:.1f}%")
                st.info(f"Recommendation: **{sig_total['signal']}** → Bet **{edge_dir}**")

            if sig_ml:
                st.subheader("Moneyline Signals")
                for side, data in sig_ml.items():
                    team_label = pred["home_team"] if side == "HOME" else pred["away_team"]
                    st.write(
                        f"**{team_label}** — Model p: **{data['model_p'] * 100:.1f}%**, "
                        f"Odds: **{data['odds']}**, Break-even: **{data['break_even'] * 100:.1f}%**, "
                        f"EV: **{data['ev_pct'] * 100:.1f}%**, Reco: **{data['signal']}**"
                    )

            # Download row
            out = pred.copy()
            if sig_spread:
                out.update(
                    {
                        "sig_spread_line": sig_spread["market_line"],
                        "sig_spread_model": sig_spread["model_spread"],
                        "sig_spread_edge_pts": sig_spread["edge_pts"],
                        "sig_spread_p_home_covers": sig_spread["p_home_covers"],
                        "sig_spread_ev_pct": sig_spread["ev_pct"],
                        "sig_spread_side": sig_spread["side"],
                        "sig_spread_reco": sig_spread["signal"],
                    }
                )
            if sig_total:
                out.update(
                    {
                        "sig_total_line": sig_total["market_total"],
                        "sig_total_model": sig_total["model_total"],
                        "sig_total_edge_pts": sig_total["edge_pts"],
                        "sig_total_p_over": sig_total["p_over"],
                        "sig_total_ev_pct": sig_total["ev_pct"],
                        "sig_total_side": sig_total["side"],
                        "sig_total_reco": sig_total["signal"],
                    }
                )
            if sig_ml:
                out.update(
                    {
                        "sig_ml_home_model_p": sig_ml.get("HOME", {}).get("model_p"),
                        "sig_ml_home_break_even": sig_ml.get("HOME", {}).get("break_even"),
                        "sig_ml_home_ev_pct": sig_ml.get("HOME", {}).get("ev_pct"),
                        "sig_ml_home_signal": sig_ml.get("HOME", {}).get("signal"),
                        "sig_ml_away_model_p": sig_ml.get("AWAY", {}).get("model_p"),
                        "sig_ml_away_break_even": sig_ml.get("AWAY", {}).get("break_even"),
                        "sig_ml_away_ev_pct": sig_ml.get("AWAY", {}).get("ev_pct"),
                        "sig_ml_away_signal": sig_ml.get("AWAY", {}).get("signal"),
                    }
                )

            out_df = pd.DataFrame([out])
            st.download_button(
                "Download this prediction as CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name=f"cfb_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error: {e}")

with batch_tab:
    st.subheader("Batch mode: score many games")
    st.caption(
        "Upload a schedule CSV with columns: home_team, away_team, optional: vegas_spread, vegas_total, neutral_site (True/False). "
        "Team names must match the advanced CSV."
    )
    sched_up = st.file_uploader("Upload schedule CSV", type=["csv"], key="sched")

    if sched_up is not None:
        try:
            sched = pd.read_csv(sched_up)
            for req in ["home_team", "away_team"]:
                if req not in sched.columns:
                    st.error(f"Schedule missing required column: {req}")
                    st.stop()

            BL = compute_baselines(df)
            team_map = {str(r.team).strip(): build_team(r) for _, r in df.iterrows()}

            def parse_spread_simple(s) -> Optional[float]:
                if pd.isna(s):
                    return None
                t = str(s).strip().lower()
                if t in {"pk", "pick", "pickem", "pick'em", "0", "0.0"}:
                    return 0.0
                m = re.search(r"([+-]?\d+(?:\.\d+)?)", t)
                if not m:
                    return None
                return float(m.group(1))

            rows = []
            for _, g in sched.iterrows():
                ht = str(g["home_team"]).strip()
                at = str(g["away_team"]).strip()
                if ht not in team_map or at not in team_map:
                    rows.append({"home_team": ht, "away_team": at, "error": "Unknown team name"})
                    continue

                home = team_map[ht]
                away = team_map[at]
                wx = Weather(is_indoor=False, temp_f=70.0, wind_mph=5.0, precip="none")
                use_mkt = bool(MARKET_BLEND.get("use", True))
                vs_spread = parse_spread_simple(g.get("vegas_spread")) if use_mkt else None
                vs_total = None
                if use_mkt and ("vegas_total" in g.index) and not pd.isna(g.get("vegas_total")):
                    try:
                        vs_total = float(g.get("vegas_total"))
                    except Exception:
                        vs_total = None
                ns = bool(g.get("neutral_site", False))
                hfa_pts = 0.0 if ns else hfa_points_base

                pred = predict_game(
                    home, away, wx, BL,
                    vegas_spread=vs_spread, vegas_total=vs_total,
                    use_market_blend=use_mkt, spread_weight=spread_w, total_weight=total_w,
                    hfa_points=hfa_pts,
                )

                rows.append({**pred, "neutral_site": ns})

            out_df = pd.DataFrame(rows)
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Download batch predictions CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name=f"cfb_batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Batch error: {e}")
