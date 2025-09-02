# app.py
# Streamlit GUI for CFB predictor (dropdowns from CSV, manual entry fallback, MC sim, signals, downloads)

from dataclasses import dataclass
from typing import Optional, Dict
import math
import csv
import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime

import streamlit as st

# ========== CONFIG & CONSTANTS ==========

def load_config(path="config.json"):
    """Load constants and weights from JSON config file, if present."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

# Defaults (can be overridden by config.json)
AVG_OFF_EPA = 0.00;     STD_OFF_EPA = 0.14
AVG_DEF_EPA = 0.00;     STD_DEF_EPA = 0.14
AVG_RET_PROD = 0.65;    STD_RET_PROD = 0.11
AVG_COMPOSITE_100 = 50.0; STD_COMPOSITE_100 = 18.0
AVG_QB_100 = 50.0;      STD_QB_100 = 20.0
AVG_OL_RET = 2.8;       STD_OL_RET = 1.2
AVG_TEMPO = 65.8;       STD_TEMPO = 7.0
AVG_3D = 0.398;         STD_3D = 0.06
AVG_RZ_TD = 0.615;      STD_RZ_TD = 0.075
AVG_RZ_FG = 0.23;       STD_RZ_FG = 0.055
AVG_FD = 20.2;          STD_FD = 3.5
AVG_GIVE = 1.45;        STD_GIVE = 0.45
AVG_TAKE = 1.45;        STD_TAKE = 0.45
BASE_POINTS_PER_DRIVE = 2.28
PLAYS_PER_DRIVE = 5.8
EFF_TO_PPD_K = 0.11
SPREAD_SIGMA_PTS = 13.0
TOTALS_SIGMA_PTS = 11.0
EV_THRESHOLDS = {"strong": 0.05, "lean": 0.02}
DEFAULT_BOOK_ODDS = -110
WX_WIND_PPD_PER_MPH = -0.008
WX_WIND_TEMPO_PER_MPH = -0.0015
WX_RAIN_PPD = -0.06
WX_SNOW_PPD = -0.10
WX_RAIN_TEMPO = -0.035
WX_SNOW_TEMPO = -0.05
WX_COLD_PPD = -0.03
WX_HOT_PPD = -0.02
MARGIN_PER_NET_TO = 3.2
DEFAULT_HFA_POINTS = 2.5
MARKET_BLEND = {"use": True, "spread_weight": 0.25, "total_weight": 0.25}
OFF_WEIGHTS = {"eff": 0.45, "ret": 0.15, "transfer": 0.08, "recruit": 0.05, "qb": 0.20, "ol": 0.05, "oc_cont": 0.05}
DEF_WEIGHTS = {"eff": 0.50, "ret": 0.20, "transfer": 0.08, "recruit": 0.05, "dc_cont": 0.10, "front7": 0.10}

# Apply config overrides (AFTER defaults are defined)
_cfg = load_config("config.json")
if _cfg:
    consts = _cfg.get("constants", {})
    for k, v in consts.items():
        globals()[k] = v
    OFF_WEIGHTS = _cfg.get("weights", {}).get("off", OFF_WEIGHTS)
    DEF_WEIGHTS = _cfg.get("weights", {}).get("def", DEF_WEIGHTS)
    MARKET_BLEND = _cfg.get("market_blend", MARKET_BLEND)
    EV_THRESHOLDS = _cfg.get("ev_thresholds", EV_THRESHOLDS)

# ========== DATA CLASSES ==========

@dataclass
class TeamInput:
    name: str
    off_epa: float = 0.0
    def_epa: float = 0.0
    returning_prod_off: float = AVG_RET_PROD
    returning_prod_def: float = AVG_RET_PROD
    transfer_grade_100: float = AVG_COMPOSITE_100
    recruit_grade_100: float = AVG_COMPOSITE_100
    qb_grade_100: float = AVG_QB_100
    ol_returning_starters: int = 3
    oc_continuity: bool = True
    dc_continuity: bool = True
    front7_continuity_100: Optional[float] = None
    tempo_plays_per_game: float = AVG_TEMPO
    third_down_off: float = AVG_3D
    third_down_def: float = AVG_3D
    rz_td_off: float = AVG_RZ_TD
    rz_fg_off: float = AVG_RZ_FG
    rz_td_def: float = AVG_RZ_TD
    rz_fg_def: float = AVG_RZ_FG
    first_downs_pg: float = AVG_FD
    giveaways_pg: float = AVG_GIVE
    takeaways_pg: float = AVG_TAKE
    sos_factor: float = 0.0

@dataclass
class GameWeather:
    is_indoor: bool = False
    temp_f: float = 70.0
    wind_mph: float = 5.0
    precip: str = "none"  # "none" | "light" | "rain" | "snow"

# ========== CORE MATH ==========

def z(v: float, m: float, s: float) -> float:
    s = s if s and s > 1e-9 else 1.0
    return (v - m) / s

def preseason_off_index(t: TeamInput) -> float:
    eff_z = z(t.off_epa + t.sos_factor * STD_OFF_EPA, AVG_OFF_EPA, STD_OFF_EPA)
    ret_z = z(t.returning_prod_off, AVG_RET_PROD, STD_RET_PROD)
    tr_z = z(t.transfer_grade_100, AVG_COMPOSITE_100, STD_COMPOSITE_100)
    rec_z = z(t.recruit_grade_100, AVG_COMPOSITE_100, STD_COMPOSITE_100)
    qb_z = z(t.qb_grade_100, AVG_QB_100, STD_QB_100)
    ol_z = z(float(t.ol_returning_starters), AVG_OL_RET, STD_OL_RET)
    oc = 1.0 if t.oc_continuity else 0.0
    return (OFF_WEIGHTS["eff"]*eff_z + OFF_WEIGHTS["ret"]*ret_z + OFF_WEIGHTS["transfer"]*tr_z +
            OFF_WEIGHTS["recruit"]*rec_z + OFF_WEIGHTS["qb"]*qb_z + OFF_WEIGHTS["ol"]*ol_z +
            OFF_WEIGHTS["oc_cont"]*oc)

def preseason_def_index(t: TeamInput) -> float:
    eff_component = -(t.def_epa - t.sos_factor * STD_DEF_EPA)
    eff_z = z(eff_component, -AVG_DEF_EPA, STD_DEF_EPA)
    ret_z = z(t.returning_prod_def, AVG_RET_PROD, STD_RET_PROD)
    tr_z = z(t.transfer_grade_100, AVG_COMPOSITE_100, STD_COMPOSITE_100)
    rec_z = z(t.recruit_grade_100, AVG_COMPOSITE_100, STD_COMPOSITE_100)
    dc = 1.0 if t.dc_continuity else 0.0
    f7_z = 0.0 if t.front7_continuity_100 is None else z(t.front7_continuity_100, AVG_COMPOSITE_100, STD_COMPOSITE_100)
    return (DEF_WEIGHTS["eff"]*eff_z + DEF_WEIGHTS["ret"]*ret_z + DEF_WEIGHTS["transfer"]*tr_z +
            DEF_WEIGHTS["recruit"]*rec_z + DEF_WEIGHTS["dc_cont"]*dc + DEF_WEIGHTS["front7"]*f7_z)

def combined_plays(base_tempo_home: float, base_tempo_away: float, home_fd: float, away_fd: float, wx: GameWeather) -> float:
    raw = 0.96 * (base_tempo_home + base_tempo_away)
    fd_bump = 1.0 + 0.004 * ((home_fd - AVG_FD) + (away_fd - AVG_FD))
    raw *= max(0.85, min(1.15, fd_bump))
    if not wx.is_indoor:
        wind_factor = 1.0 + WX_WIND_TEMPO_PER_MPH * max(0.0, wx.wind_mph)
        raw *= max(0.85, wind_factor)
        if wx.precip in ("rain", "snow"):
            raw *= (1.0 + (WX_RAIN_TEMPO if wx.precip == "rain" else WX_SNOW_TEMPO))
    return raw

def weather_ppd_multiplier(wx: GameWeather) -> float:
    if wx.is_indoor:
        return 1.0
    mult = 1.0
    wind_pen = WX_WIND_PPD_PER_MPH * max(0.0, wx.wind_mph)
    mult *= max(0.70, 1.0 + wind_pen)
    if wx.precip == "rain":
        mult *= (1.0 + WX_RAIN_PPD)
    elif wx.precip == "snow":
        mult *= (1.0 + WX_SNOW_PPD)
    if wx.temp_f < 40:
        mult *= (1.0 + WX_COLD_PPD)   # fixed
    elif wx.temp_f > 90:
        mult *= (1.0 + WX_HOT_PPD)
    return mult

def situational_ppd_multiplier(off: TeamInput, opp_def: TeamInput) -> float:
    def _z(v, m, s): s = s if s and s > 1e-9 else 1.0; return (v - m) / s
    m_3d = 1.0 + 0.05 * (_z(off.third_down_off, AVG_3D, STD_3D) - _z(opp_def.third_down_def, AVG_3D, STD_3D))
    m_3d = max(0.85, min(1.15, m_3d))
    e_off = 7.0 * max(0, min(1, off.rz_td_off)) + 3.0 * max(0, min(1, off.rz_fg_off))
    e_def_allowed = 7.0 * max(0, min(1, opp_def.rz_td_def)) + 3.0 * max(0, min(1, opp_def.rz_fg_def))
    rz_factor = (e_off / 4.8) * (4.8 / max(1e-6, e_def_allowed)) ** 0.5
    m_rz = max(0.85, min(1.15, rz_factor))
    give_pen = max(0.0, _z(off.giveaways_pg, AVG_GIVE, STD_GIVE))
    take_pen = max(0.0, _z(opp_def.takeaways_pg, AVG_TAKE, STD_TAKE))
    m_to = max(0.90, 1.0 - 0.02 * give_pen - 0.015 * take_pen)
    return m_3d * m_rz * m_to

def expected_net_turnovers(home: TeamInput, away: TeamInput) -> float:
    return (home.takeaways_pg + away.giveaways_pg) - (home.giveaways_pg + away.takeaways_pg)

def model_points_per_drive(off_idx: float, opp_def_idx: float) -> float:
    gap = off_idx - opp_def_idx
    return BASE_POINTS_PER_DRIVE * math.exp(EFF_TO_PPD_K * gap)

def predict_game(
    home: TeamInput,
    away: TeamInput,
    weather: GameWeather,
    use_preseason: bool = True,
    hfa_points: float = DEFAULT_HFA_POINTS,
    vegas_spread: Optional[float] = None,
    vegas_total: Optional[float] = None,
    use_market_blend: bool = MARKET_BLEND["use"],
    spread_weight: float = MARKET_BLEND["spread_weight"],
    total_weight: float = MARKET_BLEND["total_weight"],
) -> Dict[str, float]:

    if use_preseason:
        home_off_idx = preseason_off_index(home)
        home_def_idx = preseason_def_index(home)
        away_off_idx = preseason_off_index(away)
        away_def_idx = preseason_def_index(away)
    else:
        home_off_idx = z(home.off_epa + home.sos_factor * STD_OFF_EPA, AVG_OFF_EPA, STD_OFF_EPA)
        away_off_idx = z(away.off_epa + away.sos_factor * STD_OFF_EPA, AVG_OFF_EPA, STD_OFF_EPA)
        home_def_idx = z(-(home.def_epa - home.sos_factor * STD_DEF_EPA), -AVG_DEF_EPA, STD_DEF_EPA)
        away_def_idx = z(-(away.def_epa - away.sos_factor * STD_DEF_EPA), -AVG_DEF_EPA, STD_DEF_EPA)

    min_tempo = 50.0
    plays = combined_plays(max(min_tempo, home.tempo_plays_per_game), max(min_tempo, away.tempo_plays_per_game),
                           home.first_downs_pg, away.first_downs_pg, weather)
    total_drives = max(10.0, plays / PLAYS_PER_DRIVE)
    den = max(1e-6, home.tempo_plays_per_game + away.tempo_plays_per_game)
    drives_home = total_drives * (home.tempo_plays_per_game / den)
    drives_away = total_drives - drives_home

    home_ppd = model_points_per_drive(home_off_idx, away_def_idx)
    away_ppd = model_points_per_drive(away_off_idx, home_def_idx)

    wx_mult = weather_ppd_multiplier(weather)
    home_mult = situational_ppd_multiplier(home, away) * wx_mult
    away_mult = situational_ppd_multiplier(away, home) * wx_mult
    home_ppd *= home_mult
    away_ppd *= away_mult

    home_pts = home_ppd * drives_home
    away_pts = away_ppd * drives_away

    model_total = home_pts + away_pts
    model_spread = (home_pts - away_pts) + hfa_points

    net_to = expected_net_turnovers(home, away)
    model_spread += MARGIN_PER_NET_TO * net_to

    home_pts_adj = (model_total + model_spread) / 2.0
    away_pts_adj = model_total - home_pts_adj

    final_spread = model_spread
    final_total = model_total
    if use_market_blend and (vegas_spread is not None or vegas_total is not None):
        if vegas_spread is not None:
            final_spread = (1 - spread_weight) * model_spread + spread_weight * vegas_spread
        if vegas_total is not None:
            final_total = (1 - total_weight) * model_total + total_weight * vegas_total
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
        "used_preseason": use_preseason,
        "combined_plays_est": round(plays, 1),
        "home_ppd": round(home_ppd, 3),
        "away_ppd": round(away_ppd, 3),
        "drives_home": round(drives_home, 1),
        "drives_away": round(drives_away, 1),
        "net_turnovers_home_adv": round(net_to, 2),
        "wx_indoor": weather.is_indoor,
        "wx_temp_f": weather.temp_f,
        "wx_wind_mph": weather.wind_mph,
        "wx_precip": weather.precip,
    }

# ========== BETTING MATH HELPERS ==========

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def american_to_decimal(odds: float) -> float:
    if odds >= 100:
        return 1.0 + odds / 100.0
    elif odds <= -100:
        return 1.0 + 100.0 / abs(odds)
    else:
        raise ValueError("American odds should be ≤ -100 or ≥ +100")

def break_even_prob(odds: float) -> float:
    dec = american_to_decimal(odds)
    return 1.0 / dec

def expected_value_pct(p: float, odds: float) -> float:
    dec = american_to_decimal(odds)
    return p * (dec - 1.0) - (1.0 - p)

def tier_from_ev(ev_pct: float) -> str:
    if ev_pct >= EV_THRESHOLDS["strong"]:
        return "STRONG"
    if ev_pct >= EV_THRESHOLDS["lean"]:
        return "LEAN"
    return "PASS"

def spread_signal(model_spread: float, line_home_minus_away: float, odds: float = DEFAULT_BOOK_ODDS):
    """Return probabilities/EV for both sides and pick the better one."""
    edge_pts = model_spread - line_home_minus_away
    zv = edge_pts / SPREAD_SIGMA_PTS
    p_home_covers = _phi(zv)
    p_away_covers = 1.0 - p_home_covers

    be = break_even_prob(odds)
    ev_home = expected_value_pct(p_home_covers, odds)
    ev_away = expected_value_pct(p_away_covers, odds)

    if ev_home >= ev_away:
        side = "HOME"; p_side = p_home_covers; ev = ev_home
    else:
        side = "AWAY"; p_side = p_away_covers; ev = ev_away

    signal = tier_from_ev(ev) if ev > 0 else "PASS"
    return {
        "market_line": line_home_minus_away,
        "model_spread": round(model_spread, 2),
        "edge_pts": round(edge_pts, 2),
        "p_home_covers": round(p_home_covers, 3),
        "p_away_covers": round(p_away_covers, 3),
        "p_side": round(p_side, 3),
        "odds": odds,
        "break_even": round(be, 3),
        "ev_pct": round(ev, 3),
        "side": side,
        "signal": signal
    }

def total_signal(model_total: float, line_total: float, odds: float = DEFAULT_BOOK_ODDS):
    """Correct EV logic: choose OVER/UNDER based on which has positive EV."""
    edge_pts = model_total - line_total
    zv = edge_pts / TOTALS_SIGMA_PTS
    p_over = _phi(zv)
    p_under = 1.0 - p_over

    be = break_even_prob(odds)
    ev_over = expected_value_pct(p_over, odds)
    ev_under = expected_value_pct(p_under, odds)

    if ev_over >= ev_under:
        side = "OVER"; p_side = p_over; ev = ev_over
    else:
        side = "UNDER"; p_side = p_under; ev = ev_under

    signal = tier_from_ev(ev) if ev > 0 else "PASS"
    return {
        "market_total": line_total,
        "model_total": round(model_total, 2),
        "edge_pts": round(edge_pts, 2),
        "p_over": round(p_over, 3),
        "p_under": round(p_under, 3),
        "p_side": round(p_side, 3),
        "odds": odds,
        "break_even": round(be, 3),
        "ev_pct": round(ev, 3),
        "side": side,
        "signal": signal
    }

def moneyline_signal(model_home_win_prob: float, odds_home: float = None, odds_away: float = None):
    out = {}
    if odds_home is not None:
        be_h = break_even_prob(odds_home)
        ev_h = expected_value_pct(model_home_win_prob, odds_home)
        out["HOME"] = {
            "model_p": round(model_home_win_prob, 3),
            "odds": odds_home,
            "break_even": round(be_h, 3),
            "ev_pct": round(ev_h, 3),
            "signal": tier_from_ev(ev_h) if ev_h > 0 else "PASS",
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
            "signal": tier_from_ev(ev_a) if ev_a > 0 else "PASS",
        }
    return out

# ========== CSV HELPERS ==========

EXPECTED_HEADERS = [
    "Team",
    "Off EPA/play",
    "Def EPA/play",
    "Returning Production Offense (0–1)",
    "Returning Production Defense (0–1)",
    "Transfer Portal Impact (0–100)",
    "Recruiting / Roster Quality Grade (0–100)",
    "QB Grade (0–100)",
    "Returning OL Starters",
    "OC Continuity",
    "DC Continuity",
    "Front-7 Continuity Grade (0–100)",
    "Offensive Tempo (plays/game)",
    "3rd Down Conv % (Offense)",
    "3rd Down Conv % (Defense)",
    "Red Zone Offense – TD %",
    "Red Zone Offense – FG %",
    "Red Zone Defense – TD %",
    "Red Zone Defense – FG %",
    "First Downs per Game (Offense)",
    "Giveaways per Game",
    "Takeaways per Game",
    "sos_factor",
]

def normalize_header(h: str) -> str:
    """Normalize common dash/space/percent differences so your CSVs are robust."""
    if h is None:
        return ""
    h2 = h.strip()
    h2 = h2.replace("–", "-").replace("—", "-")
    h2 = re.sub(r"\s+", " ", h2)
    return h2

def load_csv_df(uploaded_file_or_path) -> pd.DataFrame:
    if isinstance(uploaded_file_or_path, str):
        df = pd.read_csv(uploaded_file_or_path)
    else:
        df = pd.read_csv(uploaded_file_or_path)
    df.columns = [normalize_header(c) for c in df.columns]
    return df

def team_from_row(row: pd.Series) -> TeamInput:
    # allow missing / percent strings
    def fnum(key, default):
        val = row.get(key, default)
        if pd.isna(val):
            return default
        if isinstance(val, str) and val.endswith("%"):
            try:
                return float(val.strip("%"))/100.0
            except:
                return default
        try:
            return float(val)
        except:
            return default
    def bval(key, default=True):
        v = str(row.get(key, default)).strip().lower()
        return v in ("true","t","1","y","yes")

    return TeamInput(
        name = str(row.get("Team", "")).strip(),
        off_epa = fnum("Off EPA/play", 0.0),
        def_epa = fnum("Def EPA/play", 0.0),
        returning_prod_off = fnum("Returning Production Offense (0-1)", fnum("Returning Production Offense (0–1)", AVG_RET_PROD)),
        returning_prod_def = fnum("Returning Production Defense (0-1)", fnum("Returning Production Defense (0–1)", AVG_RET_PROD)),
        transfer_grade_100 = fnum("Transfer Portal Impact (0-100)", fnum("Transfer Portal Impact (0–100)", AVG_COMPOSITE_100)),
        recruit_grade_100 = fnum("Recruiting / Roster Quality Grade (0-100)", fnum("Recruiting / Roster Quality Grade (0–100)", AVG_COMPOSITE_100)),
        qb_grade_100 = fnum("QB Grade (0-100)", fnum("QB Grade (0–100)", AVG_QB_100)),
        ol_returning_starters = int(fnum("Returning OL Starters", 3)),
        oc_continuity = bval("OC Continuity", True),
        dc_continuity = bval("DC Continuity", True),
        front7_continuity_100 = fnum("Front-7 Continuity Grade (0-100)", fnum("Front-7 Continuity Grade (0–100)", np.nan)),
        tempo_plays_per_game = fnum("Offensive Tempo (plays/game)", AVG_TEMPO),
        third_down_off = fnum("3rd Down Conv % (Offense)", AVG_3D),
        third_down_def = fnum("3rd Down Conv % (Defense)", AVG_3D),
        rz_td_off = fnum("Red Zone Offense - TD %", fnum("Red Zone Offense – TD %", AVG_RZ_TD)),
        rz_fg_off = fnum("Red Zone Offense - FG %", fnum("Red Zone Offense – FG %", AVG_RZ_FG)),
        rz_td_def = fnum("Red Zone Defense - TD %", fnum("Red Zone Defense – TD %", AVG_RZ_TD)),
        rz_fg_def = fnum("Red Zone Defense - FG %", fnum("Red Zone Defense – FG %", AVG_RZ_FG)),
        first_downs_pg = fnum("First Downs per Game (Offense)", AVG_FD),
        giveaways_pg = fnum("Giveaways per Game", AVG_GIVE),
        takeaways_pg = fnum("Takeaways per Game", AVG_TAKE),
        sos_factor = fnum("sos_factor", 0.0)
    )

# ========== UI HELPERS ==========

def parse_spread(raw: str, home_name: str, away_name: str) -> Optional[float]:
    """Return Home−Away line (positive => home favored)."""
    raw = raw.strip()
    if not raw:
        return None
    rl = raw.lower()
    if rl in {"pk", "pick", "pickem", "pick'em", "0", "0.0"}:
        return 0.0
    m = re.search(r'([+-]?\d+(?:\.\d+)?)', rl)
    if not m:
        return None
    num = float(m.group(1))

    def has_word(text, word):
        w = word.strip().lower()
        text = text.lower()
        if " " in w:
            return w in text
        return re.search(r'\b' + re.escape(w) + r'\b', text) is not None

    is_home = has_word(rl, "home") or has_word(rl, home_name)
    is_away = has_word(rl, "away") or has_word(rl, away_name)
    if is_home and is_away:
        return None
    if not (is_home or is_away):
        return num
    if is_home:
        return (+abs(num) if num < 0 else -abs(num))
    else:  # is_away
        return (-abs(num) if num < 0 else +abs(num))

def parse_float_or_none(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except:
        return None

def fmt_book_spread(home_name: str, away_name: str, line_home_minus_away: float) -> str:
    """
    Human-friendly market line.
    Always show favorite as -X and (underdog as +X) for clarity.
    """
    if abs(line_home_minus_away) < 1e-9:
        return f"PK ({home_name} 0 / {away_name} 0)"
    mag = abs(line_home_minus_away)
    if line_home_minus_away > 0:  # home favored
        return f"{home_name} -{mag:.1f} ({away_name} +{mag:.1f})"
    else:  # away favored
        return f"{away_name} -{mag:.1f} ({home_name} +{mag:.1f})"

def fmt_spread_ticket(home_name: str, away_name: str, line_home_minus_away: float, side: str) -> str:
    """
    Ticket string for the chosen side at the market line.
    If the chosen team is the favorite => -X, else +X.
    """
    mag = abs(line_home_minus_away)
    team = home_name if side == "HOME" else away_name
    home_is_fav = line_home_minus_away > 0
    team_is_fav = (side == "HOME" and home_is_fav) or (side == "AWAY" and not home_is_fav)
    sign = '-' if team_is_fav else '+'
    return f"{team} {sign}{mag:.1f}"


# ========== UI ==========

st.set_page_config(page_title="CFB Predictor", page_icon="🏈", layout="wide")
st.title("🏈 College Football Predictor (GUI)")

with st.sidebar:
    st.header("Data Source")
    up = st.file_uploader("Upload team stats CSV", type=["csv"])
    csv_path = st.text_input("...or path to CSV (optional)")
    df = None
    if up is not None:
        df = load_csv_df(up)
        st.success(f"Loaded CSV with {len(df)} rows")
    elif csv_path.strip():
        try:
            df = load_csv_df(csv_path.strip())
            st.success(f"Loaded CSV with {len(df)} rows")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

    st.markdown("---")
    st.header("Game Settings")
    use_preseason = st.checkbox("Use preseason components (EPA/ret/recruit/etc.)", True)
    hfa_points = st.slider("Home Field Advantage (pts)", 0.0, 10.0, float(DEFAULT_HFA_POINTS), 0.5)

    st.markdown("---")
    st.header("Market Blend")
    use_market = st.checkbox("Blend with Vegas numbers", MARKET_BLEND.get("use", True))
    vegas_spread = st.text_input("Vegas spread input (e.g., 'Home -7.5', 'Away +3', 'pk', or raw +/− like +3.5)", "")
    vegas_total = st.text_input("Vegas total (e.g., 54.5)", "")
    spread_w = st.slider("Weight toward Vegas spread", 0.0, 1.0, float(MARKET_BLEND.get("spread_weight", 0.25)), 0.05)
    total_w = st.slider("Weight toward Vegas total", 0.0, 1.0, float(MARKET_BLEND.get("total_weight", 0.25)), 0.05)
    st.caption("Weights are used only if the corresponding Vegas number is provided.")

    st.markdown("---")
    st.header("Moneyline Odds (optional)")
    odds_home = st.text_input("Home ML (e.g., -150 or +120)", "")
    odds_away = st.text_input("Away ML (e.g., +200 or -180)", "")

# Team selectors / manual entry
colA, colB = st.columns(2)
with colA:
    st.subheader("Home Team")
    if df is not None and "Team" in df.columns:
        home_team_name = st.selectbox("Select home team", sorted(df["Team"].astype(str).unique()))
        home_row = df[df["Team"].astype(str) == home_team_name].iloc[0]
        home = team_from_row(home_row)
    else:
        # Minimal manual entry
        home_team_name = st.text_input("Home team name", "Home")
        home = TeamInput(name=home_team_name)
        home.off_epa = st.number_input("Home OFF EPA/play", -1.0, 1.0, 0.0, 0.01)
        home.def_epa = st.number_input("Home DEF EPA/play (negative is better)", -1.0, 1.0, 0.0, 0.01)
        home.returning_prod_off = st.slider("Home Returning OFF (0–1)", 0.0, 1.0, float(AVG_RET_PROD), 0.01)
        home.returning_prod_def = st.slider("Home Returning DEF (0–1)", 0.0, 1.0, float(AVG_RET_PROD), 0.01)
        home.tempo_plays_per_game = st.slider("Home Tempo (plays/gm)", 50.0, 80.0, float(AVG_TEMPO), 0.5)
        home.first_downs_pg = st.slider("Home First Downs/gm", 10.0, 30.0, float(AVG_FD), 0.5)
        home.giveaways_pg = st.slider("Home Giveaways/gm", 0.0, 5.0, float(AVG_GIVE), 0.1)
        home.takeaways_pg = st.slider("Home Takeaways/gm", 0.0, 5.0, float(AVG_TAKE), 0.1)

with colB:
    st.subheader("Away Team")
    if df is not None and "Team" in df.columns:
        away_team_name = st.selectbox("Select away team", sorted(df["Team"].astype(str).unique()))
        away_row = df[df["Team"].astype(str) == away_team_name].iloc[0]
        away = team_from_row(away_row)
    else:
        away_team_name = st.text_input("Away team name", "Away")
        away = TeamInput(name=away_team_name)
        away.off_epa = st.number_input("Away OFF EPA/play", -1.0, 1.0, 0.0, 0.01)
        away.def_epa = st.number_input("Away DEF EPA/play (negative is better)", -1.0, 1.0, 0.0, 0.01)
        away.returning_prod_off = st.slider("Away Returning OFF (0–1)", 0.0, 1.0, float(AVG_RET_PROD), 0.01)
        away.returning_prod_def = st.slider("Away Returning DEF (0–1)", 0.0, 1.0, float(AVG_RET_PROD), 0.01)
        away.tempo_plays_per_game = st.slider("Away Tempo (plays/gm)", 50.0, 80.0, float(AVG_TEMPO), 0.5)
        away.first_downs_pg = st.slider("Away First Downs/gm", 10.0, 30.0, float(AVG_FD), 0.5)
        away.giveaways_pg = st.slider("Away Giveaways/gm", 0.0, 5.0, float(AVG_GIVE), 0.1)
        away.takeaways_pg = st.slider("Away Takeaways/gm", 0.0, 5.0, float(AVG_TAKE), 0.1)

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
wx = GameWeather(is_indoor=is_indoor, temp_f=temp_f, wind_mph=wind_mph, precip=precip)

# ========== RUN BUTTON ==========

def break_even_caption(odds=DEFAULT_BOOK_ODDS) -> str:
    return f"Break-even @ {odds:+}: {break_even_prob(odds)*100:.1f}%"

run = st.button("Run Prediction")
if run:
    try:
        vs_spread = parse_spread(vegas_spread, home.name, away.name) if use_market else None
        vs_total = parse_float_or_none(vegas_total) if use_market else None
        oh = parse_float_or_none(odds_home)
        oa = parse_float_or_none(odds_away)

        pred = predict_game(
            home, away, wx,
            use_preseason=use_preseason,
            hfa_points=hfa_points,
            vegas_spread=vs_spread, vegas_total=vs_total,
            use_market_blend=use_market, spread_weight=spread_w, total_weight=total_w
        )

        # Monte Carlo
        n_sims = 1000
        home_scores, away_scores = [], []
        for _ in range(n_sims):
            home_ppd_var = np.random.normal(pred['home_ppd'], STD_OFF_EPA * EFF_TO_PPD_K)
            away_ppd_var = np.random.normal(pred['away_ppd'], STD_OFF_EPA * EFF_TO_PPD_K)
            net_to_var = np.random.normal(pred['net_turnovers_home_adv'], STD_GIVE)
            hp = max(0, home_ppd_var * pred['drives_home'] + MARGIN_PER_NET_TO * net_to_var + pred['hfa_points'])
            ap = max(0, away_ppd_var * pred['drives_away'])
            home_scores.append(hp); away_scores.append(ap)
        home_scores = np.array(home_scores); away_scores = np.array(away_scores)
        home_win_prob = float(np.mean(home_scores > away_scores))
        sim = {
            "home_win_prob": home_win_prob,
            "sim_home_pts": float(np.mean(home_scores)),
            "sim_away_pts": float(np.mean(away_scores)),
            "sim_home_pts_50th": float(np.percentile(home_scores, 50)),
            "sim_away_pts_50th": float(np.percentile(away_scores, 50))
        }
        pred.update({
            "home_win_prob": round(sim["home_win_prob"], 3),
            "sim_home_pts": round(sim["sim_home_pts"], 1),
            "sim_away_pts": round(sim["sim_away_pts"], 1),
            "sim_home_pts_50th": round(sim["sim_home_pts_50th"], 1),
            "sim_away_pts_50th": round(sim["sim_away_pts_50th"], 1)
        })

        # Signals (use MODEL-ONLY vs book)
        sig_spread = sig_total = sig_ml = None
        if vs_spread is not None:
            sig_spread = spread_signal(pred["model_spread_no_market"], vs_spread, DEFAULT_BOOK_ODDS)
        if vs_total is not None:
            sig_total = total_signal(pred["model_total_no_market"], vs_total, DEFAULT_BOOK_ODDS)
        if (oh is not None) or (oa is not None):
            sig_ml = moneyline_signal(sim["home_win_prob"], oh, oa)

        # ---- DISPLAY ----
        st.success(f"Projected: {pred['away_team']} {pred['away_points']} @ {pred['home_team']} {pred['home_points']}")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Home Win Prob", f"{pred['home_win_prob']*100:.1f}%")
        kpi2.metric("Spread (Home−Away)", f"{pred['spread_home_minus_away']}")
        kpi3.metric("Total Points", f"{pred['total_points']}")
        st.caption("Spread sign rule: positive = home favored, negative = away favored.")

        with st.expander("Diagnostics"):
            d1, d2, d3 = st.columns(3)
            d1.write(f"Combined plays est: **{pred['combined_plays_est']}**")
            d1.write(f"Drives — Home: **{pred['drives_home']}**, Away: **{pred['drives_away']}**")
            d2.write(f"PPD — Home: **{pred['home_ppd']}**, Away: **{pred['away_ppd']}**")
            d2.write(f"Expected net TO (home+): **{pred['net_turnovers_home_adv']}**")
            d3.write(f"WX: indoor={pred['wx_indoor']} temp={pred['wx_temp_f']}°F wind={pred['wx_wind_mph']} mph precip={pred['wx_precip']}")

            # Simple charts (histograms)
            import matplotlib.pyplot as plt
            fig1 = plt.figure()
            plt.hist(home_scores - away_scores, bins=40)
            plt.title("Simulated Margin (Home - Away)")
            st.pyplot(fig1)

            fig2 = plt.figure()
            plt.hist(home_scores + away_scores, bins=40)
            plt.title("Simulated Total Points")
            st.pyplot(fig2)

        if sig_spread:
            st.subheader("Spread Signal")
            st.write(f"**Vegas says:** {fmt_book_spread(pred['home_team'], pred['away_team'], vs_spread)}")
            st.write(f"**Model says:** {fmt_book_spread(pred['home_team'], pred['away_team'], sig_spread['model_spread'])}")

            ticket = fmt_spread_ticket(pred['home_team'], pred['away_team'], vs_spread, sig_spread['side'])
            c1, c2, c3 = st.columns(3)
            c1.metric("Edge", f"+{abs(sig_spread['edge_pts']):.1f} pts on {ticket}")
            c2.metric("P(Covers)", f"{sig_spread['p_side']*100:.1f}%")
            c3.metric("EV @ -110", f"{sig_spread['ev_pct']*100:.1f}%")
            st.caption(break_even_caption(DEFAULT_BOOK_ODDS))
            st.info(f"Recommendation: **{sig_spread['signal']}** → Bet **{ticket}**")

        if sig_total:
            st.subheader("Total Signal")
            c1, c2, c3 = st.columns(3)
            c1.metric("Edge", f"+{abs(sig_total['edge_pts']):.1f} pts on {sig_total['side']} {vs_total:.1f}")
            c2.metric(f"P({sig_total['side'].title()} {vs_total:.1f})", f"{sig_total['p_side']*100:.1f}%")
            c3.metric("EV @ -110", f"{sig_total['ev_pct']*100:.1f}%")
            st.caption(break_even_caption(DEFAULT_BOOK_ODDS))
            st.info(f"Recommendation: **{sig_total['signal']}** → Bet **{sig_total['side']} {vs_total:.1f}**")

        if sig_ml:
            st.subheader("Moneyline Signals")
            for side, data in sig_ml.items():
                team = pred["home_team"] if side=="HOME" else pred["away_team"]
                st.write(f"**{team}** — Model p: **{data['model_p']*100:.1f}%**, Odds: **{data['odds']}**, "
                         f"Break-even: **{data['break_even']*100:.1f}%**, EV: **{data['ev_pct']*100:.1f}%**, "
                         f"Reco: **{data['signal']}**")

        # Download row as CSV
        out = pred.copy()
        if sig_spread:
            out.update({
                "sig_spread_line": sig_spread["market_line"],
                "sig_spread_model": sig_spread["model_spread"],
                "sig_spread_edge_pts": sig_spread["edge_pts"],
                "sig_spread_p_home_covers": sig_spread["p_home_covers"],
                "sig_spread_p_away_covers": sig_spread["p_away_covers"],
                "sig_spread_p_selected": sig_spread["p_side"],
                "sig_spread_ev_pct": sig_spread["ev_pct"],
                "sig_spread_side": sig_spread["side"],
                "sig_spread_reco": sig_spread["signal"],
            })
        if sig_total:
            out.update({
                "sig_total_line": sig_total["market_total"],
                "sig_total_model": sig_total["model_total"],
                "sig_total_edge_pts": sig_total["edge_pts"],
                "sig_total_p_over": sig_total["p_over"],
                "sig_total_p_under": sig_total["p_under"],
                "sig_total_p_selected": sig_total["p_side"],
                "sig_total_ev_pct": sig_total["ev_pct"],
                "sig_total_side": sig_total["side"],
                "sig_total_reco": sig_total["signal"],
            })
        if sig_ml:
            out.update({
                "sig_ml_home_model_p": sig_ml.get("HOME", {}).get("model_p"),
                "sig_ml_home_break_even": sig_ml.get("HOME", {}).get("break_even"),
                "sig_ml_home_ev_pct": sig_ml.get("HOME", {}).get("ev_pct"),
                "sig_ml_home_signal": sig_ml.get("HOME", {}).get("signal"),
                "sig_ml_away_model_p": sig_ml.get("AWAY", {}).get("model_p"),
                "sig_ml_away_break_even": sig_ml.get("AWAY", {}).get("break_even"),
                "sig_ml_away_ev_pct": sig_ml.get("AWAY", {}).get("ev_pct"),
                "sig_ml_away_signal": sig_ml.get("AWAY", {}).get("signal"),
            })

        out_df = pd.DataFrame([out])
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download this prediction as CSV", data=csv_bytes,
                           file_name=f"cfb_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
