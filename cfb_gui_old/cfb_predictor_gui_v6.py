# app.py — CFB PPA Monte Carlo Predictor (safe NaN handling, main-screen selectors, nicer UI)
# pip install streamlit pandas numpy

import os
import math
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------- Tunables ----------------------
BASE_TEAM_POINTS = 27.0
RATING_SCALE_TO_POINTS = 7.5
MIN_SD_POINTS = 10.0
MAX_SD_POINTS = 24.0

OFF_WEIGHTS = {
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

DEF_WEIGHTS = {
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

REQUIRED_COLUMNS = [
    "team",
    "offense.ppa",
    "offense.successRate",
    "offense.explosiveness",
    "offense.powerSuccess",
    "offense.pointsPerOpportunity",
    "offense.fieldPosition.averageStart",
    "offense.standardDowns.successRate",
    "offense.passingDowns.ppa",
    "offense.rushingPlays.ppa",
    "offense.passingPlays.ppa",
    "defense.ppa",
    "defense.successRate",
    "defense.explosiveness",
    "defense.powerSuccess",
    "defense.stuffRate",
    "defense.pointsPerOpportunity",
    "defense.fieldPosition.averageStart",
    "defense.havoc.total",
    "defense.rushingPlays.ppa",
    "defense.passingPlays.ppa",
]

# ---------------------- Helpers ----------------------
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
    if uploaded is not None:
        return pd.read_csv(uploaded)
    for path in ("advancedStats.csv", "/mnt/data/advancedStats.csv"):
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()

def coerce_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Force numeric on all model features; non-numeric -> NaN
    feature_cols = [c for c in REQUIRED_COLUMNS if c != "team"]
    for c in feature_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Impute with column medians (robust & simple)
    medians = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(medians)
    return df

def zscore_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            mean = out[c].mean()
            std = out[c].std(ddof=0)
            if std and std > 0:
                out[c + "__z"] = (out[c] - mean) / std
            else:
                out[c + "__z"] = 0.0
            # guard any lingering NaNs
            out[c + "__z"] = out[c + "__z"].fillna(0.0)
    return out

def team_row(df: pd.DataFrame, team: str) -> pd.Series:
    row = df.loc[df["team"] == team]
    if row.empty:
        raise ValueError(f"Team '{team}' not found")
    return row.iloc[0]

def composite_rating(row: pd.Series, weights: dict) -> float:
    s = 0.0
    for col, w in weights.items():
        z = row.get(col + "__z", 0.0)
        if not np.isfinite(z): z = 0.0
        s += w * float(z)
    return s

def weather_multiplier(temp_f: float, wind_mph: float, precip: str, indoor: bool) -> float:
    if indoor:
        return 1.0
    mult = 1.0
    if temp_f <= 25: mult *= 0.90
    elif temp_f <= 35: mult *= 0.95
    elif temp_f >= 95: mult *= 0.97
    elif temp_f >= 90: mult *= 0.985
    if wind_mph > 5:
        mult *= (1.0 - min(0.15, 0.01 * (wind_mph - 5)))
    mult *= {"None": 1.00, "Light": 0.97, "Moderate": 0.93, "Heavy": 0.88}.get(precip, 1.0)
    return mult

def volatility_sd(off_row: pd.Series, def_row: pd.Series) -> float:
    off_expl = off_row.get("offense.explosiveness__z", 0.0) or 0.0
    def_havoc = def_row.get("defense.havoc.total__z", 0.0) or 0.0
    sd = MIN_SD_POINTS + 4.0 * max(0.0, off_expl) - 2.0 * max(0.0, def_havoc)
    return float(np.clip(sd, MIN_SD_POINTS, MAX_SD_POINTS))

def safe_mu_sigma(mu, sd):
    if not np.isfinite(mu): mu = BASE_TEAM_POINTS
    if not (np.isfinite(sd) and sd > 0): sd = 14.0
    return float(mu), float(sd)

def simulate_scores(mu_home, mu_away, sd_home, sd_away, n_sims, seed):
    rng = np.random.default_rng(int(seed))
    mu_home, sd_home = safe_mu_sigma(mu_home, sd_home)
    mu_away, sd_away = safe_mu_sigma(mu_away, sd_away)
    home = rng.normal(mu_home, sd_home, size=n_sims)
    away = rng.normal(mu_away, sd_away, size=n_sims)
    home = np.clip(home, 0, None)
    away = np.clip(away, 0, None)
    # If any NaNs slipped through (shouldn't), replace with 0 before rounding
    home = np.nan_to_num(home, nan=0.0)
    away = np.nan_to_num(away, nan=0.0)
    return np.rint(home).astype(int), np.rint(away).astype(int)

def cover_probs_and_ev(home_scores, away_scores, market_spread_home, market_total, spread_odds, total_odds):
    margin = home_scores - away_scores
    total = home_scores + away_scores

    k = market_spread_home
    k_is_int = abs(k - round(k)) < 1e-9
    if k_is_int:
        p_home_push = float(np.mean(margin == int(round(k))))
        p_home_cover = float(np.mean(margin > k))
        p_home_lose  = max(0.0, 1.0 - p_home_cover - p_home_push)
    else:
        p_home_push = 0.0
        p_home_cover = float(np.mean(margin > k))
        p_home_lose = 1.0 - p_home_cover

    t = market_total
    t_is_int = abs(t - round(t)) < 1e-9
    if t_is_int:
        p_over_push = float(np.mean(total == int(round(t))))
        p_over = float(np.mean(total > t))
        p_under = max(0.0, 1.0 - p_over - p_over_push)
    else:
        p_over_push = 0.0
        p_over = float(np.mean(total > t))
        p_under = 1.0 - p_over

    return {
        "p_home_cover": p_home_cover,
        "p_home_push": p_home_push,
        "p_away_cover": p_home_lose,
        "ev_home_spread": ev_from_p(p_home_cover, spread_odds, p_home_push),
        "ev_away_spread": ev_from_p(p_home_lose,  spread_odds, p_home_push),
        "p_over": p_over,
        "p_over_push": p_over_push,
        "p_under": p_under,
        "ev_over": ev_from_p(p_over,  total_odds, p_over_push),
        "ev_under": ev_from_p(p_under, total_odds, p_over_push),
        "home_win": float(np.mean(margin > 0.0) + 0.5 * np.mean(margin == 0.0)),
        "away_win": float(np.mean(margin < 0.0) + 0.5 * np.mean(margin == 0.0)),
    }

def choose_recommendation(metrics, market_spread_home, market_total):
    candidates = [
        ("Home " + (f"{market_spread_home:+g}" if market_spread_home != 0 else "PK"),
         metrics["ev_home_spread"], metrics["p_home_cover"]),
        ("Away " + (f"{-market_spread_home:+g}" if market_spread_home != 0 else "PK"),
         metrics["ev_away_spread"], metrics["p_away_cover"]),
        (f"Over {market_total:g}",  metrics["ev_over"],  metrics["p_over"]),
        (f"Under {market_total:g}", metrics["ev_under"], metrics["p_under"]),
    ]
    label, ev, p = max(candidates, key=lambda x: x[1])
    return {"label": label, "ev": ev, "p": p, "confidence": bucket_confidence(ev, p)}

# ---------------------- UI ----------------------
st.set_page_config(page_title="CFB Predictor — PPA Monte Carlo", layout="wide")
st.title("🏈 CFB Predictor — PPA Monte Carlo, Market Blend & EV")

# Load data (main screen)
data_col, meta_col = st.columns([3, 2], gap="large")
with data_col:
    uploaded = st.file_uploader("Upload advanced PPA CSV (or keep advancedStats.csv next to this file)", type=["csv"])
    df = load_csv(uploaded)
    if df.empty:
        st.info("Please upload a CSV. Expected columns include offense.ppa, defense.ppa, etc.")
        st.stop()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"CSV missing required columns: {missing}")
        st.stop()
    df = coerce_and_impute(df)
    st.success(f"Loaded {len(df['team'].unique())} teams.")

# Precompute z-scores on all weighted features
zcols = list(set(list(OFF_WEIGHTS.keys()) + list(DEF_WEIGHTS.keys())))
df_z = zscore_columns(df, zcols)

# Main selectors (requested: NOT in sidebar)
match_tab, market_tab, results_tab = st.tabs(["Matchup & Weather", "Market & Simulation", "Results"])

with match_tab:
    left, right = st.columns([1, 1], gap="large")

    with left:
        team_list = sorted(df_z["team"].unique().tolist())
        home = st.selectbox("Home team", team_list, index=0 if team_list else None, key="home_team")
        away = st.selectbox("Away team", team_list, index=1 if len(team_list) > 1 else None, key="away_team")
        if home and away and home == away:
            st.warning("Pick two different teams.")
        if st.button("Swap Home ↔ Away"):
            # simple swap through session_state
            st.session_state["home_team"], st.session_state["away_team"] = (
                st.session_state["away_team"], st.session_state["home_team"]
            )
            st.experimental_rerun()

    with right:
        st.subheader("Weather & Context")
        neutral = st.checkbox("Neutral Site (zeros HFA)", value=False)
        indoor = st.checkbox("Indoors / Roof Closed", value=False)
        temp_f = st.slider("Temperature (°F)", -10, 110, 65, 1, disabled=indoor)
        wind_mph = st.slider("Wind (mph)", 0, 40, 5, 1, disabled=indoor)
        precip = st.select_slider("Precipitation", ["None", "Light", "Moderate", "Heavy"], value="None", disabled=indoor)
        default_hfa = 0.0 if neutral else 2.5
        hfa = 0.0 if neutral else st.slider("Home Field Advantage (pts)", 0.0, 6.0, default_hfa, 0.5)
        if st.button("Reset Weather/HFA"):
            for k, v in [("neutral", False), ("indoor", False)]:
                st.session_state[k] = v
            st.experimental_rerun()

with market_tab:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Market Inputs")
        st.caption("Spread is home-based (negative = home favored).")
        market_spread_home = st.number_input("Market Spread (Home perspective)", value=-3.0, step=0.5, format="%.1f")
        market_total = st.number_input("Market Total", value=52.5, step=0.5, format="%.1f")
        spread_odds = st.number_input("Spread Price (American)", value=-110, step=5)
        total_odds = st.number_input("Total Price (American)", value=-110, step=5)
        market_weight = st.slider("Market Blend Weight → Blended Lines", 0.0, 1.0, 0.35, 0.05)
    with right:
        st.subheader("Simulation")
        n_sims = st.slider("Number of Simulations", 1000, 50000, 10000, 1000)
        seed = st.number_input("Random Seed", value=42, step=1)
        st.caption("Scores simulated from truncated Normals; you can tune constants at the top of the file.")

# Stop early if bad selection
if home == away:
    st.stop()

# ---------------------- Model compute ----------------------
home_row = team_row(df_z, home)
away_row = team_row(df_z, away)

home_off = composite_rating(home_row, OFF_WEIGHTS)
home_def = composite_rating(home_row, DEF_WEIGHTS)
away_off = composite_rating(away_row, OFF_WEIGHTS)
away_def = composite_rating(away_row, DEF_WEIGHTS)

mu_home_raw = BASE_TEAM_POINTS + RATING_SCALE_TO_POINTS * (home_off - away_def)
mu_away_raw = BASE_TEAM_POINTS + RATING_SCALE_TO_POINTS * (away_off - home_def)

# HFA split to keep total stable
hfa_pts = 0.0 if neutral else hfa
mu_home_raw += hfa_pts / 2.0
mu_away_raw -= hfa_pts / 2.0

w_mult = weather_multiplier(temp_f, wind_mph, precip, indoor)
mu_home = mu_home_raw * w_mult
mu_away = mu_away_raw * w_mult

sd_home = volatility_sd(home_row, away_row)
sd_away = volatility_sd(away_row, home_row)
if not indoor:
    sd_tighten = 1.0 - (1.0 - w_mult) * 0.5
    sd_home *= sd_tighten
    sd_away *= sd_tighten

home_scores, away_scores = simulate_scores(mu_home, mu_away, sd_home, sd_away, n_sims, seed)
margins = home_scores - away_scores
totals = home_scores + away_scores

model_spread = float(np.mean(margins))
model_total = float(np.mean(totals))
blend_spread = (1 - market_weight) * model_spread + market_weight * market_spread_home
blend_total  = (1 - market_weight) * model_total  + market_weight * market_total

metrics = cover_probs_and_ev(
    home_scores, away_scores,
    market_spread_home=market_spread_home,
    market_total=market_total,
    spread_odds=spread_odds,
    total_odds=total_odds,
)
recommendation = choose_recommendation(metrics, market_spread_home, market_total)

# ---------------------- Results UI ----------------------
with results_tab:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.subheader("Projected Score (Raw Model)")
        st.metric(home, f"{np.mean(home_scores):.1f}")
        st.metric(away, f"{np.mean(away_scores):.1f}")
        st.caption(f"Median: {home} {np.median(home_scores):.0f} — {away} {np.median(away_scores):.0f}")

    with c2:
        st.subheader("Model Lines")
        st.metric("Spread (Home − Away)", f"{model_spread:+.2f}")
        st.metric("Total", f"{model_total:.2f}")
        st.caption("Negative spread ⇒ Home favored.")

    with c3:
        st.subheader("Market-Blended Lines")
        st.metric("Blended Spread (H−A)", f"{blend_spread:+.2f}")
        st.metric("Blended Total", f"{blend_total:.2f}")
        st.caption(f"Blend: {100*(1-market_weight):.0f}% model / {100*market_weight:.0f}% market")

    st.divider()
    c4, c5, c6, c7, c8 = st.columns(5)
    with c4:
        st.metric(f"Home {market_spread_home:+g}", f"{metrics['p_home_cover']*100:.1f}%")
        st.caption(f"EV @ {spread_odds}: {metrics['ev_home_spread']:.3f}/1u")
    with c5:
        st.metric(f"Away {-market_spread_home:+g}", f"{metrics['p_away_cover']*100:.1f}%")
        st.caption(f"EV @ {spread_odds}: {metrics['ev_away_spread']:.3f}/1u")
    with c6:
        st.metric(f"Over {market_total:g}", f"{metrics['p_over']*100:.1f}%")
        st.caption(f"EV @ {total_odds}: {metrics['ev_over']:.3f}/1u")
    with c7:
        st.metric(f"Under {market_total:g}", f"{metrics['p_under']*100:.1f}%")
        st.caption(f"EV @ {total_odds}: {metrics['ev_under']:.3f}/1u")
    with c8:
        st.metric("Win Prob (Model)", f"{metrics['home_win']*100:.1f}% {home}")
        st.caption(f"{away}: {metrics['away_win']*100:.1f}%")

    st.subheader("Recommendation")
    st.write(f"**{recommendation['confidence']}** — {recommendation['label']} "
             f"(EV {recommendation['ev']:.3f}/1u, P {recommendation['p']*100:.1f}%).")

    with st.expander("Quick distribution stats"):
        left, mid, right = st.columns(3)
        with left:
            st.write(f"**{home} points**  Mean {np.mean(home_scores):.2f} | Std {np.std(home_scores):.2f}")
            q = np.percentile(home_scores, [25, 50, 75]).astype(int)
            st.write(f"25/50/75%: {q.tolist()}")
        with mid:
            st.write("**Margin (H−A)**  Mean {0:.2f} | Std {1:.2f}".format(np.mean(margins), np.std(margins)))
        with right:
            st.write("**Total**  Mean {0:.2f} | Std {1:.2f}".format(np.mean(totals), np.std(totals)))
            q = np.percentile(totals, [25, 50, 75]).astype(int)
            st.write(f"25/50/75%: {q.tolist()}")

    # Downloadable summary
    summary = pd.DataFrame([{
        "home": home, "away": away,
        "market_spread_home": market_spread_home,
        "market_total": market_total,
        "model_spread": model_spread,
        "model_total": model_total,
        "blended_spread": blend_spread,
        "blended_total": blend_total,
        "p_home_cover": metrics["p_home_cover"],
        "p_away_cover": metrics["p_away_cover"],
        "p_over": metrics["p_over"],
        "p_under": metrics["p_under"],
        "ev_home_spread": metrics["ev_home_spread"],
        "ev_away_spread": metrics["ev_away_spread"],
        "ev_over": metrics["ev_over"],
        "ev_under": metrics["ev_under"],
        "home_win_prob": metrics["home_win"],
        "away_win_prob": metrics["away_win"],
        "recommendation": recommendation["label"],
        "recommendation_confidence": recommendation["confidence"],
        "recommendation_p": recommendation["p"],
        "recommendation_ev": recommendation["ev"],
        "mu_home_adj": float(mu_home),
        "mu_away_adj": float(mu_away),
        "sd_home": float(sd_home),
        "sd_away": float(sd_away),
        "weather_multiplier": float(w_mult),
        "hfa_points": float(hfa_pts),
        "market_weight": float(market_weight),
        "n_sims": int(n_sims),
        "seed": int(seed),
    }])
    st.download_button(
        "⬇️ Download summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"cfb_predictor_summary_{home}_vs_{away}.csv",
        mime="text/csv",
    )

st.caption("Model: z-scored advanced PPA features → composite ratings; expected points = baseline + scale × (Off_z − OppDef_z), adjusted by weather/HFA; scores ~ truncated Normal. Missing data are median-imputed; NaNs are guarded so no more −2,147,483,648 surprises 🙂")
