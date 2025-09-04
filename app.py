# app.py — 🥜 The Goober Model (CFB)
# - Bet Board available without CSV
# - Supabase persistence for bets + saved projections
# - Discord notify via direct Webhook (fast) with optional Edge Function fallback
# - Keeps tab position when saving
# - Uses stored team names on bets (works even without CSV)

import math
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import json, requests

try:
    import altair as alt
except Exception:
    alt = None

# ---------- Page / Styles ----------
st.set_page_config(page_title="🥜 The Goober Model", layout="wide")
st.markdown(
    """
<style>
.page-sep { height: 2px; background: linear-gradient(90deg, #0ea5e933, #22c55e55, #f59e0b33);
            border-radius: 999px; margin: 18px 0 14px 0; }
.score-card { padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.08); margin-bottom:14px; }
.score-green { background: rgba(16,185,129,0.16); }
.score-yellow{ background: rgba(234,179,8,0.16); }
.section-caption { font-size:0.85rem; opacity:0.75; }
.small { font-size:.9rem; opacity:.9; }
.value-chip { padding:12px 14px; border-radius:12px; border:1px solid rgba(255,255,255,0.08);
              background: rgba(255,255,255,0.03); margin-bottom:14px; }
.value-chip .label { font-size:0.85rem; opacity:0.75; margin-bottom:4px; }
.value-chip .value { font-size:1.25rem; font-weight:700; }
.rec-card { background: rgba(16,185,129,0.16); border:1px solid rgba(16,185,129,0.35);
            padding:16px 18px; border-radius:12px; font-weight:800; font-size:1.1rem; }
.stat-grid { display:grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap:12px; }
.stat-card { padding:12px 14px; border-radius:12px; border:1px solid rgba(255,255,255,0.08);
             background: rgba(255,255,255,0.03); }
.stat-card .title { font-size:0.85rem; opacity:0.8; margin-bottom:4px; }
.stat-card .num { font-weight:700; font-size:1.15rem; }
.muted { opacity:.85; }
.status-badge { display:inline-block; padding:2px 8px; border-radius:999px;
                font-size:0.75rem; font-weight:700; color:#fff; margin-left:8px; }
.status-pending { background:#64748b; }
.status-win { background:#16a34a; }
.status-loss { background:#dc2626; }
.status-push { background:#f59e0b; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Supabase ----------
from supabase import create_client, Client

@st.cache_resource
def get_sb() -> Client | None:
    cfg = st.secrets.get("supabase", {})
    url = cfg.get("url") or os.environ.get("SB_URL")
    key = (
        cfg.get("service_key")
        or cfg.get("anon_key")
        or os.environ.get("SB_SERVICE_KEY")
        or os.environ.get("SB_SERVICE_ROLE_KEY")
    )
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

sb: Client | None = get_sb()
USE_DB = sb is not None

# ---------- Utils ----------
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def american_to_decimal(american: float) -> float:
    a = float(american)
    return 1.0 + (100.0 / abs(a) if a < 0 else a / 100.0)

def ev_per_unit(p_win: float, american: float) -> float:
    dec = american_to_decimal(american)
    return p_win * (dec - 1.0) - (1.0 - p_win) * 1.0

def coerce_float(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan

def center(val, mean):
    if np.isnan(val) or np.isnan(mean):
        return 0.0
    return float(val) - float(mean)

def get_numeric(row: pd.Series, col: str) -> float:
    if col not in row:
        return np.nan
    return coerce_float(row[col])

def clamp_nonneg(x: float) -> float:
    return max(0.0, float(x))

def _round_line(val: float, eps: float = 0.05) -> float:
    if val is None or np.isnan(val):
        return 0.0
    r = round(float(val), 1)
    return 0.0 if abs(r) < eps else r

def format_home_away_spreads(home_team: str, away_team: str, home_line: float) -> tuple[str, str]:
    L = _round_line(home_line)
    n = abs(L)
    if L < 0:
        return f"{home_team} -{n:.1f}", f"{away_team} +{n:.1f}"
    elif L > 0:
        return f"{home_team} +{n:.1f}", f"{away_team} -{n:.1f}"
    else:
        return f"{home_team} PK", f"{away_team} PK"

def model_spread_to_home_line(spread_model: float) -> float:
    return -spread_model

def home_cover_probability(spread_model: float, home_market_line: float, sigma_margin: float) -> float:
    z = ((-home_market_line) - spread_model) / max(1e-9, sigma_margin)
    return 1.0 - normal_cdf(z)

@st.cache_data
def league_means(df: pd.DataFrame):
    off_mean = pd.to_numeric(df.get("OFF_EFF", pd.Series([])), errors="coerce").mean(skipna=True)
    def_mean = pd.to_numeric(df.get("DEF_EFF", pd.Series([])), errors="coerce").mean(skipna=True)
    sp_mean  = pd.to_numeric(df.get("SP_EFF",  pd.Series([])), errors="coerce").mean(skipna=True)
    ovrl_mean= pd.to_numeric(df.get("OVRL_EFF",pd.Series([])), errors="coerce").mean(skipna=True)
    fpi_mean = pd.to_numeric(df.get("FPI",      pd.Series([])), errors="coerce").mean(skipna=True)
    fpi_min  = pd.to_numeric(df.get("FPI", pd.Series([])), errors="coerce").min(skipna=True)
    fpi_max  = pd.to_numeric(df.get("FPI", pd.Series([])), errors="coerce").max(skipna=True)
    return off_mean, def_mean, sp_mean, ovrl_mean, fpi_mean, fpi_min, fpi_max

def get_team_row(df: pd.DataFrame, team: str) -> pd.Series:
    row = df.loc[df["Team"] == team]
    return row.iloc[0] if not row.empty else pd.Series(dtype=object)

def centered_from_rank(rank: float, n_teams: int) -> float:
    if np.isnan(rank) or rank <= 0 or n_teams <= 0:
        return 0.0
    return ((n_teams + 1 - rank) / n_teams) - 0.5

def status_badge_html(status: str) -> str:
    s = (status or "pending").lower()
    cls = {"pending":"status-pending","win":"status-win","loss":"status-loss","push":"status-push"}.get(s,"status-pending")
    return f"<span class='status-badge {cls}'>{s.capitalize()}</span>"

# ---------- DB helpers ----------
def db_list_bets():
    if USE_DB:
        try:
            res = sb.table("bets").select("*").order("created_at", desc=True).execute()
            return res.data or []
        except Exception as e:
            st.warning(f"DB list bets failed: {e}")
            return []
    return st.session_state.get("bets", [])

def db_add_bet(rec: dict):
    if USE_DB:
        try:
            ins = sb.table("bets").insert(rec).execute()
            data = getattr(ins, "data", None)
            if isinstance(data, list) and data:
                return data[0]
            if isinstance(data, dict):
                return data
            return rec
        except Exception as e:
            st.warning(f"DB insert bet failed: {e}")
            return rec
    out = rec.copy()
    out["id"] = out.get("id") or str(uuid.uuid4())
    st.session_state.setdefault("bets", []).insert(0, out)
    return out

def db_update_bet_status(bet_id: str, status: str):
    if USE_DB:
        try:
            sb.table("bets").update({"status": status}).eq("id", bet_id).execute()
            return
        except Exception as e:
            st.warning(f"DB update failed: {e}")
            return
    for x in st.session_state.get("bets", []):
        if x.get("id") == bet_id:
            x["status"] = status

def db_delete_bet(bet_id: str):
    if USE_DB:
        try:
            sb.table("bets").delete().eq("id", bet_id).execute()
            return
        except Exception as e:
            st.warning(f"DB delete failed: {e}")
            return
    arr = st.session_state.get("bets", [])
    st.session_state["bets"] = [x for x in arr if x.get("id") != bet_id]

def db_list_projections():
    if USE_DB:
        try:
            res = sb.table("projections").select("*").order("created_at", desc=True).execute()
            return res.data or []
        except Exception as e:
            st.warning(f"DB list projections failed: {e}")
            return []
    return st.session_state.get("projections", [])

def db_add_projection(p: dict):
    if USE_DB:
        try:
            sb.table("projections").insert(p).execute()
            return
        except Exception as e:
            st.warning(f"DB insert projection failed: {e}")
            return
    st.session_state.setdefault("projections", []).insert(0, p)

def db_delete_projection(pid: str):
    if USE_DB:
        try:
            sb.table("projections").delete().eq("id", pid).execute()
            return
        except Exception as e:
            st.warning(f"DB delete projection failed: {e}")
            return
    arr = st.session_state.get("projections", [])
    st.session_state["projections"] = [x for x in arr if x.get("id") != pid]

# ---------- Discord embeds -------------------------------
def _bet_color(typ: str) -> int:
    return {"Spread": 0x0ea5e9, "Total": 0xf59e0b, "Moneyline": 0x8b5cf6}.get(typ or "", 0x22c55e)

def _build_bet_embed(b: dict) -> dict:
    bettor = b.get("bettor", "Unknown")
    H = b.get("home_team", "Home")
    A = b.get("away_team", "Away")
    typ = (b.get("type") or "").strip()
    side = (b.get("side") or "").strip()  # "Home" | "Away" | "" (for Total)
    ou = (b.get("ou") or "").strip()      # "Over" | "Under" (for Total)
    price = b.get("price")
    odds_txt = f"{int(price):+d}" if isinstance(price, (int, float)) else "—"
    stake = b.get("stake")
    note = (b.get("note") or "").strip()

    # Build the “Spread | … @ price” line
    if typ == "Spread":
        hb = b.get("home_based_line")
        # Which team was picked?
        team = H if side == "Home" else A
        if hb is None:
            sel = f"{team}"
        else:
            # home_based_line is from home’s perspective; flip sign for Away picks
            line_for_pick = float(hb) if side == "Home" else -float(hb)
            sel = f"{team} {line_for_pick:+.1f}"
        mid = f"Spread | {sel} @ {odds_txt}"

    elif typ == "Total":
        ttl = b.get("total_line")
        ttl_txt = f"{float(ttl):.1f}" if isinstance(ttl, (int, float)) else "—"
        mid = f"Total | {ou} {ttl_txt} @ {odds_txt}"

    else:  # Moneyline
        team = H if side == "Home" else A
        mid = f"Moneyline | {team} ML @ {odds_txt}"

    header = f"🚨 **{bettor}'s New Bet** 🚨"
    game = f"{A} @ {H}"
    stake_line = f"Stake: {stake}u" if isinstance(stake, (int, float)) else "Stake: —"

    lines = [header, game, mid, stake_line]
    if note:
        lines += ["", note]  # note on its own line, no “Note:” label

    description = "\n".join(lines)

    return {
        # no title; everything is in the description per your spec
        "description": description,
        "color": _bet_color(typ),
        "timestamp": datetime.utcnow().isoformat() + "Z",  # Discord renders “Today at …”
    }


def _notify_discord_webhook(message: str | None = None, embed: dict | None = None) -> tuple[bool, str]:
    url = os.environ.get("DISCORD_WEBHOOK_URL") or st.secrets.get("DISCORD_WEBHOOK_URL")
    if not url:
        return False, "Discord webhook not configured (DISCORD_WEBHOOK_URL)."
    payload: dict = {}
    if message: payload["content"] = message
    if embed: payload["embeds"] = [embed]
    try:
        r = requests.post(url, json=payload, timeout=7)
        ok = 200 <= r.status_code < 300
        return ok, f"HTTP {r.status_code}"
    except requests.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, str(e)

def send_discord_bet(bet: dict):
    """Primary path: send embed directly to Discord webhook.
       Fallback: call Supabase Edge Function 'discord-bot' if webhook fails/unset."""
    embed = _build_bet_embed(bet)
    ok, detail = _notify_discord_webhook(embed=embed)
    if ok:
        return
    # Fallback to Edge Function (optional)
    sb_url = (os.environ.get("SB_URL") or st.secrets.get("supabase", {}).get("url"))
    srk = (
        os.environ.get("SB_SERVICE_KEY")
        or os.environ.get("SB_SERVICE_ROLE_KEY")
        or st.secrets.get("supabase", {}).get("service_key")
        or st.secrets.get("supabase", {}).get("anon_key")
    )
    if sb_url and srk:
        try:
            fn_url = f"{sb_url.rstrip('/')}/functions/v1/discord-bot"
            requests.post(
                fn_url,
                json={"op": "notify-bet", "bet": bet},
                headers={"Authorization": f"Bearer {srk}", "Content-Type": "application/json"},
                timeout=6
            )
            return
        except Exception:
            pass
    st.warning(f"Discord send failed: {detail}")

def trigger_weekly_recap(force: bool = True) -> tuple[bool, str]:
    """Call your weekly recap Edge Function. Tries 'weekly_recap' then 'weekly-recap'."""
    try:
        sb_url = os.environ.get("SB_URL") or st.secrets["supabase"]["url"]
        svc_key = os.environ.get("SB_SERVICE_KEY") or st.secrets["supabase"]["anon_key"]
    except Exception:
        return False, "SB_URL / SB_SERVICE_KEY not configured."

    if not sb_url or not svc_key:
        return False, "SB_URL / SB_SERVICE_KEY not set."

    def _call(fn_name: str) -> tuple[bool, str, int]:
        url = f"{sb_url.rstrip('/')}/functions/v1/{fn_name}"
        try:
            r = requests.get(
                url,
                headers={"Authorization": f"Bearer {svc_key}"},
                params={"force": "1"} if force else None,
                timeout=12.0,
            )
            try:
                payload = r.json()
            except Exception:
                payload = r.text
            return r.ok, f"{r.status_code} {payload}", r.status_code
        except requests.Timeout:
            return False, "timeout", 0
        except Exception as e:
            return False, str(e), 0

    # Your project shows weekly_recap — try that first, then the dashed variant
    ok, detail, code = _call("weekly_recap")
    if not ok and code == 404:
        ok, detail, _ = _call("weekly-recap")
    return ok, detail



# ---------- Header & Upload ----------
st.title("🥜 The Goober Model")
st.caption(
    "College Football Predictor — FPI & efficiencies with weather, market, and optional SOS / SOR / GC grounding. (Supabase-enabled)"
)

st.sidebar.header("📄 Upload data")
csv_file = st.sidebar.file_uploader("Combined CSV (FPI + Efficiencies)", type=["csv"], accept_multiple_files=False)

st.sidebar.markdown("### Admin / Debug")
if st.sidebar.button("🔔 Send weekly recap now"):
    ok, detail = trigger_weekly_recap(force=True)
    (st.sidebar.success if ok else st.sidebar.warning)(f"Weekly recap: {'sent' if ok else 'failed'} ({detail})")


DATA_READY = False
df = None
teams = []
n_teams = 0
home_team = ""
away_team = ""

if csv_file:
    try:
        df = pd.read_csv(csv_file, dtype=str, keep_default_na=False)
        if "Team" in df.columns:
            teams = sorted(df["Team"].unique().tolist())
            n_teams = max(1, len(teams))
            DATA_READY = True
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# ---------- Tabs (always render) ----------
tab_adj, tab_bets, tab_saved, tab_snap, tab_guide = st.tabs(
    ["⚙️ Adjustments", "🧾 Bet Board", "💾 Saved Projections", "📌 Matchup Snapshot", "ℹ️ Model Guide"]
)

# ---------- Adjustments tab ----------
with tab_adj:
    if not DATA_READY:
        st.info("Upload your combined CSV in the sidebar to use the predictor.", icon="🔎")
    else:
        # --- team pickers
        cA, cB = st.columns(2)
        with cA:
            home_team = st.selectbox("Home Team", teams, index=0 if teams else None)
        with cB:
            away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)
        if home_team == away_team:
            st.warning("Please select two different teams.", icon="⚠️")
            st.stop()

        home_row = get_team_row(df, home_team)
        away_row = get_team_row(df, away_team)
        (off_mean, def_mean, sp_mean, ovrl_mean, fpi_mean, fpi_min, fpi_max) = league_means(df)

        # --- adjustments UI
        st.subheader("Model & Market Adjustments")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            hfa_pts = st.slider(
                "Home Field Advantage (pts)", 0.0, 5.0, 2.5, 0.1, help="Adds to the HOME team’s expected margin. Typical CFB ~2–3 points."
            )
            base_total = st.slider(
                "Base Total (league avg)", 40.0, 70.0, 54.0, 0.5, help="Starting point for total points before efficiency & weather nudges."
            )
        with g2:
            alpha_total = st.slider(
                "α (OFF vs DEF)", 0.0, 1.0, 0.30, 0.01, help="How strongly good offense / weak defense pushes totals up (and vice versa)."
            )
            beta_st = st.slider(
                "β (Special Teams)", 0.0, 0.3, 0.05, 0.01, help="Small nudge from special teams efficiency; keep modest."
            )
        with g3:
            sigma_margin = st.slider(
                "σ (spread) — pts", 6.0, 21.0, 13.0, 0.5, help="How noisy game margins are. Larger σ ⇒ less certain spreads & cover %."
            )
            sigma_total = st.slider(
                "σ (total) — pts", 6.0, 21.0, 10.0, 0.5, help="How noisy totals are. Larger σ ⇒ P(Over/Under) closer to 50%."
            )
        with g4:
            neutral_site = st.checkbox("Neutral site", value=False, help="Remove HFA from the spread if played at a neutral site.")
            indoor_roof = st.checkbox("Indoors / Roof closed", value=False, help="Ignore weather if conditions are controlled.")

        st.markdown("#### Weather (ignored if indoors/roof closed)")
        w1, w2, w3 = st.columns(3)
        with w1:
            temp_f = st.slider("Temperature (°F)", 10, 100, 70, 1, help="Cold (<40°F) or hot (>85°F) slightly reduces expected scoring.")
        with w2:
            wind_mph = st.slider("Wind (mph)", 0, 40, 5, 1, help="Above ~10 mph, wind trims passing/kicking efficiency and lowers totals.")
        with w3:
            precip = st.select_slider(
                "Precipitation", options=["None", "Light", "Moderate", "Heavy"], value="None", help="Rain/snow reduces totals; heavier lowers more."
            )

        st.divider()
        st.subheader("Market Lines (for comparison / blending)")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            market_spread_home = st.number_input(
                "Market Spread (home line)", value=-3.5, step=0.5, help="Sportsbook home line (negative means home favorite)."
            )
        with m2:
            market_total = st.number_input("Market Total", value=54.5, step=0.5, help="Sportsbook total points line.")
        with m3:
            spread_price = st.number_input("Spread Price (American)", value=-110, step=5, help="Price for the spread (e.g., -110).")
        with m4:
            total_price = st.number_input("Total Price (American)", value=-110, step=5, help="Price for Over/Under (e.g., -110).")

        st.markdown("#### Blending")
        blend_weight = st.slider("Market Blend Weight (w)", 0.0, 1.0, 0.35, 0.05, help="0 = pure model; 1 = pure market.")

        # ----- Grounding + snapshot
        st.divider()
        st.subheader("Grounding (optional)")
        gleft, gright = st.columns([2, 1])

        with gleft:
            has_SOS = "SOS" in df.columns
            has_SOR = "SOR" in df.columns
            has_GC  = "GC"  in df.columns

            if not has_SOS:
                st.info("`SOS` column not found — SOS grounding disabled.", icon="ℹ️")
                adj_sos = False; sos_weight = 0.0; sos_sigma_pct = 0
            else:
                adj_sos = st.checkbox(
                    "Adjust for schedule strength (SOS rank 1=hardest)",
                    value=False,
                    help="Nudges the model margin for harder/easier schedules and can adjust spread σ."
                )
                sos_weight = st.slider("SOS weight (points)", 0.0, 2.0, 0.6, 0.1, disabled=not adj_sos)
                sos_sigma_pct = st.slider("SOS → σ multiplier (±%)", 0, 40, 0, 5, disabled=not adj_sos)

            if not (has_SOR and has_GC):
                st.info("`SOR` and/or `GC` columns not found — resume grounding disabled.", icon="ℹ️")
                adj_resume = False; sor_w = 0.0; gc_w = 0.0; resume_cap = 0.0; resume_sigma_pct = 0
            else:
                adj_resume = st.checkbox(
                    "Adjust for resume (SOR & GC ranks 1=best)",
                    value=False,
                    help="Tiny, capped margin nudge; optional σ scaling."
                )
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    sor_w = st.slider("SOR weight (pts)", 0.0, 1.0, 0.4, 0.1, disabled=not adj_resume)
                with c2:
                    gc_w  = st.slider("GC weight (pts)", 0.0, 1.0, 0.3, 0.1, disabled=not adj_resume)
                with c3:
                    resume_cap = st.slider("Resume nudge cap (±pts)", 0.0, 1.5, 1.0, 0.1, disabled=not adj_resume)
                with c4:
                    resume_sigma_pct = st.slider("Resume → σ multiplier (±%)", 0, 30, 0, 5, disabled=not adj_resume)

        with gright:
            st.markdown("**Current matchup — resume snapshot**")
            def _rank_center_pair(row, colname):
                if colname not in df.columns:
                    return "—"
                r = get_numeric(row, colname)
                if np.isnan(r) or r <= 0:
                    return "—"
                c = centered_from_rank(r, n_teams)
                return f"{int(r)} • {c:+.2f}"

            def _value_delta_pair(row, colname, mean, fmt="{:.2f}"):
                if colname not in df.columns:
                    return "—"
                v = get_numeric(row, colname)
                if np.isnan(v):
                    return "—"
                d = v - mean
                return f"{fmt.format(v)} • {d:+.2f}"

            data_rows = []
            data_rows.append({"Metric": "SOS (rank)", home_team: _rank_center_pair(home_row, "SOS"),
                              away_team: _rank_center_pair(away_row, "SOS")})
            data_rows.append({"Metric": "SOR (rank)", home_team: _rank_center_pair(home_row, "SOR"),
                              away_team: _rank_center_pair(away_row, "SOR")})
            data_rows.append({"Metric": "GC (rank)",  home_team: _rank_center_pair(home_row, "GC"),
                              away_team: _rank_center_pair(away_row, "GC")})
            data_rows.append({"Metric": "FPI",       home_team: _value_delta_pair(home_row, "FPI", fpi_mean),
                              away_team: _value_delta_pair(away_row, "FPI", fpi_mean)})
            data_rows.append({"Metric": "OVRL_EFF",  home_team: _value_delta_pair(home_row, "OVRL_EFF", ovrl_mean),
                              away_team: _value_delta_pair(away_row, "OVRL_EFF", ovrl_mean)})
            data_rows.append({"Metric": "OFF_EFF",   home_team: _value_delta_pair(home_row, "OFF_EFF", off_mean),
                              away_team: _value_delta_pair(away_row, "OFF_EFF", off_mean)})
            data_rows.append({"Metric": "DEF_EFF",   home_team: _value_delta_pair(home_row, "DEF_EFF", def_mean),
                              away_team: _value_delta_pair(away_row, "DEF_EFF", def_mean)})
            snap_df = pd.DataFrame(data_rows).set_index("Metric")
            st.table(snap_df)
            st.caption("Ranks show **rank • centered** in [-0.50,+0.50] (higher = harder/better). Values show **value • Δ** vs league mean.")

        # ---------- separator before results ----------
        st.markdown('<div class="page-sep"></div>', unsafe_allow_html=True)

        # ---------- Core model ----------
        FPI_h = get_numeric(home_row, "FPI");  FPI_a = get_numeric(away_row, "FPI")
        OFF_h = get_numeric(home_row, "OFF_EFF"); OFF_a = get_numeric(away_row, "OFF_EFF")
        DEF_h = get_numeric(home_row, "DEF_EFF"); DEF_a = get_numeric(away_row, "DEF_EFF")
        SP_h  = get_numeric(home_row, "SP_EFF");  SP_a  = get_numeric(away_row, "SP_EFF")

        spread_neutral = (FPI_h - FPI_a) if (not np.isnan(FPI_h) and not np.isnan(FPI_a)) else 0.0
        spread_model = spread_neutral if neutral_site else spread_neutral + hfa_pts

        has_SOS = "SOS" in df.columns
        hard_h = hard_a = 0.0
        if 'adj_sos' in locals() and adj_sos and has_SOS:
            sos_h_rank = get_numeric(home_row, "SOS"); sos_a_rank = get_numeric(away_row, "SOS")
            hard_h = centered_from_rank(sos_h_rank, n_teams)
            hard_a = centered_from_rank(sos_a_rank, n_teams)
            spread_model += sos_weight * (hard_h - hard_a)

        has_SOR = "SOR" in df.columns
        has_GC  = "GC"  in df.columns
        sor_h = sor_a = gc_h = gc_a = 0.0
        if 'adj_resume' in locals() and adj_resume and has_SOR and has_GC:
            sor_h_rank = get_numeric(home_row, "SOR"); sor_a_rank = get_numeric(away_row, "SOR")
            gc_h_rank  = get_numeric(home_row, "GC");  gc_a_rank  = get_numeric(away_row, "GC")
            sor_h = centered_from_rank(sor_h_rank, n_teams); sor_a = centered_from_rank(sor_a_rank, n_teams)
            gc_h  = centered_from_rank(gc_h_rank, n_teams);  gc_a  = centered_from_rank(gc_a_rank, n_teams)
            resume_nudge = sor_w * (sor_h - sor_a) + gc_w * (gc_h - gc_a)
            resume_nudge = float(np.clip(resume_nudge, -resume_cap, resume_cap))
            spread_model += resume_nudge

        off_term = center(OFF_h, off_mean) + center(OFF_a, off_mean)
        def_term = center(DEF_h, def_mean) + center(DEF_a, def_mean)
        sp_term  = center(SP_h,  sp_mean)  + center(SP_a,  sp_mean)
        eff_component = alpha_total * (off_term - def_term) / 2.0
        st_component  = beta_st * (sp_term) / 4.0

        weather_adj = 0.0
        if not indoor_roof:
            if wind_mph > 10: weather_adj -= max(0, (wind_mph - 10)) * 0.15
            if temp_f < 40:   weather_adj -= (40 - temp_f) * 0.05
            if temp_f > 85:   weather_adj -= (temp_f - 85) * 0.03
            precip_map = {"None":0.0, "Light":-0.5, "Moderate":-1.5, "Heavy":-3.0}
            weather_adj += precip_map.get(precip, 0.0)

        total_model = float(np.clip(base_total + eff_component + st_component + weather_adj, 20.0, 90.0))

        home_pts_model = clamp_nonneg((total_model + spread_model) / 2.0)
        away_pts_model = clamp_nonneg(total_model - home_pts_model)
        home_pts_model_r = int(round(home_pts_model)); away_pts_model_r = int(round(away_pts_model))

        home_line_model = _round_line(model_spread_to_home_line(spread_model))
        blend_home_line = _round_line(blend_weight * market_spread_home + (1.0 - blend_weight) * home_line_model)
        total_blended   = blend_weight * market_total + (1.0 - blend_weight) * total_model

        S_b = -blend_home_line
        home_pts_blend = clamp_nonneg((total_blended + S_b) / 2.0)
        away_pts_blend = clamp_nonneg(total_blended - home_pts_blend)
        home_pts_blend_r = int(round(home_pts_blend)); away_pts_blend_r = int(round(away_pts_blend))

        sigma_scale = 1.0
        if 'adj_sos' in locals() and adj_sos and has_SOS and sos_sigma_pct:
            sigma_scale *= 1.0 + (sos_sigma_pct/100.0) * (-(hard_h + hard_a))
        if 'adj_resume' in locals() and adj_resume and has_SOR and has_GC and resume_sigma_pct:
            avg_resume = (sor_h + sor_a + gc_h + gc_a) / 4.0
            sigma_scale *= 1.0 + (resume_sigma_pct/100.0) * (-(avg_resume))
        sigma_margin_eff = float(np.clip(sigma_margin * sigma_scale, 3.0, 40.0))

        p_home_win_model   = normal_cdf(spread_model / max(1e-9, sigma_margin_eff))
        p_home_cover_model = home_cover_probability(spread_model, market_spread_home, sigma_margin_eff)
        p_over_model       = 1.0 - normal_cdf((market_total - total_model) / max(1e-9, sigma_total))
        ev_spread_model    = ev_per_unit(p_home_cover_model, float(spread_price))
        ev_total_over      = ev_per_unit(p_over_model, float(total_price))
        ev_total_under     = ev_per_unit(1.0 - p_over_model, float(total_price))

        home_spread_str, away_spread_str = format_home_away_spreads(home_team, away_team, market_spread_home)
        spread_pick = home_spread_str if ev_spread_model >= 0 else away_spread_str
        tot_pick = f"Over {market_total:.1f}" if ev_total_over >= ev_total_under else f"Under {market_total:.1f}"
        recommendation = f"{spread_pick} | {tot_pick}"

        model_home_spread_str, _ = format_home_away_spreads(home_team, away_team, home_line_model)
        blended_home_spread_str, _ = format_home_away_spreads(home_team, away_team, blend_home_line)
        st.session_state.latest_projection = {
            "home_team": home_team, "away_team": away_team,
            "home_pts": float(home_pts_model), "away_pts": float(away_pts_model),
            "total_model": float(total_model), "total_blended": float(total_blended),
            "model_home_spread_str": model_home_spread_str,
            "blended_home_spread_str": blended_home_spread_str,
            "recommendation": recommendation,
        }

        st.markdown("### 🧾 Score Cards")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"""
<div class="score-card score-green">
  <div class="section-caption">Model Score</div>
  <div style="font-weight:700; font-size:1.05rem;">
    {home_team} {home_pts_model:.1f} — {away_team} {away_pts_model:.1f}
  </div>
  <div class="small">Rounded (median): {home_team} <b>{home_pts_model_r}</b> — {away_team} <b>{away_pts_model_r}</b></div>
  <div class="section-caption">Derived from model margin & total.</div>
</div>
""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
<div class="score-card score-yellow">
  <div class="section-caption">Blended Score</div>
  <div style="font-weight:700; font-size:1.05rem;">
    {home_team} {home_pts_blend:.1f} — {away_team} {away_pts_blend:.1f}
  </div>
  <div class="small">Rounded (median): {home_team} <b>{home_pts_blend_r}</b> — {away_team} <b>{away_pts_blend_r}</b></div>
  <div class="section-caption">Blend of model and market lines via weight w.</div>
</div>
""", unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""
<div class="value-chip">
  <div class="label">Model Home Line</div>
  <div class="value">{format_home_away_spreads(home_team, away_team, home_line_model)[0]}</div>
</div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
<div class="value-chip">
  <div class="label">Blended Home Line</div>
  <div class="value">{format_home_away_spreads(home_team, away_team, blend_home_line)[0]}</div>
</div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
<div class="value-chip">
  <div class="label">Market Home Line</div>
  <div class="value">{format_home_away_spreads(home_team, away_team, market_spread_home)[0]}</div>
</div>""", unsafe_allow_html=True)

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown(f"""
<div class="value-chip">
  <div class="label">Model Total</div>
  <div class="value">{total_model:.1f}</div>
</div>""", unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""
<div class="value-chip">
  <div class="label">Blended Total</div>
  <div class="value">{total_blended:.1f}</div>
</div>""", unsafe_allow_html=True)
        with tc3:
            st.markdown(f"""
<div class="value-chip">
  <div class="label">Market Total</div>
  <div class="value">{market_total:.1f}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📊 Prediction Details (always shown)")
        st.write(f"**Matchup:** {away_team} @ {home_team}{' (Neutral)' if neutral_site else ''}")

        home_line_model_lbl, _ = format_home_away_spreads(home_team, away_team, home_line_model)
        home_line_market_lbl, _ = format_home_away_spreads(home_team, away_team, market_spread_home)

        if abs(spread_model) < 0.05:
            favored_str = "Pick'em"
        else:
            favored_team = home_team if spread_model > 0 else away_team
            favored_str = f"{favored_team} by {abs(spread_model):.1f}"

        cL, cR = st.columns(2)
        with cL:
            st.subheader("Spread & Win Prob")
            st.markdown(
                f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="title">Model Margin (favored team)</div>
    <div class="num">{favored_str}</div>
  </div>
  <div class="stat-card">
    <div class="title">Model Home Line (sportsbook)</div>
    <div class="num">{home_line_model_lbl}</div>
  </div>
  <div class="stat-card">
    <div class="title">Home Win Probability (model)</div>
    <div class="num">{float(100*p_home_win_model):.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="title">P({home_line_market_lbl} covers)</div>
    <div class="num">{float(100*p_home_cover_model):.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="title">EV (spread @ {int(spread_price)})</div>
    <div class="num">{float(ev_spread_model):+0.2f}u</div>
  </div>
  <div class="stat-card">
    <div class="title">σ (spread, effective)</div>
    <div class="num">{sigma_margin_eff:.1f}</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.caption("Positive margin favors HOME. Home line = −(Model margin). Effective σ includes any SOS/SOR/GC scaling.")

        with cR:
            st.subheader("Totals")
            st.markdown(
                f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="title">Model Total</div>
    <div class="num">{total_model:.1f}</div>
  </div>
  <div class="stat-card">
    <div class="title">Probability Over {market_total:.1f}</div>
    <div class="num">{float(100*p_over_model):.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="title">EV Over (@ {int(total_price)})</div>
    <div class="num">{float(ev_total_over):+0.2f}u</div>
  </div>
  <div class="stat-card">
    <div class="title">EV Under (@ {int(total_price)})</div>
    <div class="num">{float(ev_total_under):+0.2f}u</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.subheader("Recommendation")
        st.markdown(f'<div class="rec-card">{recommendation}</div>', unsafe_allow_html=True)

# ---------- Bet Board (always available) ----------
with tab_bets:
    st.subheader("Bet Board")
    bc = st.container(border=True)
    with bc:
        left, right = st.columns([1, 1])
        with left:
            show_filter = st.selectbox("Board Settings — Show", ["All", "Pending only", "Wins only", "Losses only"])
        with right:
            if st.button("🔄 Refresh board"):
                st.rerun()

    bettor_names = ["Zak", "Tyler", "John"]
    st.markdown("### Log a Bet")

    # Bet type OUTSIDE the form so changing it re-renders dependent fields immediately
    bettor_names = ["Zak", "Tyler", "John"]
    bet_type_choice = st.selectbox("Bet Type", ["Spread", "Total", "Moneyline"], key="bet_type_choice")

    with st.form("bet_form_v2", clear_on_submit=True):
        l1, l2, l3 = st.columns(3)
        with l1:
            bettor = st.selectbox("Bettor", bettor_names, key="bettor_sel")
            sportsbook = st.text_input("Sportsbook", placeholder="(optional)")

        with l2:
            # Team inputs if no CSV
            if not DATA_READY:
                home_team_in = st.text_input("Home team", "")
                away_team_in = st.text_input("Away team", "")
                side_opts = ["Over", "Under"] if bet_type_choice == "Total" else ["Home", "Away"]
            else:
                home_team_in = home_team
                away_team_in = away_team
                side_opts = (
                    ["Over", "Under"]
                    if bet_type_choice == "Total"
                    else [f"Home ({home_team})", f"Away ({away_team})"]
                )
            side = st.selectbox("Side / O-U", side_opts, key="side_sel")

            # Lines depend on bet type
            home_based_line = None
            total_line = None
            if bet_type_choice == "Spread":
                home_based_line = st.number_input("Line (home-based; -3.5 = Home -3.5)", value=-3.5, step=0.5)
            elif bet_type_choice == "Total":
                total_line = st.number_input("Total (pts)", value=float(54.5), step=0.5)

        with l3:
            price = st.number_input("Price (American, optional)", value=-110, step=5)
            stake = st.number_input("Stake (units or dollars)", value=1.0, min_value=0.0, step=0.5)
            notify_discord = st.checkbox("🔔 Notify Discord", value=False)

        note = st.text_area("Description / Note (optional)", "")
        submitted = st.form_submit_button("Save Bet")

        if submitted:
            record = {
                "bettor": bettor,
                "book": sportsbook,
                "type": bet_type_choice,
                "side": (
                    "Home" if bet_type_choice != "Total" and str(side).startswith("Home")
                    else ("Away" if bet_type_choice != "Total" else None)
                ),
                "ou": (side if bet_type_choice == "Total" else None),
                "home_based_line": (float(home_based_line) if bet_type_choice == "Spread" and home_based_line is not None else None),
                "total_line": (float(total_line) if bet_type_choice == "Total" and total_line is not None else None),
                "price": float(price),
                "stake": float(stake),
                "status": "pending",
                "note": note,
                "notify_discord": bool(notify_discord),
                "home_team": home_team_in or "Home",
                "away_team": away_team_in or "Away",
            }
            row = db_add_bet(record)
            if notify_discord:
                send_discord_bet(row or record)
            st.success("Bet saved.")
            st.rerun()


    # Board columns
    def _pass_filter(b):
        if show_filter == "All":
            return True
        if show_filter == "Pending only":
            return b.get("status") == "pending"
        if show_filter == "Wins only":
            return b.get("status") == "win"
        if show_filter == "Losses only":
            return b.get("status") == "loss"
        return True

    bets = db_list_bets()
    c_zak, c_ty, c_john = st.columns(3)
    cols_map = {"Zak": c_zak, "Tyler": c_ty, "John": c_john}
    totals = {name: {"win": 0, "loss": 0, "push": 0} for name in cols_map}
    for b in bets:
        name = b.get("bettor", "")
        if name in totals and b.get("status") in totals[name]:
            totals[name][b.get("status")] += 1

    for name, col in cols_map.items():
        with col:
            t = totals[name]
            st.subheader(f"{name} — {t['win']}-{t['loss']}-{t['push']}")
            shown_any = False
            for i, b in enumerate(bets):
                if b.get("bettor") != name or not _pass_filter(b):
                    continue
                shown_any = True
                with st.container(border=True):
                    H = b.get("home_team", "Home")
                    A = b.get("away_team", "Away")
                    if b.get("type") == "Total":
                        sel_txt = f"{b.get('ou')} {float(b.get('total_line', 0)):.1f}"
                    elif b.get("type") == "Spread":
                        who = H if b.get("side") == "Home" else A
                        line = b.get("home_based_line")
                        if line is not None and float(line) > 0: line = f"+{float(line)}"
                        sel_txt = f"{b.get('side')} ({who}) {line}"
                    else:
                        who = H if b.get("side") == "Home" else A
                        sel_txt = f"{b.get('side')} ({who}) ML"

                    created = b.get("created_at", "")
                    st.markdown(f"**{sel_txt}**")
                    st.markdown(
                        f"<span class='muted'>{ b.get('type') }</span> {status_badge_html(b.get('status','pending'))}",
                        unsafe_allow_html=True,
                    )
                    st.write(f"Odds: {int(b.get('price', 0))} • {created}")
                    if b.get("note"): st.write(b["note"])
                    if b.get("book"): st.caption(f"Book: {b['book']}")

                    with st.expander("Edit"):
                        new_status = st.selectbox(
                            "Status",
                            ["pending", "win", "loss", "push"],
                            index=["pending", "win", "loss", "push"].index(b.get("status", "pending")),
                            key=f"stat_{name}_{i}",
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Save", key=f"save_{name}_{i}"):
                                db_update_bet_status(b.get("id", ""), new_status)
                                st.success("Status updated."); st.rerun()
                        with c2:
                            if st.button("Delete", key=f"del_{name}_{i}"):
                                db_delete_bet(b.get("id", "")); st.rerun()
            if not shown_any: st.caption("No bets yet.")

# ---------- Saved Projections ----------
with tab_saved:
    st.subheader("Saved Projections")
    st.caption("Backed by Supabase (falls back to local if not configured).")

    c1, c2 = st.columns([1, 1])
    with c1:
        disabled_save = not bool(st.session_state.get("latest_projection"))
        if st.button("💾 Save current projection", disabled=disabled_save):
            if disabled_save:
                st.warning("Nothing to save yet.")
            else:
                db_add_projection(st.session_state["latest_projection"])
                st.success("Projection saved.")
    with c2:
        if st.button("🧹 Clear all saved"):
            if USE_DB:
                for p in db_list_projections():
                    db_delete_projection(p.get("id", ""))
            else:
                st.session_state["projections"] = []
            st.success("Cleared.")

    projs = db_list_projections()
    if not projs:
        st.info("No saved projections yet.")
    else:
        def _safe_float(x):
            try: return float(x)
            except Exception: return np.nan
        def _spread_str_from_points(p):
            ht = p.get("home_team","Home"); at = p.get("away_team","Away")
            hp = _safe_float(p.get("home_pts")); ap = _safe_float(p.get("away_pts"))
            if not np.isnan(hp) and not np.isnan(ap):
                s = hp - ap
                return format_home_away_spreads(ht, at, model_spread_to_home_line(s))[0]
            return "N/A"
        for j, p in enumerate(projs):
            home_team_p = p.get("home_team","Home"); away_team_p = p.get("away_team","Away")
            tm = _safe_float(p.get("total_model")); tb = _safe_float(p.get("total_blended"))
            model_spread_str = p.get("model_home_spread_str") or _spread_str_from_points(p)
            blended_spread_str = p.get("blended_home_spread_str","N/A")
            hp = _safe_float(p.get("home_pts")); ap = _safe_float(p.get("away_pts"))
            with st.container(border=True):
                st.write(f"**{away_team_p} @ {home_team_p}**")
                if not np.isnan(hp) and not np.isnan(ap):
                    st.write(f"**Model:** {home_team_p} {hp:.1f} — {away_team_p} {ap:.1f}")
                st.write(f"Spread: {model_spread_str} | Total: {tm:.1f}" if not np.isnan(tm) else f"Spread: {model_spread_str}")
                if not np.isnan(tb): st.write(f"**Blended:** Spread {blended_spread_str} | Total {tb:.1f}")
                st.write(f"**Rec:** {p.get('recommendation','—')}")
                if st.button("Delete saved", key=f"dels_{p.get('id', j)}"):
                    db_delete_projection(p.get("id","")); st.rerun()

# ---------- Matchup Snapshot ----------
with tab_snap:
    if not DATA_READY:
        st.info("Upload a CSV to view the matchup snapshot.", icon="ℹ️")
    else:
        st.subheader("Matchup Snapshot")
        st.caption("Bars compare teams on ranks (higher = better/harder) and efficiencies (0–100 scale). Tooltip shows raw values.")

        def _rank_score(row, col):
            r = get_numeric(row, col)
            if np.isnan(r) or r <= 0: return np.nan, "—"
            score = (n_teams + 1 - r) / n_teams * 100.0
            return score, f"rank {int(r)}"
        def _eff_score(row, col):
            v = get_numeric(row, col)
            if np.isnan(v): return np.nan, "—"
            return float(v), f"{v:.1f}"
        def _fpi_score(row):
            v = get_numeric(row, "FPI")
            if np.isnan(v): return np.nan, "—"
            rng = max(1e-6, float(fpi_max - fpi_min))
            score = (float(v) - float(fpi_min)) / rng * 100.0
            return score, f"{v:.2f}"

        home_row = get_team_row(df, home_team)
        away_row = get_team_row(df, away_team)
        (off_mean, def_mean, sp_mean, ovrl_mean, fpi_mean, fpi_min, fpi_max) = league_means(df)

        metrics = [
            ("Strength of Schedule (rank)", "SOS", "rank"),
            ("Strength of Record (rank)",   "SOR", "rank"),
            ("Game Control (rank)",         "GC",  "rank"),
            ("Football Power Index",        "FPI", "fpi"),
            ("Overall Efficiency",          "OVRL_EFF", "eff"),
            ("Offensive Efficiency",        "OFF_EFF",  "eff"),
            ("Defensive Efficiency",        "DEF_EFF",  "eff"),
            ("Special Teams Efficiency",    "SP_EFF",   "eff"),
        ]

        rows = []
        for label, col, typ in metrics:
            if col not in df.columns: 
                continue
            if typ == "rank":
                hs, hd = _rank_score(home_row, col); as_, ad = _rank_score(away_row, col)
            elif typ == "fpi":
                hs, hd = _fpi_score(home_row);        as_, ad = _fpi_score(away_row)
            else:
                hs, hd = _eff_score(home_row, col);   as_, ad = _eff_score(away_row, col)
            rows += [
                {"MetricNice": label, "Team": home_team, "Score": hs, "Display": hd},
                {"MetricNice": label, "Team": away_team, "Score": as_, "Display": ad},
            ]
        longdf = pd.DataFrame(rows).dropna()
        order = [m[0] for m in metrics if m[1] in df.columns]

        if alt is not None and not longdf.empty:
            chart = (
                alt.Chart(longdf)
                .mark_bar(size=22)
                .encode(
                    y=alt.Y("MetricNice:N", sort=list(reversed(order)),
                            axis=alt.Axis(title=None, labelLimit=260)),
                    x=alt.X("Score:Q", title="Scaled score (0–100)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Team:N", legend=alt.Legend(orient="bottom")),
                    tooltip=["Team:N","MetricNice:N","Display:N"]
                )
                .properties(height=alt.Step(30))
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            if longdf.empty:
                st.info("Snapshot unavailable for this CSV.")
            else:
                st.dataframe(longdf.pivot_table(index="MetricNice", columns="Team",
                                                values="Display", aggfunc="first"))

# ---------- Model Guide ----------
with tab_guide:
    st.subheader("Model Guide")
    st.markdown(
        """
### Acronyms
- **FPI** — Football Power Index (rating; higher = better).
- **SOS** — *Strength of Schedule* **rank** to date (1 = hardest).
- **SOR** — *Strength of Record* **rank** (1 = best resume vs schedule).
- **GC** — *Game Control* **rank** (1 = most wire-to-wire control).
- **OFF/DEF/SP_EFF** — Offense / Defense / Special Teams efficiencies (0–100 style).

---

### What the model does (high level)
1) **Model Margin (Home − Away)** starts from FPI and Home Field Advantage (HFA).  
2) **Model Total** starts at a base league average and is nudged by:
   - Offense vs Defense balance (**α**)  
   - Special teams (**β**)  
   - Weather (temp/wind/precip; ignored indoors/roof closed)
3) **Grounding (optional)** applies small, bounded corrections:
   - **SOS** (harder/easier slates): tiny margin nudge; can also alter spread **σ**  
   - **Resume (SOR & GC)**: tiny, capped margin nudge; can also alter spread **σ**
4) **Market blending (w)** mixes your model with sportsbook numbers to produce **Blended** line/total.
5) **Probabilities & EV** come from treating margin/total as noisy (Normal) with adjustable **σ**.

---

### Controls & what they change
- **Home Field Advantage (pts)** — adds directly to the home team in Model Margin  
- **Base Total** — starting point for Model Total before nudges  
- **α (OFF vs DEF)** — how much offenses vs defenses push **totals**  
- **β (Special Teams)** — small totals nudge from special teams  
- **σ (spread)** — uncertainty for margin; affects win/cover probabilities and EV  
- **σ (total)** — uncertainty for the game total; affects P(Over/Under) and EV  
- **Neutral site** — removes HFA  
- **Indoors / Roof closed** — ignores weather effects  
- **Weather** — temp, wind, precipitation each reduce totals a bit when adverse  
- **Market Spread / Market Total / Prices** — used for P(cover/over) and EV  
- **Blend weight (w)** — blend of market & model → **Blended** outputs  
- **SOS / Resume** — tiny, controlled margin nudges; optional **σ** shrink/expand

---

### Reading the outputs
- **Model Margin (favored team)** — who’s favored & by how much (home minus away).  
  **Model Home Line** converts that to sportsbook notation: `Home line = −(Model margin)`.
- **Model Total** — implied points after all nudges.
- **Score Cards** — median scores implied by Model or **Blended** numbers.
- **Home Win Prob / P(Home line covers)** — probabilities from the model + **σ (spread, effective)**.
- **Probability Over** — P(game total > market total) using **σ (total)**.
- **EV** — expected value per 1u at given American odds (positive = +EV).
- **σ (spread, effective)** — final uncertainty after SOS/Resume scaling (smaller = more confident).

**Tip:** Use grounding sparingly. The idea is to *temper* ratings (e.g., soft schedules inflate teams), not to rewrite them. Keep σ realistic; overly small σ will overstate edges.
        """
    )

