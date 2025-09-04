#!/usr/bin/env python3
"""
merge_fpi_csvs.py — Merge three ESPN FPI CSVs by Team, replacing W-L with Wins/Losses

What it does
- Loads FPI, RESUME, EFFICIENCIES (any row order) and merges on Team.
- De-duplicates by Team.
- Keeps a single CONF column (prefers FPI's).
- **Parses FPI's 'W-L' into numeric 'Wins' and 'Losses'** even if it's been turned into
  date-like text such as 'Jan-00', '2-Oct', '10/2/2025', '2025-10-02', etc.
- Drops 'W-L' entirely before merging. Leaves other columns (e.g., 'PROJ W-L') untouched.
- Sorts output alphabetically by Team.
- Writes a CSV only.

Usage:
  python merge_fpi_csvs.py \
    --fpi "fpi_rankings - FPI.csv" \
    --resume "fpi_rankings - RESUME.csv" \
    --eff "fpi_rankings - EFFICIENCIES.csv" \
    -o "fpi_rankings - COMBINED.csv"
"""

import argparse
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# ------------------------
# Helpers
# ------------------------

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def month_to_num(tok: str) -> Optional[int]:
    return MONTHS.get(tok.lower())

def clean_team(s: str) -> str:
    """Normalize team strings (quotes, nbsp, extra whitespace)."""
    if pd.isna(s):
        return s
    s = str(s)
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV as strings, normalize headers and Team, drop blank/dup Team rows."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    if "Team" not in df.columns:
        raise ValueError(f"'Team' column not found in {path}")
    df["Team"] = df["Team"].map(clean_team)
    df = df[df["Team"].notna() & (df["Team"] != "")]
    df = df.drop_duplicates(subset=["Team"], keep="first")
    if "CONF" in df.columns:
        df["CONF"] = df["CONF"].astype(str).str.strip()
    return df

def _norm_colname(name: str) -> str:
    """Normalize a header for robust matching (lowercase, collapse spaces, unify dashes)."""
    s = name.lower().strip()
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)  # any dash -> hyphen
    s = re.sub(r"\s+", "", s)  # remove spaces
    return s

def find_wl_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find the W-L column under various header spellings:
    e.g., 'W-L', 'W–L', 'W L', 'wl', 'record'. Prefer exact 'W-L' if present.
    """
    if "W-L" in df.columns:
        return "W-L"
    candidates = { _norm_colname(c): c for c in df.columns }
    for key in ("w-l", "wl", "record"):
        if key in candidates:
            return candidates[key]
    return None

def parse_wl(val) -> Tuple[Optional[int], Optional[int]]:
    """
    Bulletproof parser for Wins/Losses from a text cell that may already be
    mutated by Excel into a 'date-like' string.

    Handles:
      - "10-2", "10–2" (various dashes)
      - "10/2" or "10/2/2025"
      - "Oct-02", "October 2", "2-Oct", "2 October"
      - "2025-10-02" (returns 10,2)
      - Extra decorations like parentheses/commas
      - Day 0 allowed (e.g., "Jan-00" -> 1, 0)
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return (None, None)

    s = str(val).strip()
    if not s:
        return (None, None)

    # Normalize dashes to hyphen
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)

    # Extract all integers once (for several branches)
    nums = [int(x) for x in re.findall(r"\d+", s)]

    # Month name first (e.g., "Oct-02", "October 2")
    m = re.search(r"\b([A-Za-z]{3,9})\b[\s\-\/,]*([0-3]?\d)", s)
    if m:
        mm = month_to_num(m.group(1))
        if mm:
            day = int(m.group(2))
            if 0 <= day <= 31:
                return mm, day

    # Day first then month name (e.g., "2-Oct", "2 October")
    m = re.search(r"\b([0-3]?\d)\b[\s\-\/,]*([A-Za-z]{3,9})", s)
    if m:
        day = int(m.group(1))
        mm = month_to_num(m.group(2))
        if mm and 0 <= day <= 31:
            return mm, day

    # If there are 3+ integers, prefer a (month, day) adjacent pair
    if len(nums) >= 3:
        for i in range(len(nums) - 1):
            a, b = nums[i], nums[i + 1]
            if 1 <= a <= 12 and 0 <= b <= 31:
                return a, b

    # If there are exactly 2 integers, use them directly
    if len(nums) == 2:
        return nums[0], nums[1]

    return (None, None)

def safe_rename_resume(df_resume: pd.DataFrame, fpi_cols: set) -> pd.DataFrame:
    """Add '_resume' suffix to any resume columns that also exist in FPI (excluding Team/CONF)."""
    rename_map = {}
    for c in df_resume.columns:
        if c in ("Team", "CONF"):
            continue
        if c in fpi_cols:
            rename_map[c] = f"{c}_resume"
    return df_resume.rename(columns=rename_map)


# ------------------------
# Core
# ------------------------

def add_wins_losses_from_wl_and_drop(df_fpi: pd.DataFrame) -> pd.DataFrame:
    """
    On the FPI dataframe:
      - Find W-L,
      - Create integer Wins/Losses with robust parsing,
      - Drop W-L entirely.
    If not found, still ensure Wins/Losses exist (empty).
    """
    wl_col = find_wl_column(df_fpi)
    if wl_col and wl_col in df_fpi.columns:
        wins_list: List[Optional[int]] = []
        losses_list: List[Optional[int]] = []
        for val in df_fpi[wl_col]:
            w, l = parse_wl(val)
            wins_list.append(w)
            losses_list.append(l)
        # Use pandas nullable integer dtype to avoid "0.0" formatting
        df_fpi["Wins"] = pd.Series(wins_list, dtype="Int64")
        df_fpi["Losses"] = pd.Series(losses_list, dtype="Int64")
        df_fpi = df_fpi.drop(columns=[wl_col])
    else:
        if "Wins" not in df_fpi.columns:
            df_fpi["Wins"] = pd.Series([None] * len(df_fpi), dtype="Int64")
        if "Losses" not in df_fpi.columns:
            df_fpi["Losses"] = pd.Series([None] * len(df_fpi), dtype="Int64")
    return df_fpi

def merge_frames(fpi_path: str, resume_path: str, eff_path: str) -> pd.DataFrame:
    df_fpi = load_csv(fpi_path)
    df_res = load_csv(resume_path)
    df_eff = load_csv(eff_path)

    # Build Wins/Losses from W-L on the FPI frame BEFORE merging
    df_fpi = add_wins_losses_from_wl_and_drop(df_fpi)

    # Suffix resume columns that collide with FPI (except Team/CONF)
    fpi_cols = set(df_fpi.columns) - {"Team", "CONF"}
    df_res = safe_rename_resume(df_res, fpi_cols)

    # Merge on Team; drop CONF from right frames so FPI's CONF wins
    merged = df_fpi.merge(
        df_res.drop(columns=[c for c in ["CONF"] if c in df_res.columns]),
        on="Team",
        how="outer",
        # validate="one_to_one",  # comment out to be resilient if ESPN duplicates slip in
    )
    merged = merged.merge(
        df_eff.drop(columns=[c for c in ["CONF"] if c in df_eff.columns]),
        on="Team",
        how="outer",
        # validate="one_to_one",
    )

    # If any CONF_x/CONF_y slipped in, fold them down
    conf_like = [c for c in merged.columns if c.startswith("CONF")]
    if len(conf_like) > 1:
        merged["CONF"] = merged[conf_like].bfill(axis=1).iloc[:, 0]
        merged = merged.drop(columns=[c for c in conf_like if c != "CONF"])

    # Order columns: Team, CONF, Wins, Losses, then the rest
    first = ["Team"]
    if "CONF" in merged.columns:
        first.append("CONF")
    first += ["Wins", "Losses"]

    rest = [c for c in merged.columns if c not in first]
    merged = merged[first + rest].sort_values("Team").reset_index(drop=True)
    return merged


def main():
    ap = argparse.ArgumentParser(description="Merge ESPN FPI CSVs by Team (W-L removed; Wins/Losses added, robust parsing)")
    ap.add_argument("--fpi", required=True, help="Path to FPI CSV")
    ap.add_argument("--resume", required=True, help="Path to RESUME CSV")
    ap.add_argument("--eff", required=True, help="Path to EFFICIENCIES CSV")
    ap.add_argument("-o", "--out", default="fpi_rankings - COMBINED.csv", help="Output CSV path")
    args = ap.parse_args()

    merged = merge_frames(args.fpi, args.resume, args.eff)

    # Optional: report rows that failed parsing (should be none after this fix)
    fails = merged[merged["Wins"].isna() | merged["Losses"].isna()]
    if not fails.empty:
        print(f"[info] {len(fails)} row(s) had missing Wins/Losses after parsing. Example teams:",
              ", ".join(fails["Team"].head(10).astype(str)))

    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(merged)} rows and {len(merged.columns)} columns.")


if __name__ == "__main__":
    main()
