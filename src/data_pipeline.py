"""
World Cup 2026 Predictor — Data Pipeline
-----------------------------------------
Input  (place in data/raw/):
    results.csv   — Kaggle international football results dataset

Outputs (written to data/processed/):
    matches_featured.csv
    train.csv / val.csv / test.csv

Elo ratings are computed from scratch using all matches in results.csv
(including pre-2010 data) so that 2010+ matches have well-warmed-up ratings.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Works whether you run from VSCode, terminal, or Jupyter
ROOT      = Path(__file__).resolve().parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

print(f"Project root : {ROOT}")
print(f"Looking for  : {RAW / 'results.csv'}")

# ── Competition whitelist ──────────────────────────────────────────────────────
COMPETITIONS = [
    # World Cup
    "FIFA World Cup",
    "FIFA World Cup qualification",
    # UEFA
    "UEFA Euro",
    "UEFA Euro qualification",
    #"UEFA Nations League",
    # CONMEBOL
    "Copa América",
    # CONCACAF  (dataset calls it "Gold Cup", not "CONCACAF Gold Cup")
    "Gold Cup",
    "Gold Cup qualification",
    #"CONCACAF Nations League",
    #"CONCACAF Nations League qualification",
    # CAF  (dataset uses "African", not "Africa")
    "African Cup of Nations",
    "African Cup of Nations qualification",
    # AFC
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    # FIFA inter-confederation
    "Confederations Cup",
]

# ── Federation map ─────────────────────────────────────────────────────────────
FEDERATION_MAP = {
    "Germany":"UEFA","France":"UEFA","Spain":"UEFA","England":"UEFA",
    "Italy":"UEFA","Portugal":"UEFA","Netherlands":"UEFA","Belgium":"UEFA",
    "Croatia":"UEFA","Switzerland":"UEFA","Denmark":"UEFA","Sweden":"UEFA",
    "Norway":"UEFA","Poland":"UEFA","Serbia":"UEFA","Austria":"UEFA",
    "Scotland":"UEFA","Wales":"UEFA","Czech Republic":"UEFA","Czechia":"UEFA",
    "Hungary":"UEFA","Romania":"UEFA","Slovakia":"UEFA","Turkey":"UEFA",
    "Türkiye":"UEFA","Greece":"UEFA","Ukraine":"UEFA","Russia":"UEFA",
    "Bosnia and Herzegovina":"UEFA","Kosovo":"UEFA","Finland":"UEFA",
    "Iceland":"UEFA","Republic of Ireland":"UEFA","Ireland":"UEFA",
    "North Macedonia":"UEFA","Albania":"UEFA","Slovenia":"UEFA",
    "Montenegro":"UEFA","Bulgaria":"UEFA","Georgia":"UEFA","Armenia":"UEFA",
    "Azerbaijan":"UEFA","Belarus":"UEFA","Estonia":"UEFA","Latvia":"UEFA",
    "Lithuania":"UEFA","Moldova":"UEFA","Luxembourg":"UEFA","Malta":"UEFA",
    "Cyprus":"UEFA","Faroe Islands":"UEFA","Andorra":"UEFA","Gibraltar":"UEFA",
    "Liechtenstein":"UEFA","San Marino":"UEFA","Northern Ireland":"UEFA",
    "Brazil":"CONMEBOL","Argentina":"CONMEBOL","Uruguay":"CONMEBOL",
    "Colombia":"CONMEBOL","Chile":"CONMEBOL","Ecuador":"CONMEBOL",
    "Peru":"CONMEBOL","Venezuela":"CONMEBOL","Paraguay":"CONMEBOL",
    "Bolivia":"CONMEBOL",
    "Mexico":"CONCACAF","United States":"CONCACAF","Canada":"CONCACAF",
    "Costa Rica":"CONCACAF","Honduras":"CONCACAF","Jamaica":"CONCACAF",
    "Panama":"CONCACAF","Trinidad and Tobago":"CONCACAF","Haiti":"CONCACAF",
    "El Salvador":"CONCACAF","Guatemala":"CONCACAF","Cuba":"CONCACAF",
    "Curaçao":"CONCACAF","Curacao":"CONCACAF",
    "Senegal":"CAF","Morocco":"CAF","Nigeria":"CAF","Ghana":"CAF",
    "Egypt":"CAF","Algeria":"CAF","Tunisia":"CAF","Cameroon":"CAF",
    "Ivory Coast":"CAF","Côte d'Ivoire":"CAF","South Africa":"CAF",
    "Mali":"CAF","Burkina Faso":"CAF","Guinea":"CAF","Tanzania":"CAF",
    "DR Congo":"CAF","Congo DR":"CAF","Zambia":"CAF","Zimbabwe":"CAF",
    "Uganda":"CAF","Kenya":"CAF","Ethiopia":"CAF","Angola":"CAF",
    "Cape Verde":"CAF","Cabo Verde":"CAF","Gabon":"CAF","Benin":"CAF",
    "Equatorial Guinea":"CAF","Namibia":"CAF","Mauritania":"CAF",
    "Libya":"CAF","Sudan":"CAF","Rwanda":"CAF","Mozambique":"CAF",
    "Japan":"AFC","South Korea":"AFC","Australia":"AFC","Iran":"AFC",
    "Saudi Arabia":"AFC","Qatar":"AFC","UAE":"AFC",
    "United Arab Emirates":"AFC","Iraq":"AFC","Jordan":"AFC",
    "Uzbekistan":"AFC","Oman":"AFC","Bahrain":"AFC","Kuwait":"AFC",
    "China":"AFC","China PR":"AFC","India":"AFC","Vietnam":"AFC",
    "Thailand":"AFC","Malaysia":"AFC","Indonesia":"AFC",
    "Philippines":"AFC","Kyrgyzstan":"AFC","Tajikistan":"AFC",
    "North Korea":"AFC","Palestine":"AFC","Lebanon":"AFC",
    "Hong Kong":"AFC","Myanmar":"AFC",
    "New Zealand":"OFC","Fiji":"OFC","Papua New Guinea":"OFC",
    "Solomon Islands":"OFC","Vanuatu":"OFC","Tahiti":"OFC",
}

ELO_K       = 32
ELO_DEFAULT = 1500

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL RESULTS (full history for Elo warm-up)
# ══════════════════════════════════════════════════════════════════════════════

def load_all_results(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[raw] {len(df):,} total matches ({df['date'].min().date()} → {df['date'].max().date()})")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE ELO FROM SCRATCH
# ══════════════════════════════════════════════════════════════════════════════

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def compute_elo_timeseries(all_matches):
    ratings   = {}
    pre_match = {}   # original_index -> (home_elo_before, away_elo_before)

    for idx, row in all_matches.iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        r_h   = ratings.get(home, ELO_DEFAULT)
        r_a   = ratings.get(away, ELO_DEFAULT)

        pre_match[idx] = (r_h, r_a)

        if row["home_score"] > row["away_score"]:
            s_h, s_a = 1.0, 0.0
        elif row["home_score"] == row["away_score"]:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        e_h = expected_score(r_h, r_a)
        ratings[home] = r_h + ELO_K * (s_h - e_h)
        ratings[away] = r_a + ELO_K * (s_a - (1 - e_h))

    print(f"[elo] Done — {len(ratings)} unique teams")
    return pre_match

# ══════════════════════════════════════════════════════════════════════════════
# 3. LEGACY ELO (20-year rolling average, point-in-time correct)
# ══════════════════════════════════════════════════════════════════════════════

def compute_legacy_elo(all_matches, pre_match_elo, window_years=20):
    # Build long-format Elo time series
    records = []
    for idx, row in all_matches.iterrows():
        h_elo, a_elo = pre_match_elo[idx]
        records.append({"date": row["date"], "team": row["home_team"], "elo": h_elo})
        records.append({"date": row["date"], "team": row["away_team"], "elo": a_elo})

    elo_ts = (pd.DataFrame(records)
                .sort_values(["team", "date"])
                .reset_index(drop=True))

    # For each team, compute expanding 20-year rolling mean
    # We use a merge_asof-style approach: group by team, rolling with time window
    legacy_map = {}  # (orig_index, "home"/"away") -> legacy_elo

    # Group elo_ts by team into a dict for fast lookup
    team_elo = {t: g.reset_index(drop=True) for t, g in elo_ts.groupby("team")}

    for idx, row in all_matches.iterrows():
        cutoff = row["date"]
        start  = cutoff - pd.DateOffset(years=window_years)

        for side, team in [("home", row["home_team"]), ("away", row["away_team"])]:
            tdf = team_elo.get(team)
            if tdf is None:
                legacy_map[(idx, side)] = ELO_DEFAULT
                continue
            mask = (tdf["date"] >= start) & (tdf["date"] < cutoff)
            vals = tdf.loc[mask, "elo"]
            legacy_map[(idx, side)] = vals.mean() if len(vals) > 0 else ELO_DEFAULT

    print(f"[legacy] Done")
    return legacy_map

# ══════════════════════════════════════════════════════════════════════════════
# 4. FORM: LAST 10 MATCHES
# ══════════════════════════════════════════════════════════════════════════════

def compute_form(all_matches):
    home_df = all_matches[["date","home_team","home_score","away_score"]].copy()
    home_df.columns = ["date","team","gf","ga"]
    away_df = all_matches[["date","away_team","away_score","home_score"]].copy()
    away_df.columns = ["date","team","gf","ga"]

    long = pd.concat([home_df, away_df], ignore_index=True)
    long["win"] = (long["gf"] > long["ga"]).astype(float)
    long["gd"]  = long["gf"] - long["ga"]
    long = long.sort_values(["team","date"]).reset_index(drop=True)

    long["win_rate"]     = (long.groupby("team")["win"]
                               .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()))
    long["goal_diff_pg"] = (long.groupby("team")["gd"]
                               .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()))

    # Build lookup (team, date) -> stats; keep first entry per (team, date)
    long_deduped = long.drop_duplicates(subset=["team","date"], keep="first")
    form_lookup  = long_deduped.set_index(["team","date"])[["win_rate","goal_diff_pg"]].to_dict("index")

    print(f"[form] Done — {long['team'].nunique()} teams")
    return form_lookup

# ══════════════════════════════════════════════════════════════════════════════
# 5. FILTER TO TRAINING WINDOW & COMPETITIONS
# ══════════════════════════════════════════════════════════════════════════════

def filter_matches(df):
    df = df[(df["date"] >= "2010-01-01") & (df["date"] <= "2025-12-31")].copy()
    pattern = "|".join(COMPETITIONS)
    df = df[df["tournament"].str.contains(pattern, case=False, na=False)]
    df = df.reset_index()   # keep original index as column "index"
    print(f"[filter] {len(df):,} matches")
    print(df["tournament"].value_counts().to_string())
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 6. ASSEMBLE FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_features(filtered, pre_match_elo, legacy_map, form_lookup):
    def get_result(row):
        if row["home_score"] > row["away_score"]: return 2
        if row["home_score"] == row["away_score"]: return 1
        return 0

    filtered = filtered.copy()
    filtered["result"] = filtered.apply(get_result, axis=1)

    rows = []
    for _, match in filtered.iterrows():
        orig = match["index"]   # original index into all_matches
        home = match["home_team"]
        away = match["away_team"]
        date = match["date"]

        h_elo, a_elo = pre_match_elo.get(orig, (ELO_DEFAULT, ELO_DEFAULT))
        h_leg = legacy_map.get((orig, "home"), ELO_DEFAULT)
        a_leg = legacy_map.get((orig, "away"), ELO_DEFAULT)

        hf = form_lookup.get((home, date), {})
        af = form_lookup.get((away, date), {})

        rows.append({
            "date":            date,
            "home_team":       home,
            "away_team":       away,
            "tournament":      match["tournament"],
            "result":          match["result"],
            "home_elo":        h_elo,
            "away_elo":        a_elo,
            "elo_diff":        h_elo - a_elo,
            "home_legacy":     h_leg,
            "away_legacy":     a_leg,
            "legacy_diff":     h_leg - a_leg,
            "home_win_rate":   hf.get("win_rate", 0.5),
            "away_win_rate":   af.get("win_rate", 0.5),
            "home_gd_pg":      hf.get("goal_diff_pg", 0.0),
            "away_gd_pg":      af.get("goal_diff_pg", 0.0),
            "home_federation": FEDERATION_MAP.get(home, "OTHER"),
            "away_federation": FEDERATION_MAP.get(away, "OTHER"),
        })

    df = pd.DataFrame(rows)
    print(f"[features] Shape: {df.shape}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 7. ENCODE & SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def encode(df):
    df = pd.get_dummies(df, columns=["home_federation","away_federation"], drop_first=False)

    # Fill NaNs — appear for early matches (2000-2009) where rolling windows
    # don't have enough prior history, or for obscure teams missing Elo data.
    elo_cols  = ["home_elo","away_elo","elo_diff",
                 "home_legacy","away_legacy","legacy_diff"]
    form_cols = ["home_win_rate","away_win_rate","home_gd_pg","away_gd_pg"]

    for col in elo_cols:
        if col in df.columns:
            median   = df[col].median()
            n_filled = df[col].isna().sum()
            if n_filled > 0:
                print(f"  [fillna] {col}: {n_filled} NaNs → median ({median:.1f})")
            df[col] = df[col].fillna(median)

    for col in form_cols:
        if col in df.columns:
            fill_val = 0.5 if "win_rate" in col else 0.0
            n_filled = df[col].isna().sum()
            if n_filled > 0:
                print(f"  [fillna] {col}: {n_filled} NaNs → {fill_val}")
            df[col] = df[col].fillna(fill_val)

    # Safety net for anything else
    remaining = df.isnull().sum().sum()
    if remaining > 0:
        print(f"  [fillna] {remaining} remaining NaNs → column median")
        df = df.fillna(df.median(numeric_only=True))

    return df

def time_split(df):
    df = df.sort_values("date").reset_index(drop=True)
    n  = len(df)
    i1, i2 = int(n*0.70), int(n*0.85)
    train, val, test = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    for name, s in [("train",train),("val",val),("test",test)]:
        print(f"[split] {name:5s}: {len(s):,}  ({s['date'].min().date()} → {s['date'].max().date()})")
    return train, val, test

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== 1. Load all results ===")
    all_matches = load_all_results(RAW / "results.csv")

    print("\n=== 2. Compute Elo ===")
    pre_match_elo = compute_elo_timeseries(all_matches)

    print("\n=== 3. Compute legacy Elo ===")
    legacy_map = compute_legacy_elo(all_matches, pre_match_elo)

    print("\n=== 4. Compute form ===")
    form_lookup = compute_form(all_matches)

    print("\n=== 5. Filter to 2000-2025 competitive matches ===")
    filtered = filter_matches(all_matches)

    print("\n=== 6. Build features ===")
    featured = build_features(filtered, pre_match_elo, legacy_map, form_lookup)
    featured = encode(featured)
    featured.to_csv(PROCESSED / "matches_featured.csv", index=False)

    print("\n=== 7. Split ===")
    train, val, test = time_split(featured)
    train.to_csv(PROCESSED / "train.csv", index=False)
    val.to_csv(PROCESSED / "val.csv",     index=False)
    test.to_csv(PROCESSED / "test.csv",   index=False)

    feat_cols = [c for c in featured.columns
                 if c not in ["date","home_team","away_team","tournament","result"]]
    print(f"\nFeatures ({len(feat_cols)}): {feat_cols}")
    print(f"\nClass balance:\n{featured['result'].value_counts(normalize=True).rename({2:'Home Win',1:'Draw',0:'Away Win'}).to_string()}")

if __name__ == "__main__":
    main()
