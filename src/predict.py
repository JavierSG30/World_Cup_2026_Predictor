"""
World Cup 2026 — Tournament Predictor & Monte Carlo Simulator
--------------------------------------------------------------
Follows the real FIFA 2026 bracket structure exactly.

Run:
    python src/predict.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

ROOT   = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

NAME_MAP = {"Czechia": "Czech Republic", "Türkiye": "Turkey"}

FEDERATION_MAP = {
    "Germany":"UEFA","France":"UEFA","Spain":"UEFA","England":"UEFA",
    "Portugal":"UEFA","Netherlands":"UEFA","Belgium":"UEFA","Croatia":"UEFA",
    "Switzerland":"UEFA","Sweden":"UEFA","Norway":"UEFA","Austria":"UEFA",
    "Scotland":"UEFA","Czechia":"UEFA","Czech Republic":"UEFA",
    "Bosnia and Herzegovina":"UEFA","Turkey":"UEFA","Türkiye":"UEFA",
    "Brazil":"CONMEBOL","Argentina":"CONMEBOL","Uruguay":"CONMEBOL",
    "Colombia":"CONMEBOL","Ecuador":"CONMEBOL","Paraguay":"CONMEBOL",
    "Mexico":"CONCACAF","United States":"CONCACAF","Canada":"CONCACAF",
    "Panama":"CONCACAF","Haiti":"CONCACAF","Curaçao":"CONCACAF",
    "Senegal":"CAF","Morocco":"CAF","Ghana":"CAF","Egypt":"CAF",
    "Algeria":"CAF","Tunisia":"CAF","Ivory Coast":"CAF","South Africa":"CAF",
    "DR Congo":"CAF","Cape Verde":"CAF",
    "Japan":"AFC","South Korea":"AFC","Australia":"AFC","Iran":"AFC",
    "Saudi Arabia":"AFC","Qatar":"AFC","Iraq":"AFC","Jordan":"AFC",
    "Uzbekistan":"AFC",
    "New Zealand":"OFC",
}

GROUPS = {
    "A": ["Mexico",        "South Africa",          "South Korea", "Czechia"],
    "B": ["Canada",        "Bosnia and Herzegovina", "Qatar",       "Switzerland"],
    "C": ["Brazil",        "Morocco",                "Haiti",       "Scotland"],
    "D": ["United States", "Paraguay",               "Australia",   "Türkiye"],
    "E": ["Germany",       "Curaçao",                "Ivory Coast", "Ecuador"],
    "F": ["Netherlands",   "Japan",                  "Sweden",      "Tunisia"],
    "G": ["Belgium",       "Egypt",                  "Iran",        "New Zealand"],
    "H": ["Spain",         "Cape Verde",             "Saudi Arabia","Uruguay"],
    "I": ["France",        "Senegal",                "Iraq",        "Norway"],
    "J": ["Argentina",     "Algeria",                "Austria",     "Jordan"],
    "K": ["Portugal",      "DR Congo",               "Uzbekistan",  "Colombia"],
    "L": ["England",       "Croatia",                "Ghana",       "Panama"],
}

# ── Real FIFA 2026 R32 bracket slots ─────────────────────────────────────────
# Each slot: (team_slot_1, team_slot_2)
# team slots are tuples: (group, position) where position 0=1st,1=2nd,2=best3rd
# best3rd slots specify which groups the third-place team must come from
# Format: ("1E") = group E winner, ("2C") = group C runner-up,
#         ("3ABCDF") = best third from groups A,B,C,D,F
R32_BRACKET = [
    # Left side — these 8 pairs feed into Left QF1 and Left QF2
    # Left QF1 (L1 winner vs L2 winner, L3 winner vs L4 winner)
    ("1E",  "3ABCDF"),   # L1
    ("1I",  "3CDFGH"),   # L2
    ("2A",  "2B"),        # L3
    ("1F",  "2C"),        # L4
    # Left QF2
    ("2K",  "2L"),        # L5
    ("1H",  "2J"),        # L6
    ("1D",  "3BEFIJ"),   # L7
    ("1G",  "3AEHIJ"),   # L8
    # Right side
    # Right QF1
    ("1C",  "2F"),        # R1
    ("2E",  "2I"),        # R2
    ("1A",  "3CEFHI"),   # R3
    ("1L",  "3EHIJK"),   # R4
    # Right QF2
    ("1J",  "2H"),        # R5
    ("2D",  "2G"),        # R6
    ("1B",  "3EFGIJ"),   # R7
    ("1K",  "3DEIJL"),   # R8
]

# R16: pairs of R32 slot indices (0-indexed)
R16_PAIRS = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]
# QF: pairs of R16 slot indices
QF_PAIRS  = [(0,1),(2,3),(4,5),(6,7)]
# SF: pairs of QF slot indices
SF_PAIRS  = [(0,1),(2,3)]

ROUNDS       = ["r32","r16","qf","sf","final","winner"]
FED_COLS     = ["AFC","CAF","CONCACAF","CONMEBOL","OFC","OTHER","UEFA"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. TEAM FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_team_features(results_path):
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    ELO_K, ELO_DEFAULT = 32, 1500
    ratings = {}
    elo_history = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        r_h  = ratings.get(h, ELO_DEFAULT)
        r_a  = ratings.get(a, ELO_DEFAULT)
        elo_history.append({"date": row["date"], "team": h, "elo": r_h})
        elo_history.append({"date": row["date"], "team": a, "elo": r_a})
        if   row["home_score"] > row["away_score"]:  s_h, s_a = 1.0, 0.0
        elif row["home_score"] == row["away_score"]: s_h, s_a = 0.5, 0.5
        else:                                         s_h, s_a = 0.0, 1.0
        e_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
        ratings[h] = r_h + ELO_K * (s_h - e_h)
        ratings[a] = r_a + ELO_K * (s_a - (1 - e_h))

    elo_ts = pd.DataFrame(elo_history)
    latest_date  = df["date"].max()
    start_legacy = latest_date - pd.DateOffset(years=20)

    legacy = {}
    for team, grp in elo_ts.groupby("team"):
        mask = (grp["date"] >= start_legacy) & (grp["date"] < latest_date)
        vals = grp.loc[mask, "elo"]
        legacy[team] = vals.mean() if len(vals) > 0 else ELO_DEFAULT

    home_df = df[["date","home_team","home_score","away_score"]].copy()
    home_df.columns = ["date","team","gf","ga"]
    away_df = df[["date","away_team","away_score","home_score"]].copy()
    away_df.columns = ["date","team","gf","ga"]
    long = pd.concat([home_df, away_df], ignore_index=True)
    long["win"] = (long["gf"] > long["ga"]).astype(float)
    long["gd"]  = long["gf"] - long["ga"]
    long = long.sort_values(["team","date"]).reset_index(drop=True)
    long["win_rate"] = (long.groupby("team")["win"]
                            .transform(lambda x: x.shift(1).rolling(10,min_periods=1).mean()))
    long["gd_pg"]    = (long.groupby("team")["gd"]
                            .transform(lambda x: x.shift(1).rolling(10,min_periods=1).mean()))
    latest_form = long.groupby("team").last()[["win_rate","gd_pg"]]

    all_wc_teams = [t for g in GROUPS.values() for t in g]
    team_features = {}
    for team in all_wc_teams:
        lookup = NAME_MAP.get(team, team)
        elo    = ratings.get(lookup, ELO_DEFAULT)
        leg    = legacy.get(lookup, ELO_DEFAULT)
        fed    = FEDERATION_MAP.get(team, "OTHER")
        wr     = latest_form.loc[lookup,"win_rate"] if lookup in latest_form.index else 0.5
        gd     = latest_form.loc[lookup,"gd_pg"]    if lookup in latest_form.index else 0.0
        team_features[team] = {"elo":elo,"legacy":leg,"win_rate":wr,"gd_pg":gd,"federation":fed}

    return team_features

# ══════════════════════════════════════════════════════════════════════════════
# 2. PRE-COMPUTE SYMMETRIC PAIRWISE PROBABILITIES
# ══════════════════════════════════════════════════════════════════════════════

def precompute_probs(model, team_stats, feat_cols):
    all_teams  = [t for g in GROUPS.values() for t in g]
    raw_cache  = {}

    for home in all_teams:
        for away in all_teams:
            if home == away: continue
            h, a = team_stats[home], team_stats[away]
            row = {
                "home_elo":      h["elo"],     "away_elo":    a["elo"],
                "elo_diff":      h["elo"]-a["elo"],
                "home_legacy":   h["legacy"],  "away_legacy": a["legacy"],
                "legacy_diff":   h["legacy"]-a["legacy"],
                "home_win_rate": h["win_rate"],"away_win_rate":a["win_rate"],
                "home_gd_pg":    h["gd_pg"],   "away_gd_pg":  a["gd_pg"],
            }
            for fed in FED_COLS:
                row[f"home_federation_{fed}"] = int(h["federation"]==fed)
                row[f"away_federation_{fed}"] = int(a["federation"]==fed)
            X     = pd.DataFrame([row])[feat_cols]
            probs = model.predict_proba(X)[0]
            raw_cache[(home,away)] = (probs[2], probs[1], probs[0])

    # Symmetrise: average both orderings
    prob_cache = {}
    for home in all_teams:
        for away in all_teams:
            if home == away: continue
            if (away,home) in prob_cache:
                p_a, p_d, p_h = prob_cache[(away,home)]
                prob_cache[(home,away)] = (p_h, p_d, p_a)
            else:
                p_h1,p_d1,p_a1 = raw_cache[(home,away)]
                p_h2,p_d2,p_a2 = raw_cache[(away,home)]
                prob_cache[(home,away)] = (
                    (p_h1+p_a2)/2,
                    (p_d1+p_d2)/2,
                    (p_a1+p_h2)/2,
                )
    return prob_cache

# ══════════════════════════════════════════════════════════════════════════════
# 3. GROUP STAGE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_group(teams, prob_cache):
    pts = defaultdict(int)
    gd  = defaultdict(int)
    gf  = defaultdict(int)
    pairs = [(teams[i],teams[j]) for i in range(4) for j in range(i+1,4)]
    for home,away in pairs:
        p_h,p_d,p_a = prob_cache[(home,away)]
        r = np.random.random()
        if r < p_h:
            hg,ag = np.random.randint(1,4), np.random.randint(0,2)
            pts[home] += 3
        elif r < p_h+p_d:
            hg = ag = np.random.randint(0,3)
            pts[home] += 1; pts[away] += 1
        else:
            hg,ag = np.random.randint(0,2), np.random.randint(1,4)
            pts[away] += 3
        gd[home]+=hg-ag; gd[away]+=ag-hg
        gf[home]+=hg;    gf[away]+=ag

    ranked = sorted(teams, key=lambda t:(pts[t],gd[t],gf[t]), reverse=True)
    return ranked, {t: pts[t] for t in teams}, {t: gd[t] for t in teams}, {t: gf[t] for t in teams}

# ══════════════════════════════════════════════════════════════════════════════
# 4. RESOLVE BRACKET SLOT → ACTUAL TEAM
# ══════════════════════════════════════════════════════════════════════════════

def resolve_slot(slot_str, group_results, thirds_ranked):
    """
    slot_str examples: '1E', '2C', '3ABCDF'
    group_results: {gname: [1st,2nd,3rd,4th]}
    thirds_ranked: list of (team, pts, gd, gf) sorted best→worst
    """
    if slot_str.startswith("1"):
        g = slot_str[1]
        return group_results[g][0]
    elif slot_str.startswith("2"):
        g = slot_str[1]
        return group_results[g][1]
    else:  # best third from allowed groups
        allowed = set(slot_str[1:])
        for team, _, _, _ in thirds_ranked:
            grp = next(g for g,ranked in group_results.items() if ranked[2]==team)
            if grp in allowed:
                return team
        return thirds_ranked[0][0]  # fallback

# ══════════════════════════════════════════════════════════════════════════════
# 5. KNOCKOUT MATCH (no draw — redistribute draw prob)
# ══════════════════════════════════════════════════════════════════════════════

def ko_match(t1, t2, prob_cache):
    p_h, p_d, _ = prob_cache[(t1,t2)]
    return t1 if np.random.random() < p_h + p_d/2 else t2

# ══════════════════════════════════════════════════════════════════════════════
# 6. SIMULATE ONE FULL TOURNAMENT WITH REAL BRACKET
# ══════════════════════════════════════════════════════════════════════════════

def simulate_once(prob_cache):
    all_teams = [t for g in GROUPS.values() for t in g]
    progress  = {t: "group" for t in all_teams}

    # ── Group stage ───────────────────────────────────────────────────────────
    group_results = {}
    all_pts, all_gd, all_gf = {}, {}, {}

    for gname, teams in GROUPS.items():
        ranked, pts, gd, gf = simulate_group(teams, prob_cache)
        group_results[gname] = ranked
        for t in teams:
            all_pts[t]=pts[t]; all_gd[t]=gd[t]; all_gf[t]=gf[t]

    # Rank all 12 third-place teams by pts, gd, gf
    thirds = [group_results[g][2] for g in GROUPS]
    thirds_ranked = sorted(
        thirds,
        key=lambda t: (all_pts[t], all_gd[t], all_gf[t]),
        reverse=True
    )
    thirds_with_stats = [(t, all_pts[t], all_gd[t], all_gf[t]) for t in thirds_ranked]

    # Pre-assign all third-place slots upfront to avoid duplicate assignments
    # Each third-place slot requires the best available third from allowed groups
    # We resolve all 8 third-place slots in bracket order, marking each used
    used_thirds = set()
    third_slot_assignments = {}
    for s1, s2 in R32_BRACKET:
        for slot_str in (s1, s2):
            if slot_str.startswith("3"):
                allowed = set(slot_str[1:])
                assigned = None
                for t, _, _, _ in thirds_with_stats:
                    grp = next(g for g,ranked in group_results.items() if ranked[2]==t)
                    if grp in allowed and t not in used_thirds:
                        used_thirds.add(t)
                        assigned = t
                        break
                if assigned is None:
                    # fallback: any unused third
                    for t, _, _, _ in thirds_with_stats:
                        if t not in used_thirds:
                            used_thirds.add(t)
                            assigned = t
                            break
                third_slot_assignments[slot_str] = assigned

    def get_slot_team(slot_str):
        if slot_str.startswith("1"):
            return group_results[slot_str[1]][0]
        elif slot_str.startswith("2"):
            return group_results[slot_str[1]][1]
        else:
            return third_slot_assignments[slot_str]

    # ── Round of 32 ───────────────────────────────────────────────────────────
    r32_teams = []
    for s1, s2 in R32_BRACKET:
        t1 = get_slot_team(s1)
        t2 = get_slot_team(s2)
        # Mark both teams as having qualified from the group stage
        progress[t1] = "qualified"
        progress[t2] = "qualified"
        winner = ko_match(t1, t2, prob_cache)
        progress[winner] = "r32"
        r32_teams.append(winner)

    # ── Round of 16 ───────────────────────────────────────────────────────────
    r16_teams = []
    for i1, i2 in R16_PAIRS:
        winner = ko_match(r32_teams[i1], r32_teams[i2], prob_cache)
        progress[winner] = "r16"
        r16_teams.append(winner)

    # ── Quarter-finals ────────────────────────────────────────────────────────
    qf_teams = []
    for i1, i2 in QF_PAIRS:
        winner = ko_match(r16_teams[i1], r16_teams[i2], prob_cache)
        progress[winner] = "qf"
        qf_teams.append(winner)

    # ── Semi-finals ───────────────────────────────────────────────────────────
    sf_teams = []
    for i1, i2 in SF_PAIRS:
        winner = ko_match(qf_teams[i1], qf_teams[i2], prob_cache)
        progress[winner] = "sf"
        sf_teams.append(winner)

    # ── Final ─────────────────────────────────────────────────────────────────
    if len(sf_teams) >= 2:
        champion = ko_match(sf_teams[0], sf_teams[1], prob_cache)
        progress[champion] = "winner"

    return progress, all_pts

# ══════════════════════════════════════════════════════════════════════════════
# 7. RUN SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def reached(progress_val, target_round):
    order = ["group", "qualified"] + ROUNDS
    return order.index(progress_val) >= order.index(target_round)

def run_simulations(n, prob_cache):
    all_teams = [t for g in GROUPS.values() for t in g]
    counts    = {t: defaultdict(int) for t in all_teams}
    pts_total = {t: 0 for t in all_teams}

    print(f"Running {n:,} simulations ", end="", flush=True)
    for i in range(n):
        if i % 1000 == 0: print(".", end="", flush=True)
        result, group_pts = simulate_once(prob_cache)
        for team, prog in result.items():
            # Count group qualification separately (binary: did they make R32?)
            if prog != "group":
                counts[team]["qualified"] += 1
            # Count knockout round progression
            for rnd in ROUNDS:
                if reached(prog, rnd):
                    counts[team][rnd] += 1
        for team, pts in group_pts.items():
            pts_total[team] += pts
    print(" done.\n")
    avg_pts = {t: pts_total[t]/n for t in all_teams}
    return counts, avg_pts

# ══════════════════════════════════════════════════════════════════════════════
# 8. PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def print_results(counts, avg_pts, n, team_stats):
    all_teams = sorted(counts.keys(), key=lambda t: counts[t]["winner"], reverse=True)

    print("=" * 78)
    print(f"  2026 WORLD CUP SIMULATION RESULTS  ({n:,} simulations)")
    print(f"  (Real FIFA 2026 bracket structure)")
    print("=" * 78)
    print(f"  {'Team':<28} {'R32':>5} {'R16':>5} {'QF':>5} "
          f"{'SF':>5} {'Final':>6} {'Win%':>6}  {'Elo':>6}")
    print(f"  {'─'*28} {'─'*5} {'─'*5} {'─'*5} "
          f"{'─'*5} {'─'*6} {'─'*6}  {'─'*6}")

    for team in all_teams:
        c   = counts[team]
        elo = team_stats[team]["elo"]
        def pct(r): return f"{100*c[r]/n:.1f}%"
        print(f"  {team:<28} {pct('r32'):>5} {pct('r16'):>5} {pct('qf'):>5} "
              f"{pct('sf'):>5} {pct('final'):>6} {pct('winner'):>6}  {elo:>6.0f}")

    print()
    print("  TOP 10 MOST LIKELY WINNERS")
    print(f"  {'─'*50}")
    for i, team in enumerate(all_teams[:10], 1):
        pct = 100 * counts[team]["winner"] / n
        bar = "█" * int(pct * 2)
        print(f"  {i:>2}. {team:<26} {pct:>5.1f}%  {bar}")

    print()
    print("  GROUP STAGE — QUALIFICATION % AND AVERAGE POINTS")
    print(f"  {'─'*65}")
    for gname, teams in GROUPS.items():
        print(f"\n  Group {gname}:")
        print(f"    {'Team':<28} {'Qual%':>6}  {'Avg pts':>8}  Chart")
        print(f"    {'─'*28} {'─'*6}  {'─'*8}  {'─'*20}")
        ranked = sorted(teams, key=lambda t: avg_pts[t], reverse=True)
        for team in ranked:
            # Qualification probability is P(advance from group) — capped at 100%
            # Teams can appear in R32 as 1st, 2nd, or best-third
            # We show P(advance) directly from simulation counts
            qual_pct = 100 * counts[team]["qualified"] / n
            pts      = avg_pts[team]
            bar      = "█" * int(min(qual_pct, 100) / 5)
            print(f"    {team:<28} {qual_pct:>5.1f}%  {pts:>8.2f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== Loading model ===")
    with open(MODELS / "logistic_model.pkl", "rb") as f: model     = pickle.load(f)
    with open(MODELS / "feature_cols.pkl",   "rb") as f: feat_cols = pickle.load(f)

    print("=== Computing team features ===")
    cache_path = ROOT / "models" / "team_stats_cache.pkl"
    prob_path  = ROOT / "models" / "prob_cache.pkl"

    # Recompute if model or results.csv is newer than cache
    results_mtime = (ROOT / "data" / "raw" / "results.csv").stat().st_mtime
    model_mtime   = (MODELS / "logistic_model.pkl").stat().st_mtime
    cache_mtime   = cache_path.stat().st_mtime if cache_path.exists() else 0
    cache_valid   = cache_mtime > max(results_mtime, model_mtime)

    if cache_valid:
        print("  Loading cached team stats and probabilities...")
        with open(cache_path, "rb") as f: team_stats = pickle.load(f)
        with open(prob_path,  "rb") as f: prob_cache = pickle.load(f)
        print("  Done (from cache)")
    else:
        print("  Cache stale or missing — recomputing...")
        team_stats = compute_team_features(ROOT / "data" / "raw" / "results.csv")
        print("=== Pre-computing match probabilities ===")
        prob_cache = precompute_probs(model, team_stats, feat_cols)
        with open(cache_path, "wb") as f: pickle.dump(team_stats, f)
        with open(prob_path,  "wb") as f: pickle.dump(prob_cache,  f)
        print("  Cached for next run")

    print("=== Running simulations ===")
    counts, avg_pts = run_simulations(100000, prob_cache)

    print_results(counts, avg_pts, 100000, team_stats)

if __name__ == "__main__":
    main()
