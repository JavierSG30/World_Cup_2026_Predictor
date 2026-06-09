"""
World Cup 2026 — Matchup Analyser
-----------------------------------
Edit TEAM_1 and TEAM_2 below, then run:
    python -m src.matchup

Probabilities are averaged over both team orderings to ensure
full symmetry at neutral venues.
"""

import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from predict import compute_team_features, FED_COLS

ROOT   = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def build_row(home, away, team_stats, feat_cols):
    h, a = team_stats[home], team_stats[away]
    row = {
        "home_elo":      h["elo"],
        "away_elo":      a["elo"],
        "elo_diff":      h["elo"]    - a["elo"],
        "home_legacy":   h["legacy"],
        "away_legacy":   a["legacy"],
        "legacy_diff":   h["legacy"] - a["legacy"],
        "home_win_rate": h["win_rate"],
        "away_win_rate": a["win_rate"],
        "home_gd_pg":    h["gd_pg"],
        "away_gd_pg":    a["gd_pg"],
    }
    for fed in FED_COLS:
        row[f"home_federation_{fed}"] = int(h["federation"] == fed)
        row[f"away_federation_{fed}"] = int(a["federation"] == fed)
    return pd.DataFrame([row])[feat_cols]


def neutral_probs(team1, team2, model, team_stats, feat_cols):
    """
    Average predictions over both orderings to remove home/away asymmetry.
    Returns (p_team1, p_draw, p_team2) — fully symmetric.
    """
    # Order 1: team1 in home slot
    p1 = model.predict_proba(build_row(team1, team2, team_stats, feat_cols))[0]
    p1_wins_as_home, p_draw1, p2_wins_as_away = p1[2], p1[1], p1[0]

    # Order 2: team2 in home slot
    p2 = model.predict_proba(build_row(team2, team1, team_stats, feat_cols))[0]
    p2_wins_as_home, p_draw2, p1_wins_as_away = p2[2], p2[1], p2[0]

    p_team1 = (p1_wins_as_home + p1_wins_as_away) / 2
    p_team2 = (p2_wins_as_home + p2_wins_as_away) / 2
    p_draw  = (p_draw1 + p_draw2) / 2

    return p_team1, p_draw, p_team2


def analyse_matchup(team1, team2, model, feat_cols, team_stats):
    h = team_stats[team1]
    a = team_stats[team2]

    p_t1, p_draw, p_t2 = neutral_probs(team1, team2, model, team_stats, feat_cols)

    # Unwrap Pipeline if logistic regression, otherwise use model directly
    if hasattr(model, "named_steps"):
        scaler = model.named_steps["scaler"]
        clf    = model.named_steps["clf"]
    else:
        scaler = None
        clf    = model

    # Scale inputs if scaler exists (logistic regression only)
    if scaler is not None:
        X1_scaled = scaler.transform(build_row(team1, team2, team_stats, feat_cols))
        X2_scaled = scaler.transform(build_row(team2, team1, team_stats, feat_cols))
    else:
        X1_scaled = build_row(team1, team2, team_stats, feat_cols).values
        X2_scaled = build_row(team2, team1, team_stats, feat_cols).values

    W = 70
    print()
    print("=" * W)
    print(f"  MATCHUP ANALYSIS: {team1}  vs  {team2}")
    print(f"  (Neutral venue — probabilities averaged over both orderings)")
    print("=" * W)

    def prob_bar(p, char, width=30):
        return char * int(p * width) + "░" * (width - int(p * width))

    print(f"\n  PREDICTED PROBABILITIES")
    print(f"  {'─'*45}")
    print(f"  {team1:<22} Win  {p_t1*100:>5.1f}%  {prob_bar(p_t1, '█')}")
    print(f"  {'Draw':<22}      {p_draw*100:>5.1f}%  {prob_bar(p_draw, '▒')}")
    print(f"  {team2:<22} Win  {p_t2*100:>5.1f}%  {prob_bar(p_t2, '░')}")

    print(f"\n  TEAM FEATURES SIDE BY SIDE")
    print(f"  {'─'*62}")
    print(f"  {'Feature':<20} {team1:>20}  {'vs':^5}  {team2:<20}")
    print(f"  {'─'*20} {'─'*20}  {'─'*5}  {'─'*20}")

    def row_str(label, hval, aval, fmt=".1f"):
        hm = " ◀" if hval > aval else ""
        am = " ◀" if aval > hval else ""
        print(f"  {label:<20} {format(hval,fmt)+hm:>22}    {format(aval,fmt)+am:<22}")

    row_str("Elo rating",        h["elo"],      a["elo"])
    row_str("Legacy Elo",        h["legacy"],   a["legacy"])
    row_str("Win rate (L10)",    h["win_rate"], a["win_rate"], fmt=".2f")
    row_str("Goal diff/g (L10)", h["gd_pg"],    a["gd_pg"],   fmt=".2f")
    print(f"  {'Federation':<20} {h['federation']:>22}    {a['federation']:<22}")

    print(f"\n  FEATURE CONTRIBUTIONS")
    print(f"  {'─'*62}")

    # Feature contributions only available for logistic regression (has coef_)
    if hasattr(clf, "coef_"):
        coef_home = clf.coef_[2]
        coef_away = clf.coef_[0]

        print(f"  (positive = favours {team1}, negative = favours {team2})")
        print(f"  {'Feature':<30} {'Net effect':>12}  Direction")
        print(f"  {'─'*30} {'─'*12}  {'─'*25}")

        contributions = []
        for i, fname in enumerate(feat_cols):
            # In ordering 1: team1 is home
            net1 = coef_home[i] * X1_scaled[0][i] - coef_away[i] * X1_scaled[0][i]
            # In ordering 2: team1 is away — flip sign to keep "favours team1" consistent
            net2 = -(coef_home[i] * X2_scaled[0][i] - coef_away[i] * X2_scaled[0][i])
            contributions.append((fname, (net1 + net2) / 2))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        for fname, net in contributions[:12]:
            if abs(net) < 0.001:
                continue
            direction = f"favours {team1}" if net > 0 else f"favours {team2}"
            bar = ("█" if net > 0 else "░") * min(int(abs(net) * 15), 25)
            print(f"  {fname:<30} {net:>+12.4f}  {direction}  {bar}")

        print(f"\n  SUMMARY")
        print(f"  {'─'*62}")
        t1_adv = [f for f, n in contributions if n > 0.05]
        t2_adv = [f for f, n in contributions if n < -0.05]
        print(f"  Favours {team1:<20}: {', '.join(t1_adv[:4]) if t1_adv else 'none'}")
        print(f"  Favours {team2:<20}: {', '.join(t2_adv[:4]) if t2_adv else 'none'}")

    else:
        # Random Forest / Gradient Boosting — show feature importances instead
        print(f"  (Feature contributions not available for tree-based models)")
        if hasattr(clf, "feature_importances_"):
            importances = list(zip(feat_cols, clf.feature_importances_))
            importances.sort(key=lambda x: x[1], reverse=True)
            print(f"  {'Feature':<30} {'Importance':>12}")
            print(f"  {'─'*30} {'─'*12}")
            for fname, imp in importances[:12]:
                bar = "█" * min(int(imp * 100), 25)
                print(f"  {fname:<30} {imp:>12.4f}  {bar}")

    if p_t1 > p_t2 and p_t1 > p_draw:
        winner, pct = team1, p_t1
    elif p_t2 > p_t1 and p_t2 > p_draw:
        winner, pct = team2, p_t2
    else:
        winner, pct = "Draw", p_draw
    print(f"\n  Most likely outcome: {winner} ({pct*100:.1f}%)")
    print("=" * W)


def main():
    # ── EDIT THESE TWO LINES TO CHANGE THE MATCHUP ───────────────────────────
    TEAM_1 = "Spain"
    TEAM_2 = "Argentina"
    # ─────────────────────────────────────────────────────────────────────────

    with open(MODELS / "logistic_model.pkl", "rb") as f: model     = pickle.load(f)
    with open(MODELS / "feature_cols.pkl",   "rb") as f: feat_cols = pickle.load(f)
    cache_path = ROOT / "models" / "team_stats_cache.pkl"
    prob_path  = ROOT / "models" / "prob_cache.pkl"
    results_mtime = (ROOT / "data" / "raw" / "results.csv").stat().st_mtime
    model_mtime   = (MODELS / "logistic_model.pkl").stat().st_mtime
    cache_mtime   = cache_path.stat().st_mtime if cache_path.exists() else 0
    cache_valid   = cache_mtime > max(results_mtime, model_mtime)

    if cache_valid:
        print("  Loading cached team stats...")
        with open(cache_path, "rb") as f: team_stats = pickle.load(f)
    else:
        print("  Recomputing team stats (run predict.py first to cache)...")
        team_stats = compute_team_features(ROOT / "data" / "raw" / "results.csv")

    for team in [TEAM_1, TEAM_2]:
        if team not in team_stats:
            print(f"Team not found: '{team}'")
            print("Available teams: " + ", ".join(sorted(team_stats.keys())))
            sys.exit(1)

    analyse_matchup(TEAM_1, TEAM_2, model, feat_cols, team_stats)


if __name__ == "__main__":
    main()