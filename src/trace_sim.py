"""
World Cup 2026 — Single Simulation Trace
-----------------------------------------
Runs exactly ONE simulation and prints every decision made,
so you can verify the logic is correct step by step.

Run:
    python -m src.trace_sim
"""

import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from predict import (
    compute_team_features, GROUPS, precompute_probs,
    simulate_group, ko_match, R32_BRACKET, R16_PAIRS, QF_PAIRS, SF_PAIRS
)

ROOT   = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

np.random.seed(99)  # fixed seed so result is reproducible

def separator(title=""):
    if title:
        print(f"\n{'═'*65}")
        print(f"  {title}")
        print(f"{'═'*65}")
    else:
        print(f"  {'─'*60}")

def ko(t1, t2, prob_cache, label=""):
    p_h, p_d, p_a = prob_cache[(t1, t2)]
    p_t1_wins = p_h + p_d / 2   # draw redistributed 50/50
    roll = np.random.random()
    winner = t1 if roll < p_t1_wins else t2
    loser  = t2 if winner == t1 else t1
    print(f"  {t1:<26} vs {t2:<26}")
    print(f"    P({t1} wins) = {p_t1_wins*100:.1f}%  |  "
          f"P({t2} wins) = {(1-p_t1_wins)*100:.1f}%")
    print(f"    Dice roll: {roll:.4f}  →  {'✓' if roll < p_t1_wins else '✗'} {t1} / "
          f"{'✓' if roll >= p_t1_wins else '✗'} {t2}")
    print(f"    ▶ Winner: {winner}")
    return winner

def main():
    print("Loading model and computing probabilities...")
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
    prob_cache = precompute_probs(model, team_stats, feat_cols)
    print("Ready. Running one simulation with seed=99.\n")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 — GROUP STAGE
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 1 — GROUP STAGE")
    print("  Each group plays a round-robin (every team vs every other once).")
    print("  Points: Win=3, Draw=1, Loss=0.")
    print("  Tiebreaker: Points → Goal Difference → Goals Scored.\n")

    group_results = {}
    all_pts, all_gd, all_gf = {}, {}, {}

    for gname, teams in GROUPS.items():
        ranked, pts, gd, gf = simulate_group(teams, prob_cache)
        group_results[gname] = ranked
        for t in teams:
            all_pts[t] = pts[t]
            all_gd[t]  = gd[t]
            all_gf[t]  = gf[t]

        print(f"  Group {gname}  (6 matches played)")
        print(f"    {'Pos':<4} {'Team':<28} {'Pts':>4} {'GD':>5} {'GF':>5}")
        print(f"    {'─'*4} {'─'*28} {'─'*4} {'─'*5} {'─'*5}")
        for pos, team in enumerate(ranked, 1):
            marker = " ← advances (top 2)" if pos <= 2 else \
                     " ← 3rd place (best-third pool)" if pos == 3 else ""
            print(f"    {pos:<4} {team:<28} {pts[team]:>4} "
                  f"{gd[team]:>+5} {gf[team]:>5}{marker}")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2 — RANK THE 12 THIRD-PLACE TEAMS
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 2 — RANKING THE 12 THIRD-PLACE TEAMS")
    print("  The 3rd-place team from each of the 12 groups enters a pool.")
    print("  They are ranked by: Points → Goal Difference → Goals Scored.")
    print("  The best 8 advance to the Round of 32.\n")

    thirds = [group_results[g][2] for g in GROUPS]
    thirds_ranked = sorted(
        thirds,
        key=lambda t: (all_pts[t], all_gd[t], all_gf[t]),
        reverse=True
    )

    print(f"  {'Rank':<5} {'Team':<28} {'From':>6} {'Pts':>4} {'GD':>5} {'GF':>5}")
    print(f"  {'─'*5} {'─'*28} {'─'*6} {'─'*4} {'─'*5} {'─'*5}")
    for rank, team in enumerate(thirds_ranked, 1):
        grp = next(g for g, ranked in group_results.items() if ranked[2] == team)
        advances = "✓ ADVANCES" if rank <= 8 else "✗ eliminated"
        print(f"  {rank:<5} {team:<28} {'Grp '+grp:>6} "
              f"{all_pts[team]:>4} {all_gd[team]:>+5} {all_gf[team]:>5}"
              f"   {advances}")

    thirds_with_stats = [(t, all_pts[t], all_gd[t], all_gf[t])
                         for t in thirds_ranked]

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3 — ASSIGN THIRD-PLACE TEAMS TO BRACKET SLOTS
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 3 — ASSIGNING THIRD-PLACE TEAMS TO BRACKET SLOTS")
    print("  Each R32 third-place slot specifies which groups are allowed.")
    print("  We assign the best available third from the allowed groups.\n")

    used_thirds = set()
    third_slot_assignments = {}

    for s1, s2 in R32_BRACKET:
        for slot_str in (s1, s2):
            if slot_str.startswith("3"):
                allowed = set(slot_str[1:])
                for t, _, _, _ in thirds_with_stats:
                    grp = next(g for g, ranked in group_results.items()
                               if ranked[2] == t)
                    if grp in allowed and t not in used_thirds:
                        used_thirds.add(t)
                        third_slot_assignments[slot_str] = t
                        print(f"  Slot {slot_str:<10} (allowed groups: "
                              f"{', '.join(sorted(allowed)):<12}) "
                              f"→ assigned: {t} (from Group {grp}, "
                              f"{all_pts[t]}pts {all_gd[t]:+}gd)")
                        break

    def get_slot_team(slot_str):
        if slot_str.startswith("1"):
            return group_results[slot_str[1]][0]
        elif slot_str.startswith("2"):
            return group_results[slot_str[1]][1]
        else:
            return third_slot_assignments[slot_str]

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4 — ROUND OF 32
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 4 — ROUND OF 32  (32 teams → 16)")
    print("  No draws. Draw probability redistributed 50/50 to each team.\n")

    r32_teams = []
    for i, (s1, s2) in enumerate(R32_BRACKET, 1):
        t1 = get_slot_team(s1)
        t2 = get_slot_team(s2)
        print(f"  Match {i:>2}  [{s1} vs {s2}]")
        winner = ko(t1, t2, prob_cache)
        r32_teams.append(winner)
        print()

    print(f"  R32 winners: {', '.join(r32_teams)}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5 — ROUND OF 16
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 5 — ROUND OF 16  (16 teams → 8)")
    print("  Bracket pairs are fixed by FIFA draw.\n")

    r16_teams = []
    for i, (i1, i2) in enumerate(R16_PAIRS, 1):
        t1, t2 = r32_teams[i1], r32_teams[i2]
        print(f"  Match {i:>2}")
        winner = ko(t1, t2, prob_cache)
        r16_teams.append(winner)
        print()

    print(f"  R16 winners: {', '.join(r16_teams)}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6 — QUARTER-FINALS
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 6 — QUARTER-FINALS  (8 teams → 4)")
    print()

    qf_teams = []
    for i, (i1, i2) in enumerate(QF_PAIRS, 1):
        t1, t2 = r16_teams[i1], r16_teams[i2]
        print(f"  QF {i}")
        winner = ko(t1, t2, prob_cache)
        qf_teams.append(winner)
        print()

    print(f"  QF winners: {', '.join(qf_teams)}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7 — SEMI-FINALS
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 7 — SEMI-FINALS  (4 teams → 2)")
    print()

    sf_teams = []
    for i, (i1, i2) in enumerate(SF_PAIRS, 1):
        t1, t2 = qf_teams[i1], qf_teams[i2]
        print(f"  SF {i}")
        winner = ko(t1, t2, prob_cache)
        sf_teams.append(winner)
        print()

    print(f"  Finalists: {sf_teams[0]}  vs  {sf_teams[1]}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8 — FINAL
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 8 — THE FINAL")
    print()

    champion = ko(sf_teams[0], sf_teams[1], prob_cache)

    separator()
    print(f"\n  🏆  WORLD CHAMPION (this simulation): {champion}\n")

if __name__ == "__main__":
    main()
