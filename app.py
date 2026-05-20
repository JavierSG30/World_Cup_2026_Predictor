"""
World Cup 2026 Predictor — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
MODELS = ROOT / "models"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #1a1a2e;
        margin-bottom: 8px;
    }
    .team-rank {
        font-size: 2rem;
        font-weight: 300;
        color: #aaa;
        font-family: 'DM Serif Display', serif;
    }
    .win-pct {
        font-size: 1.6rem;
        font-weight: 500;
        color: #1a1a2e;
    }
    .federation-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        background: #e8e8e8;
        color: #444;
    }
    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #16213e;
    }
    .sidebar-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_results():
    with open(ROOT / "simulation_results.json") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    """Load precomputed data from JSON — no sklearn dependency needed."""
    with open(ROOT / "prob_cache.json") as f:
        raw = json.load(f)
    prob_cache = {(k.split("||")[0], k.split("||")[1]): tuple(v) 
                  for k, v in raw.items()}
    with open(ROOT / "team_stats.json") as f:
        team_stats = json.load(f)
    return None, None, prob_cache, team_stats

def results_to_df(data):
    rows = []
    for team, stats in data["teams"].items():
        rows.append({"Team": team, **stats})
    return pd.DataFrame(rows).sort_values("winner", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-title">⚽ WC 2026 Predictor</div>', unsafe_allow_html=True)
    st.caption("Monte Carlo simulation · 10,000 runs · Random Forest")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏆 Overview", "👥 Group Stage", "⚔️ Matchup Analyser",
         "🎲 Run a Simulation", "📊 Methodology"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Built by Javier Suarez Grandal")
    st.caption("Model: Random Forest · Features: Elo, Form, Legacy, Federation")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏆 Overview":
    data = load_results()
    df   = results_to_df(data)

    st.title("2026 FIFA World Cup Predictor")
    st.markdown(f"Based on **{data['n_sims']:,} Monte Carlo simulations** using a Random Forest model trained on international football results from 2000–2025.")

    st.divider()

    # ── Top 10 bar chart ──────────────────────────────────────────────────────
    st.subheader("Most Likely Winners")
    top10 = df.head(10)

    chart_data = top10[["Team", "winner"]].set_index("Team")
    chart_data.columns = ["Win Probability (%)"]
    st.bar_chart(chart_data, color="#1a1a2e")

    # ── Full table ────────────────────────────────────────────────────────────
    st.subheader("All 48 Teams")
    st.caption("Probabilities show % of simulations where each team reached that stage")

    display_df = df[["Team", "group", "federation", "elo",
                      "qualified", "r32", "r16", "qf", "sf", "final", "winner"]].copy()
    display_df.columns = ["Team", "Group", "Federation", "Elo",
                           "Qualify %", "R32 %", "R16 %", "QF %", "SF %", "Final %", "Win %"]
    display_df.index = range(1, len(display_df) + 1)

    st.dataframe(
        display_df.style.format({"Elo": "{:.0f}",
                                 "Qualify %": "{:.1f}%", "R32 %": "{:.1f}%",
                                 "R16 %": "{:.1f}%",     "QF %": "{:.1f}%",
                                 "SF %": "{:.1f}%",      "Final %": "{:.1f}%",
                                 "Win %": "{:.1f}%"}),
        use_container_width=True,
        height=600,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GROUP STAGE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👥 Group Stage":
    data = load_results()
    df   = results_to_df(data)

    st.title("Group Stage")
    st.markdown("Select a group to see qualification probabilities and expected points.")

    groups = data["groups"]
    selected_group = st.selectbox(
        "Select Group",
        [f"Group {g}" for g in sorted(groups.keys())],
    )
    gname  = selected_group.split(" ")[1]
    teams  = groups[gname]

    st.divider()

    group_df = df[df["Team"].isin(teams)].copy()
    group_df = group_df.sort_values("avg_pts", ascending=False)

    # ── Summary cards ─────────────────────────────────────────────────────────
    cols = st.columns(4)
    for i, (_, row) in enumerate(group_df.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.8rem;color:#888;margin-bottom:4px">
                    #{i+1} most likely
                </div>
                <div style="font-size:1.1rem;font-weight:600">{row['Team']}</div>
                <div style="font-size:0.85rem;color:#555;margin-top:4px">
                    {row['avg_pts']:.2f} avg pts
                </div>
                <div style="font-size:1.4rem;font-weight:500;color:#1a1a2e">
                    {row['qualified']:.1f}%
                </div>
                <div style="font-size:0.75rem;color:#888">qualify chance</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Detailed table ────────────────────────────────────────────────────────
    st.subheader(f"Group {gname} — Detailed Stats")

    detail_df = group_df[["Team", "elo", "avg_pts", "qualified",
                            "win_rate", "gd_pg", "federation"]].copy()
    detail_df.columns = ["Team", "Elo", "Avg Points", "Qualify %",
                          "Form (Win Rate)", "Form (GD/g)", "Federation"]
    detail_df = detail_df.reset_index(drop=True)
    detail_df.index = range(1, 5)

    st.dataframe(
        detail_df.style
            .format({"Elo": "{:.0f}", "Avg Points": "{:.2f}",
                     "Qualify %": "{:.1f}%", "Form (Win Rate)": "{:.2f}",
                     "Form (GD/g)": "{:+.2f}"}),
        use_container_width=True,
    )

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.subheader("Qualification Probability")
    chart = group_df.set_index("Team")[["qualified"]]
    chart.columns = ["Qualification %"]
    st.bar_chart(chart, color="#2d6a4f")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MATCHUP ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚔️ Matchup Analyser":
    data = load_results()
    all_teams = sorted(data["teams"].keys())

    st.title("Matchup Analyser")
    st.markdown("Select any two teams to see how the model evaluates their head-to-head.")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", all_teams, index=all_teams.index("Spain"))
    with col2:
        team2 = st.selectbox("Team 2", all_teams, index=all_teams.index("Argentina"))

    if team1 == team2:
        st.warning("Please select two different teams.")
        st.stop()

    model, feat_cols, prob_cache, team_stats = load_model()

    # Get symmetric probabilities
    p_h1, p_d1, p_a1 = prob_cache[(team1, team2)]
    p_h2, p_d2, p_a2 = prob_cache[(team2, team1)]
    p_t1   = (p_h1 + p_a2) / 2
    p_draw = (p_d1 + p_d2) / 2
    p_t2   = (p_a1 + p_h2) / 2

    st.divider()

    # ── Probability display ───────────────────────────────────────────────────
    st.subheader("Match Probabilities")
    st.caption("Averaged over both team orderings for neutral venue symmetry")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"{team1} Win", f"{p_t1*100:.1f}%")
    with c2:
        st.metric("Draw", f"{p_draw*100:.1f}%")
    with c3:
        st.metric(f"{team2} Win", f"{p_t2*100:.1f}%")

    # Probability bar
    prob_df = pd.DataFrame({
        "Outcome": [f"{team1} Win", "Draw", f"{team2} Win"],
        "Probability": [p_t1*100, p_draw*100, p_t2*100]
    }).set_index("Outcome")
    st.bar_chart(prob_df, color="#1a1a2e")

    st.divider()

    # ── Feature comparison ────────────────────────────────────────────────────
    st.subheader("Team Features")

    h = team_stats[team1]
    a = team_stats[team2]

    feat_comp = pd.DataFrame({
        "Feature":  ["Elo Rating", "Legacy Elo (20yr avg)",
                     "Win Rate (last 10)", "Goal Diff/game (last 10)", "Federation"],
        team1: [f"{h['elo']:.0f}", f"{h['legacy']:.0f}",
                f"{h['win_rate']:.2f}", f"{h['gd_pg']:+.2f}", h['federation']],
        team2: [f"{a['elo']:.0f}", f"{a['legacy']:.0f}",
                f"{a['win_rate']:.2f}", f"{a['gd_pg']:+.2f}", a['federation']],
    })
    st.dataframe(feat_comp.set_index("Feature"), use_container_width=True)

    st.divider()

    # ── Feature delta table ───────────────────────────────────────────────────
    st.subheader("Feature Advantage")
    deltas = []
    feature_map = [
        ("Elo Rating",          h["elo"],      a["elo"],      True),
        ("Legacy Elo (20yr)",   h["legacy"],   a["legacy"],   True),
        ("Win Rate (last 10)",  h["win_rate"], a["win_rate"], True),
        ("Goal Diff/g (last 10)",h["gd_pg"],  a["gd_pg"],    True),
    ]
    for label, hval, aval, higher_better in feature_map:
        diff = hval - aval
        favours = team1 if diff > 0 else (team2 if diff < 0 else "Even")
        deltas.append({"Feature": label,
                        team1: round(hval, 2),
                        team2: round(aval, 2),
                        "Difference": round(diff, 2),
                        "Favours": favours})
    st.dataframe(pd.DataFrame(deltas).set_index("Feature"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RUN A SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎲 Run a Simulation":
    st.title("Run a Simulation")
    st.markdown("Simulate one complete 2026 World Cup tournament. Every run is different — this is one random draw from the probability distribution.")

    _, _, prob_cache, _ = load_model()

    # Import only what we need — no sklearn involved
    from src.predict import (GROUPS as _GROUPS, simulate_once, ROUNDS,
                              reached, simulate_group,
                              R32_BRACKET, R16_PAIRS, QF_PAIRS, SF_PAIRS)

    if st.button("▶ Run Tournament Simulation", type="primary"):
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)

        with st.spinner("Simulating tournament..."):
            progress, group_pts = simulate_once(prob_cache)

        st.success(f"Simulation complete! (seed: {seed})")
        st.divider()

        # ── Group results ──────────────────────────────────────────────────────
        st.subheader("Group Stage Results")

        # Reconstruct group standings from progress + group_pts
        all_teams_list = [t for g in GROUPS.values() for t in g]

        # Rerun group stage deterministically with same seed for display
        np.random.seed(seed)
        group_standings = {}
        gp_all = {}
        gd_all = {}
        gf_all = {}

        for gname, teams in _GROUPS.items():
            ranked, pts, gd, gf = simulate_group(teams, prob_cache)
            group_standings[gname] = ranked
            for t in teams:
                gp_all[t] = pts[t]
                gd_all[t] = gd[t]
                gf_all[t] = gf[t]

        cols = st.columns(3)
        for i, (gname, ranked) in enumerate(group_standings.items()):
            with cols[i % 3]:
                st.markdown(f"**Group {gname}**")
                rows = []
                for pos, team in enumerate(ranked, 1):
                    adv = "✓" if pos <= 2 else ""
                    rows.append({
                        "Pos": pos, "Team": team,
                        "Pts": gp_all[team], "GD": gd_all[team],
                        "→": adv
                    })
                gdf = pd.DataFrame(rows).set_index("Pos")
                st.dataframe(gdf, use_container_width=True, hide_index=False)

        st.divider()

        # ── Knockout results ───────────────────────────────────────────────────
        st.subheader("Knockout Stage")

        # Find champion
        champion = next((t for t, p in progress.items() if p == "winner"), "Unknown")
        finalist = [t for t, p in progress.items() if p in ["final","winner"]]
        semis    = [t for t, p in progress.items() if p in ["sf","final","winner"]]
        quarters = [t for t, p in progress.items() if p in ["qf","sf","final","winner"]]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Quarter-finalists**")
            for t in sorted(quarters): st.markdown(f"- {t}")
        with c2:
            st.markdown("**Semi-finalists**")
            for t in sorted(semis): st.markdown(f"- {t}")

        st.divider()
        st.markdown(f"**Finalists:** {' vs '.join(finalist)}")

        st.divider()
        st.markdown(f"## 🏆 Champion: {champion}")
        st.balloons()

    else:
        st.info("Click the button above to simulate a complete tournament.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Methodology":
    st.title("Methodology")

    st.subheader("Overview")
    st.markdown("""
    This predictor uses a **Random Forest classifier** trained on 11,500+ competitive 
    international football matches from 2000–2025. Match outcome probabilities (Win / Draw / Loss) 
    are fed into a **Monte Carlo simulation** that plays out the full 2026 FIFA World Cup bracket 
    10,000 times, producing stable probability estimates for each team at every stage.
    """)

    st.subheader("Features")
    feat_data = {
        "Feature": ["Elo Rating", "Elo Difference", "Legacy Elo (20yr avg)",
                     "Legacy Difference", "Win Rate (last 10)", "Goal Diff/game (last 10)",
                     "Federation"],
        "Description": [
            "Team's current Elo rating computed from full match history (1872–present)",
            "Home Elo minus Away Elo — strongest single predictor",
            "Rolling 20-year average Elo — captures historical strength",
            "Legacy Elo difference between teams",
            "Fraction of last 10 competitive matches won",
            "Average goal difference per game in last 10 matches",
            "FIFA confederation (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC)"
        ],
        "Importance": ["High", "Very High", "High", "High", "Medium", "Medium", "Low–Medium"]
    }
    st.dataframe(pd.DataFrame(feat_data).set_index("Feature"), use_container_width=True)

    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", "62.7%", help="% of match outcomes correctly predicted")
    with col2:
        st.metric("Test Log Loss", "0.827", help="Lower is better. Random guessing = 1.099")
    with col3:
        st.metric("CV Std Dev", "±0.8%", help="Stability across 3 independent time periods")

    st.subheader("Competitions Used for Training")
    st.markdown("""
    - FIFA World Cup & Qualifiers
    - UEFA European Championship & Qualifiers  
    - Copa América
    - Africa Cup of Nations & Qualifiers
    - AFC Asian Cup & Qualifiers
    - CONCACAF Gold Cup & Qualifiers
    - FIFA Confederations Cup
    """)

    st.subheader("Tournament Simulation")
    st.markdown("""
    The simulation follows the **real FIFA 2026 bracket structure** exactly — fixed R32 pairings 
    based on group finish (1E vs best-3(ABCDF), etc.). Third-place teams are ranked by points, 
    then goal difference, then goals scored, and the best 8 advance.
    
    In knockout rounds, draw probability is redistributed 50/50 to each team (simulating extra 
    time and penalties). All World Cup matches are treated as neutral venue.
    
    Match probabilities are **symmetrised** — averaged over both team orderings — to remove 
    any artificial home/away bias.
    """)

    st.subheader("Data Sources")
    st.markdown("""
    - Match results: [Kaggle — International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
    - Elo ratings: computed from scratch using the standard Elo update formula (K=32)
    """)
