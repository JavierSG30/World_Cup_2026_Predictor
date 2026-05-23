"""
World Cup 2026 Predictor — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
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
    top10 = df.head(10).copy()

    # Flag emoji map
    FLAGS = {
        "Spain":"🇪🇸","Argentina":"🇦🇷","France":"🇫🇷","Brazil":"🇧🇷",
        "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Germany":"🇩🇪","Netherlands":"🇳🇱","Portugal":"🇵🇹",
        "Croatia":"🇭🇷","Colombia":"🇨🇴","Belgium":"🇧🇪","Uruguay":"🇺🇾",
        "Ecuador":"🇪🇨","Mexico":"🇲🇽","United States":"🇺🇸","Canada":"🇨🇦",
        "Morocco":"🇲🇦","Senegal":"🇸🇳","Japan":"🇯🇵","South Korea":"🇰🇷",
        "Australia":"🇦🇺","Iran":"🇮🇷","Switzerland":"🇨🇭","Denmark":"🇩🇰",
        "Sweden":"🇸🇪","Norway":"🇳🇴","Austria":"🇦🇹","Czechia":"🇨🇿",
        "Turkey":"🇹🇷","Türkiye":"🇹🇷","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Wales":"🏴󠁧󠁢󠁷󠁬󠁳󠁿",
        "Serbia":"🇷🇸","Poland":"🇵🇱","Ukraine":"🇺🇦","Slovakia":"🇸🇰",
        "Hungary":"🇭🇺","Bosnia and Herzegovina":"🇧🇦","Albania":"🇦🇱",
        "Georgia":"🇬🇪","Slovenia":"🇸🇮","Kosovo":"🇽🇰","Finland":"🇫🇮",
        "Iceland":"🇮🇸","Romania":"🇷🇴","Bulgaria":"🇧🇬","North Macedonia":"🇲🇰",
        "Paraguay":"🇵🇾","Chile":"🇨🇱","Peru":"🇵🇪","Venezuela":"🇻🇪","Bolivia":"🇧🇴",
        "Costa Rica":"🇨🇷","Honduras":"🇭🇳","Jamaica":"🇯🇲","Panama":"🇵🇦",
        "Haiti":"🇭🇹","El Salvador":"🇸🇻","Guatemala":"🇬🇹","Curaçao":"🇨🇼",
        "Trinidad and Tobago":"🇹🇹",
        "Egypt":"🇪🇬","Nigeria":"🇳🇬","Ghana":"🇬🇭","Ivory Coast":"🇨🇮",
        "Algeria":"🇩🇿","Tunisia":"🇹🇳","Cameroon":"🇨🇲","South Africa":"🇿🇦",
        "Mali":"🇲🇱","Burkina Faso":"🇧🇫","Guinea":"🇬🇳","DR Congo":"🇨🇩",
        "Cape Verde":"🇨🇻","Benin":"🇧🇯","Gabon":"🇬🇦","Namibia":"🇳🇦",
        "Rwanda":"🇷🇼","Mozambique":"🇲🇿","Tanzania":"🇹🇿","Uganda":"🇺🇬",
        "Saudi Arabia":"🇸🇦","Qatar":"🇶🇦","Iraq":"🇮🇶","Jordan":"🇯🇴",
        "UAE":"🇦🇪","Uzbekistan":"🇺🇿","Oman":"🇴🇲","Bahrain":"🇧🇭",
        "China PR":"🇨🇳","India":"🇮🇳","Vietnam":"🇻🇳","Thailand":"🇹🇭",
        "Indonesia":"🇮🇩","Philippines":"🇵🇭","Palestine":"🇵🇸","Lebanon":"🇱🇧",
        "New Zealand":"🇳🇿","Fiji":"🇫🇯",
    }

    labels = [f"{FLAGS.get(t,'🏳')}\n{t}" for t in top10["Team"]]
    vals   = top10["winner"].tolist()

    fig = go.Figure(go.Bar(
        x=labels,
        y=vals,
        marker_color="#3a7bd5",
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color="white"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="DM Sans"),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        xaxis=dict(tickfont=dict(size=12), tickangle=0),
        margin=dict(t=40, b=20, l=0, r=0),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

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
            np.random.seed(seed)
            from src.predict import ko_match
            group_standings = {}
            all_pts2, all_gd2, all_gf2 = {}, {}, {}
            for gname, teams in _GROUPS.items():
                ranked, pts, gd, gf = simulate_group(teams, prob_cache)
                group_standings[gname] = ranked
                for t in teams:
                    all_pts2[t]=pts[t]; all_gd2[t]=gd[t]; all_gf2[t]=gf[t]

            thirds2 = [group_standings[g][2] for g in _GROUPS]
            thirds_ranked2 = sorted(thirds2, key=lambda t:(all_pts2[t],all_gd2[t],all_gf2[t]), reverse=True)
            thirds_ws2 = [(t,all_pts2[t],all_gd2[t],all_gf2[t]) for t in thirds_ranked2]
            used2 = set(); third_slot2 = {}
            for s1,s2 in R32_BRACKET:
                for slot in (s1,s2):
                    if slot.startswith("3"):
                        allowed = set(slot[1:])
                        for t,_,_,_ in thirds_ws2:
                            grp = next(g for g,r in group_standings.items() if r[2]==t)
                            if grp in allowed and t not in used2:
                                used2.add(t); third_slot2[slot]=t; break

            def get_slot2(s):
                if s.startswith("1"): return group_standings[s[1]][0]
                elif s.startswith("2"): return group_standings[s[1]][1]
                else: return third_slot2.get(s,"TBD")

            r32=[]; r16=[]; qf=[]; sf=[]
            for s1,s2 in R32_BRACKET:
                t1,t2=get_slot2(s1),get_slot2(s2)
                w=ko_match(t1,t2,prob_cache); r32.append((t1,t2,w))
            for i1,i2 in R16_PAIRS:
                t1,t2=r32[i1][2],r32[i2][2]
                w=ko_match(t1,t2,prob_cache); r16.append((t1,t2,w))
            for i1,i2 in QF_PAIRS:
                t1,t2=r16[i1][2],r16[i2][2]
                w=ko_match(t1,t2,prob_cache); qf.append((t1,t2,w))
            for i1,i2 in SF_PAIRS:
                t1,t2=qf[i1][2],qf[i2][2]
                w=ko_match(t1,t2,prob_cache); sf.append((t1,t2,w))
            champion = ko_match(sf[0][2],sf[1][2],prob_cache)

        st.success(f"Simulation complete! (seed: {seed})")

        # Group Stage
        st.subheader("Group Stage")
        gcols = st.columns(3)
        for i, (gname, ranked) in enumerate(group_standings.items()):
            with gcols[i % 3]:
                st.markdown(f"**Group {gname}**")
                rows = []
                for pos, team in enumerate(ranked, 1):
                    adv = "✓" if pos <= 2 else ("?" if pos==3 else "")
                    rows.append({"Pos":pos,"Team":team,"Pts":all_pts2[team],"GD":all_gd2[team],"":adv})
                st.dataframe(pd.DataFrame(rows).set_index("Pos"), use_container_width=True)

        st.divider()

        # Bracket
        st.subheader("Knockout Bracket")
        st.caption("Green = winner of that match | Gold = champion")

        def truncate(name, n=12):
            return name if len(name) <= n else name[:n-1]+"."

        def make_bracket(r32, r16, qf, sf, champ):
            W, H = 1100, 700
            BG="#0f1117"; BOX="#1e2130"; WIN="#1a472a"; TEXT="#e0e0e0"
            DIM="#555"; GOLD="#f0b429"; LINE="#444"
            BW=105; BH=18; GAP=5; MGAP=30

            out = []
            out.append(f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="background:{BG};border-radius:10px;font-family:monospace">')

            def b(x,y,team,winner,is_champ=False):
                bg = GOLD if is_champ else (WIN if winner else BOX)
                tc = "#000" if is_champ else TEXT
                fs = 8 if len(team)>14 else 9
                t = truncate(team, 14)
                out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{BW}" height="{BH}" rx="3" fill="{bg}" stroke="#333" stroke-width="0.5"/>')
                out.append(f'<text x="{x+BW/2:.0f}" y="{y+BH/2+3:.0f}" text-anchor="middle" fill="{tc}" font-size="{fs}">{t}</text>')

            def conn(x1,y1,x2,y2):
                mx=(x1+x2)/2
                out.append(f'<polyline points="{x1:.0f},{y1:.0f} {mx:.0f},{y1:.0f} {mx:.0f},{y2:.0f} {x2:.0f},{y2:.0f}" fill="none" stroke="{LINE}" stroke-width="0.8"/>')

            def match_y(i):
                return 30 + i*(BH*2+GAP+MGAP)

            def draw_col(x, matches, cy_prev=None, cx_from=None, cx_to=None, right=False):
                cys = []
                for i, (t1,t2,w) in enumerate(matches):
                    if cy_prev is not None:
                        y1c = cy_prev[i*2]
                        y2c = cy_prev[i*2+1]
                        ybox = (y1c+y2c)/2 - BH - GAP/2
                    else:
                        ybox = match_y(i)
                    b(x, ybox,      t1, w==t1)
                    b(x, ybox+BH+GAP, t2, w==t2)
                    if cy_prev is not None:
                        if right:
                            conn(cx_from, y1c, x+BW, ybox+BH/2)
                            conn(cx_from, y2c, x+BW, ybox+BH+GAP+BH/2)
                        else:
                            conn(cx_from, y1c, x, ybox+BH/2)
                            conn(cx_from, y2c, x, ybox+BH+GAP+BH/2)
                    cys.append(ybox + BH + GAP/2)
                return cys

            # Left side columns
            x0=8
            cy0L = draw_col(x0, r32[:8])
            x1=x0+BW+16
            cy1L = draw_col(x1, r16[:4], cy0L, x0+BW)
            x2=x1+BW+16
            cy2L = draw_col(x2, qf[:2], cy1L, x1+BW)
            x3=x2+BW+16
            cy3L = draw_col(x3, sf[:1], cy2L, x2+BW)

            # Right side columns (mirror)
            x0r=W-8-BW
            cy0R = draw_col(x0r, r32[8:], right=False)
            x1r=x0r-BW-16
            cy1R = draw_col(x1r, r16[4:], cy0R, x0r, right=True)
            x2r=x1r-BW-16
            cy2R = draw_col(x2r, qf[2:], cy1R, x1r, right=True)
            x3r=x2r-BW-16
            cy3R = draw_col(x3r, sf[1:], cy2R, x2r, right=True)

            # Final
            xfc=W//2-BW//2
            t1f,t2f=sf[0][2],sf[1][2]
            yf=(cy3L[0]+cy3R[0])/2-BH-GAP/2
            b(xfc, yf,        t1f, champ==t1f)
            b(xfc, yf+BH+GAP, t2f, champ==t2f)
            conn(x3+BW, cy3L[0], xfc, yf+BH/2)
            conn(x3r,   cy3R[0], xfc+BW, yf+BH+GAP+BH/2)

            # Champion
            ych = yf+BH*2+GAP+18
            out.append(f'<rect x="{xfc:.0f}" y="{ych:.0f}" width="{BW}" height="{BH+4}" rx="4" fill="{GOLD}" stroke="#aaa" stroke-width="0.8"/>')
            tc2 = truncate(champ, 12)
            out.append(f'<text x="{xfc+BW/2:.0f}" y="{ych+BH/2+5:.0f}" text-anchor="middle" fill="#000" font-size="10" font-weight="bold">🏆 {tc2}</text>')

            # Headers
            for lbl,x in [("R32",x0),("R16",x1),("QF",x2),("SF",x3),("FINAL",xfc),("SF",x3r),("QF",x2r),("R16",x1r),("R32",x0r)]:
                out.append(f'<text x="{x+BW/2:.0f}" y="22" text-anchor="middle" fill="{DIM}" font-size="8">{lbl}</text>')

            out.append("</svg>")
            return "\n".join(out)

        svg_html = make_bracket(r32, r16, qf, sf, champion)
        st.markdown(svg_html, unsafe_allow_html=True)

        st.divider()
        st.markdown(f"## 🏆 Champion: **{champion}**")
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
