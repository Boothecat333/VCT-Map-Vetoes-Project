"""
VCT 2026 Map Analytics & BO3 Veto Simulator Web Application.
Interactive Streamlit dashboard featuring Map Power Rankings, Head-to-Head Comparison,
Interactive 'Be The GM' BO3 Veto Simulator, and SQL Portfolio Explorer.
"""

import os
from pathlib import Path
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.database import (
    DEFAULT_DB_PATH,
    query_dataframe,
    get_all_teams,
    get_all_maps,
    get_team_map_summary,
    get_head_to_head_matches,
    get_teams_by_league,
)
from src.power_score import PowerScoreModel, ACTIVE_MAP_POOL, DEFAULT_HALF_LIFE_DAYS
from src.veto_simulator import VetoSimulator, ActionType, VetoState

# Page configuration
st.set_page_config(
    page_title="VCT 2026 Map Analytics & Veto Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1a1f2c;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #2a3142;
    }
    .veto-card {
        background: #161b26;
        border: 1px solid #2e384d;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .ban-badge {
        color: #ff4d4f;
        font-weight: bold;
        background: rgba(255, 77, 79, 0.15);
        padding: 3px 8px;
        border-radius: 4px;
    }
    .pick-badge {
        color: #52c41a;
        font-weight: bold;
        background: rgba(82, 196, 26, 0.15);
        padding: 3px 8px;
        border-radius: 4px;
    }
    .decider-badge {
        color: #faad14;
        font-weight: bold;
        background: rgba(250, 173, 20, 0.15);
        padding: 3px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data_and_fit_model():
    """Loads map results from SQLite/CSV and fits the Power Score model."""
    csv_path = Path(__file__).resolve().parent / "data" / "processed" / "vct_2026_maps.csv"
    if DEFAULT_DB_PATH.exists():
        try:
            df = query_dataframe("SELECT * FROM map_results;")
            if not df.empty:
                model = PowerScoreModel(half_life_days=DEFAULT_HALF_LIFE_DAYS).fit(df)
                return df, model
        except Exception:
            pass

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        model = PowerScoreModel(half_life_days=DEFAULT_HALF_LIFE_DAYS).fit(df)
        return df, model

    return pd.DataFrame(), None


# Header
st.title("🎯 VCT 2026 Map Analytics & Veto Simulator")
st.caption("Opponent-adjusted Power Scores, 45-day exponential recency decay, and BO3 'Be The GM' engine.")

df, model = load_data_and_fit_model()

if df.empty or model is None:
    st.warning("⚠️ No match data found. Please execute the data pipeline first.")
    if st.button("🚀 Run Data Pipeline Now"):
        with st.spinner("Downloading VCT DuckDB and processing 2026 maps..."):
            from src.data_pipeline import run_pipeline
            run_pipeline()
            st.success("Pipeline complete! Refreshing...")
            st.rerun()
    st.stop()

# Navigation tabs
tab_rankings, tab_h2h, tab_gm, tab_sql = st.tabs([
    "🏆 Map Power Rankings",
    "⚔️ Head-to-Head Matchup",
    "🎮 'Be The GM' Veto Simulator",
    "🗄️ SQL Portfolio Explorer",
])

# ====================================================================
# TAB 1: MAP POWER RANKINGS
# ====================================================================
with tab_rankings:
    st.header("🏆 2026 VCT Map Power Rankings")
    st.markdown("""
    Power Scores (0–100) are evaluated on every 2026 match using **round differentials**, 
    **Bayesian shrinkage**, **45-day exponential recency decay**, and **two-degree opponent strength adjustment**.
    """)

    col1, col2, col3, col4 = st.columns(4)
    total_matches = df["match_id"].nunique()
    total_maps_played = len(df) // 2
    total_teams = df["team"].nunique()
    latest_date = df["date"].max()

    col1.metric("Series Analyzed", f"{total_matches:,}")
    col2.metric("Maps Played", f"{total_maps_played:,}")
    col3.metric("Teams Tracked", f"{total_teams:,}")
    col4.metric("Latest Match Date", str(latest_date))

    teams_by_league = get_teams_by_league()

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_league = st.selectbox(
            "Filter by League:",
            options=["All Leagues", "Americas", "EMEA", "Pacific", "China"],
            index=0,
            key="rankings_league",
        )
    with filter_col2:
        selected_map = st.selectbox(
            "Filter by Map Leaderboard:",
            options=["All Active Maps (Heatmap)"] + ACTIVE_MAP_POOL,
            index=0,
            key="rankings_map",
        )

    ratings_df = model.to_dataframe(map_pool=ACTIVE_MAP_POOL)
    if selected_league != "All Leagues":
        allowed_teams = set(teams_by_league.get(selected_league, []))
        ratings_df = ratings_df[ratings_df["team"].isin(allowed_teams)]

    if selected_map == "All Active Maps (Heatmap)":
        pivot_ps = ratings_df.pivot(index="team", columns="map", values="power_score").fillna(50.0)
        # Sort teams by their average power score
        pivot_ps["Average"] = pivot_ps.mean(axis=1)
        pivot_ps = pivot_ps.sort_values(by="Average", ascending=False).drop(columns=["Average"])

        avail_count = len(pivot_ps)
        if avail_count > 5:
            top_teams_count = st.slider("Number of teams to display:", min_value=min(5, avail_count), max_value=avail_count, value=min(25, avail_count))
            display_pivot = pivot_ps.head(top_teams_count)
        else:
            display_pivot = pivot_ps

        fig_heat = px.imshow(
            display_pivot,
            labels=dict(x="Map", y="Team", color="Power Score"),
            x=display_pivot.columns,
            y=display_pivot.index,
            color_continuous_scale="Viridis",
            text_auto=".1f",
            aspect="auto",
            height=max(450, top_teams_count * 22),
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f2f6"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    else:
        map_filter = ratings_df[ratings_df["map"] == selected_map].copy()
        map_filter = map_filter.sort_values(by="power_score", ascending=False).reset_index(drop=True)
        map_filter.index += 1

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader(f"Top Teams on {selected_map}")
            display_df = map_filter[[
                "team", "power_score", "record", "win_rate_pct", "avg_round_diff", "maps_played"
            ]].rename(columns={
                "team": "Team",
                "power_score": "Power Score (0-100)",
                "record": "2026 Record (W-L)",
                "win_rate_pct": "Win Rate %",
                "avg_round_diff": "Avg Round Diff",
                "maps_played": "Maps Played",
            })
            st.dataframe(
                display_df.style.background_gradient(subset=["Power Score (0-100)"], cmap="Greens"),
                use_container_width=True,
                height=450,
            )

        with c2:
            st.subheader(f"{selected_map} Score Distribution")
            fig_hist = px.histogram(
                map_filter,
                x="power_score",
                nbins=15,
                title=f"Power Score Spread on {selected_map}",
                color_discrete_sequence=["#00f2fe"],
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f2f6"),
                xaxis_title="Power Score",
                yaxis_title="Team Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# ====================================================================
# TAB 2: HEAD-TO-HEAD MATCHUP
# ====================================================================
with tab_h2h:
    st.header("⚔️ Head-to-Head Map Comparison")
    teams_by_league = get_teams_by_league()
    all_teams_list = sorted(model.teams)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Team A")
        league_a = st.selectbox(
            "Select League A:",
            options=["Americas", "EMEA", "Pacific", "China", "All Leagues"],
            index=0,
            key="h2h_league_a"
        )
        teams_a_opts = teams_by_league.get(league_a, all_teams_list) if league_a != "All Leagues" else all_teams_list
        def_a_idx = teams_a_opts.index("G2 Esports") if "G2 Esports" in teams_a_opts else 0
        team_a = st.selectbox("Select Team A:", options=teams_a_opts, index=def_a_idx, key="h2h_team_a")

    with col_t2:
        st.subheader("Team B")
        league_b = st.selectbox(
            "Select League B:",
            options=["Americas", "EMEA", "Pacific", "China", "All Leagues"],
            index=0,
            key="h2h_league_b"
        )
        teams_b_opts = teams_by_league.get(league_b, all_teams_list) if league_b != "All Leagues" else all_teams_list
        def_b_target = "Sentinels" if "Sentinels" in teams_b_opts else (teams_b_opts[1] if len(teams_b_opts) > 1 else teams_b_opts[0])
        def_b_idx = teams_b_opts.index(def_b_target)
        team_b = st.selectbox("Select Team B:", options=teams_b_opts, index=def_b_idx, key="h2h_team_b")

    if team_a == team_b:
        st.info("Select two different teams to compare map pool strengths.")
    else:
        # Build comparison metrics for active pool
        h2h_data = []
        for m in ACTIVE_MAP_POOL:
            r_a = model.get_rating(team_a, m)
            r_b = model.get_rating(team_b, m)
            prob_a = model.predict_map_win_probability(team_a, team_b, m)
            h2h_data.append({
                "Map": m,
                f"{team_a} Score": r_a.power_score,
                f"{team_b} Score": r_b.power_score,
                f"{team_a} Win Prob %": round(prob_a * 100, 1),
                f"{team_b} Win Prob %": round((1.0 - prob_a) * 100, 1),
                "Advantage": f"{team_a} (+{r_a.power_score - r_b.power_score:.1f})" if r_a.power_score >= r_b.power_score else f"{team_b} (+{r_b.power_score - r_a.power_score:.1f})",
                f"{team_a} Rec": f"{r_a.wins}-{r_a.losses}",
                f"{team_b} Rec": f"{r_b.wins}-{r_b.losses}",
            })
        h2h_df = pd.DataFrame(h2h_data)

        # Radar Chart
        col_radar, col_table = st.columns([1, 1])
        with col_radar:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[f"{team_a} Score"] for row in h2h_data] + [h2h_data[0][f"{team_a} Score"]],
                theta=ACTIVE_MAP_POOL + [ACTIVE_MAP_POOL[0]],
                fill='toself',
                name=team_a,
                line_color='#00f2fe',
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[f"{team_b} Score"] for row in h2h_data] + [h2h_data[0][f"{team_b} Score"]],
                theta=ACTIVE_MAP_POOL + [ACTIVE_MAP_POOL[0]],
                fill='toself',
                name=team_b,
                line_color='#ff4b4b',
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color="#a0aec0"),
                    bgcolor="#161b26",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f2f6"),
                title=f"Active Map Pool Power Score Radar",
                height=420,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_table:
            st.subheader(f"Map Pool Breakdown")
            st.dataframe(
                h2h_df[[
                    "Map", f"{team_a} Score", f"{team_b} Score", f"{team_a} Win Prob %", "Advantage"
                ]],
                use_container_width=True,
                height=380,
            )

        # Historical Head to Head
        st.subheader(f"Historical 2026 Matches: {team_a} vs {team_b}")
        h2h_matches = get_head_to_head_matches(team_a, team_b)
        if not h2h_matches.empty:
            st.dataframe(
                h2h_matches[[
                    "date", "event", "stage", "map", "rounds_for", "rounds_against", "result"
                ]].rename(columns={
                    "date": "Date", "event": "Event", "stage": "Stage", "map": "Map",
                    "rounds_for": f"{team_a} Rounds", "rounds_against": f"{team_b} Rounds", "result": "Result"
                }),
                use_container_width=True,
            )
        else:
            st.caption(f"No direct tier-1 map matches recorded between {team_a} and {team_b} in 2026 yet.")

# ====================================================================
# TAB 3: BE THE GM & BO3 VETO SIMULATOR
# ====================================================================
with tab_gm:
    st.header("🎮 'Be The GM' & BO3 Veto Simulator")
    st.markdown("""
    Simulate official VCT BO3 map vetoes with the 7-map pool: 
    **Abyss, Ascent, Haven, Lotus, Split, Summit, Sunset**.
    """)

    teams_by_league = get_teams_by_league()
    all_teams_list = sorted(model.teams)

    col_sim_a, col_sim_b = st.columns(2)
    with col_sim_a:
        st.subheader("Team A (1st Ban & 1st Pick)")
        gm_league_a = st.selectbox(
            "Select League A:",
            options=["Americas", "EMEA", "Pacific", "China", "All Leagues"],
            index=0,
            key="gm_league_a"
        )
        teams_gm_a = teams_by_league.get(gm_league_a, all_teams_list) if gm_league_a != "All Leagues" else all_teams_list
        def_gm_a_idx = teams_gm_a.index("G2 Esports") if "G2 Esports" in teams_gm_a else 0
        gm_team_a = st.selectbox("Select Team A:", options=teams_gm_a, index=def_gm_a_idx, key="gm_a")

    with col_sim_b:
        st.subheader("Team B (2nd Ban & 2nd Pick)")
        gm_league_b = st.selectbox(
            "Select League B:",
            options=["Americas", "EMEA", "Pacific", "China", "All Leagues"],
            index=0,
            key="gm_league_b"
        )
        teams_gm_b = teams_by_league.get(gm_league_b, all_teams_list) if gm_league_b != "All Leagues" else all_teams_list
        def_gm_b_target = "Sentinels" if "Sentinels" in teams_gm_b else (teams_gm_b[1] if len(teams_gm_b) > 1 else teams_gm_b[0])
        def_gm_b_idx = teams_gm_b.index(def_gm_b_target)
        gm_team_b = st.selectbox("Select Team B:", options=teams_gm_b, index=def_gm_b_idx, key="gm_b")

    sim_mode = st.radio(
        "Select Simulation Mode:",
        ["🤖 Fully Automated Simulation (AI vs AI)", "👔 'Be The GM' Interactive Mode"],
        horizontal=True,
    )

    simulator = VetoSimulator(model=model, map_pool=ACTIVE_MAP_POOL)

    if sim_mode == "🤖 Fully Automated Simulation (AI vs AI)":
        if st.button("▶️ Run Automated Veto Simulation", type="primary"):
            sim_state = simulator.simulate_full_veto(gm_team_a, gm_team_b)
            series_odds = simulator.calculate_series_odds(sim_state)

            st.success("✅ Veto Simulation Complete!")

            # Series Odds Banner
            odds_col1, odds_col2 = st.columns(2)
            odds_col1.metric(f"{gm_team_a} Series Win Probability", f"{series_odds['team_a_pct']}%")
            odds_col2.metric(f"{gm_team_b} Series Win Probability", f"{series_odds['team_b_pct']}%")

            # Final Match Sheet
            st.subheader("📋 Final Series Map Order")
            m1 = series_odds["map_1"]
            m2 = series_odds["map_2"]
            m3 = series_odds["map_3"]

            card_col1, card_col2, card_col3 = st.columns(3)
            with card_col1:
                st.markdown(f"""
                <div class="veto-card">
                    <h4>Map 1: <span class="pick-badge">{m1['map']}</span></h4>
                    <p><b>Picked by:</b> {m1['picker']}</p>
                    <p><b>Side Choice:</b> {gm_team_b if m1['picker'] == gm_team_a else gm_team_a}</p>
                    <p><b>Win Odds:</b> {gm_team_a} {m1['team_a_win_pct']}% — {gm_team_b} {100-m1['team_a_win_pct']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with card_col2:
                st.markdown(f"""
                <div class="veto-card">
                    <h4>Map 2: <span class="pick-badge">{m2['map']}</span></h4>
                    <p><b>Picked by:</b> {m2['picker']}</p>
                    <p><b>Side Choice:</b> {gm_team_a if m2['picker'] == gm_team_b else gm_team_b}</p>
                    <p><b>Win Odds:</b> {gm_team_a} {m2['team_a_win_pct']}% — {gm_team_b} {100-m2['team_a_win_pct']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with card_col3:
                st.markdown(f"""
                <div class="veto-card">
                    <h4>Map 3: <span class="decider-badge">{m3['map']}</span></h4>
                    <p><b>Status:</b> Decider Map</p>
                    <p><b>Side Choice:</b> Higher Seed / Coin Toss</p>
                    <p><b>Win Odds:</b> {gm_team_a} {m3['team_a_win_pct']}% — {gm_team_b} {100-m3['team_a_win_pct']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # Pick-Ban Steps History
            st.subheader("📜 Turn-by-Turn Veto Log")
            for step in sim_state.history:
                badge_class = "ban-badge" if step.action_type == ActionType.BAN else ("pick-badge" if step.action_type == ActionType.PICK else "decider-badge")
                st.markdown(f"""
                <div class="veto-card">
                    <b>Turn {step.step_number}:</b> {step.actor_team} <span class="{badge_class}">{step.action_type.value}</span> <b>{step.map_name}</b><br>
                    <small style="color: #94a3b8;">💡 {step.reasoning}</small>
                </div>
                """, unsafe_allow_html=True)

    else:
        # Interactive "Be The GM" Mode
        gm_seat = st.selectbox("Take GM Seat for:", options=[gm_team_a, gm_team_b], index=0)

        # Initialize or reset interactive session state
        session_key = f"veto_{gm_team_a}_{gm_team_b}"
        if session_key not in st.session_state or st.button("🔄 Reset Veto Session"):
            st.session_state[session_key] = simulator.create_session(gm_team_a, gm_team_b)

        state: VetoState = st.session_state[session_key]

        if state.is_complete:
            st.success("🎉 BO3 Veto Complete!")
            series_odds = simulator.calculate_series_odds(state)
            
            c_odds1, c_odds2 = st.columns(2)
            c_odds1.metric(f"{gm_team_a} Series Win Chance", f"{series_odds['team_a_pct']}%")
            c_odds2.metric(f"{gm_team_b} Series Win Chance", f"{series_odds['team_b_pct']}%")

            st.write("---")
            st.subheader("Final Maps:")
            st.write(f"**Map 1 ({state.picked_maps[0][1]}'s pick):** {state.picked_maps[0][0]}")
            st.write(f"**Map 2 ({state.picked_maps[1][1]}'s pick):** {state.picked_maps[1][0]}")
            st.write(f"**Map 3 (Decider):** {state.decider_map}")

        else:
            prompt = simulator.get_current_prompt(state)
            rec_map, rec_reason = simulator.recommend_action(state)

            st.info(f"**Turn {prompt['step_number']} of 7**: **{prompt['actor']}** {prompt['description']}")

            # Show AI Advisor Advice
            st.markdown(f"""
            <div class="veto-card">
                <b>💡 AI Advisor Recommendation:</b> Suggests <b>{prompt['action_type'].value} {rec_map}</b><br>
                <small style="color: #cbd5e1;">{rec_reason}</small>
            </div>
            """, unsafe_allow_html=True)

            is_user_turn = (prompt["actor"] == gm_seat)

            if is_user_turn:
                st.write(f"**Your Decision as {gm_seat} ({prompt['action_type'].value}):**")
                cols = st.columns(len(state.available_maps))
                for i, map_name in enumerate(state.available_maps):
                    prob_user = model.predict_map_win_probability(gm_seat, gm_team_b if gm_seat == gm_team_a else gm_team_a, map_name)
                    btn_label = f"{map_name}\n({prob_user*100:.0f}%)"
                    if cols[i].button(btn_label, key=f"btn_{map_name}_{state.current_step_index}"):
                        state = simulator.execute_step(state, map_name, reasoning="Selected by GM")
                        st.session_state[session_key] = state
                        st.rerun()

            else:
                st.write(f"**Opponent Turn:** Waiting for {prompt['actor']}...")
                if st.button(f"Execute {prompt['actor']}'s Move ({rec_map})", type="primary"):
                    state = simulator.execute_step(state, rec_map, reasoning=rec_reason)
                    st.session_state[session_key] = state
                    st.rerun()

        # History log
        if state.history:
            st.subheader("Current Veto Log")
            for step in state.history:
                badge_class = "ban-badge" if step.action_type == ActionType.BAN else ("pick-badge" if step.action_type == ActionType.PICK else "decider-badge")
                st.markdown(f"""
                <div class="veto-card">
                    <b>Turn {step.step_number}:</b> {step.actor_team} <span class="{badge_class}">{step.action_type.value}</span> <b>{step.map_name}</b>
                </div>
                """, unsafe_allow_html=True)

# ====================================================================
# TAB 4: SQL PORTFOLIO EXPLORER
# ====================================================================
with tab_sql:
    st.header("🗄️ SQL Portfolio Explorer")
    st.markdown("""
    Demonstrating real-world **data analyst SQL queries** against SQLite (`vct_analytics.db`).
    These queries retrieve, aggregate, and compute rolling metrics from raw map observations.
    """)

    PRELOADED_QUERIES = {
        "1. Team Map Record Summary (Aggregation & Win Rates)": """-- 1. Team Performance by Map (Record, Win Rate, Round Differential)
SELECT 
    team,
    map,
    COUNT(*) AS maps_played,
    SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) AS losses,
    ROUND(100.0 * SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_rate_pct,
    ROUND(AVG(rounds_for), 1) AS avg_rounds_for,
    ROUND(AVG(rounds_against), 1) AS avg_rounds_against,
    ROUND(AVG(round_differential), 2) AS avg_round_margin
FROM map_results
GROUP BY team, map
HAVING COUNT(*) >= 2
ORDER BY win_rate_pct DESC, avg_round_margin DESC
LIMIT 15;""",

        "2. Active Map Pool Popularity & Overtime Rates": """-- 2. Active Map Pool Popularity & Balance
SELECT 
    map,
    COUNT(*) / 2 AS total_times_played,
    ROUND(AVG(rounds_for + rounds_against), 2) AS avg_total_rounds,
    SUM(CASE WHEN rounds_for > 13 OR rounds_against > 13 THEN 1 ELSE 0 END) / 2 AS overtime_games
FROM map_results
GROUP BY map
ORDER BY total_times_played DESC;""",

        "3. Rolling Form / Momentum (SQL Window Functions)": """-- 3. Recent Form / Rolling Round Margin (SQL Window Function)
SELECT 
    date,
    event,
    team,
    opponent,
    map,
    round_differential,
    ROUND(AVG(round_differential) OVER (
        PARTITION BY team 
        ORDER BY date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_3gm_round_margin
FROM map_results
WHERE team IN ('G2 Esports', 'Sentinels', 'Fnatic', 'Gen.G')
ORDER BY team, date DESC
LIMIT 20;""",

        "4. Top 10 Most Dominant Map Wins in 2026": """-- 4. Top 10 Most Dominant Map Wins in 2026
SELECT 
    date,
    event,
    team AS winner,
    opponent AS loser,
    map,
    rounds_for || '-' || rounds_against AS score,
    round_differential
FROM map_results
WHERE result = 'W'
ORDER BY round_differential DESC, date DESC
LIMIT 10;""",
    }

    selected_query_label = st.selectbox("Select a Portfolio Query to inspect:", options=list(PRELOADED_QUERIES.keys()))
    default_query_text = PRELOADED_QUERIES[selected_query_label]

    user_sql = st.text_area(
        "SQL Query Editor (Live on vct_analytics.db):",
        value=default_query_text,
        height=200,
    )

    if st.button("⚡ Execute SQL Query", type="primary"):
        try:
            query_res = query_dataframe(user_sql)
            st.success(f"Query returned {len(query_res)} rows.")
            st.dataframe(query_res, use_container_width=True)

            csv_data = query_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results as CSV",
                data=csv_data,
                file_name="sql_query_result.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"SQL Error: {e}")
