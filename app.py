"""
VCT 2026 Map Analytics & BO3 Veto Simulator Web Application.
Interactive Streamlit dashboard featuring Map Power Rankings, Head-to-Head Comparison,
and Interactive 'Be The GM' BO3 Veto Simulator.
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
    get_team_leagues,
)
from src.power_score import PowerScoreModel, ACTIVE_MAP_POOL, DEFAULT_DECAY_DAYS, DEFAULT_RETENTION_RATE
from src.veto_simulator import VetoSimulator, ActionType, VetoState
from src.veto_analysis import VetoTendencyAnalyzer

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
                model = PowerScoreModel().fit(df)
                return df, model
        except Exception:
            pass

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        model = PowerScoreModel().fit(df)
        return df, model

    return pd.DataFrame(), None


@st.cache_resource
def load_veto_analyzer():
    """Loads and caches the VetoTendencyAnalyzer instance."""
    return VetoTendencyAnalyzer()


# Header
st.title("🎯 VCT 2026 Map Analytics & Veto Simulator")
st.caption("Opponent-adjusted Power Scores, 60-day recency decay (75% retention), and BO3 'Be The GM' engine.")

df, model = load_data_and_fit_model()
veto_analyzer = load_veto_analyzer()

if df.empty or model is None:
    st.warning("⚠️ No match data found. Please execute the data pipeline first.")
    if st.button("🚀 Run Data Pipeline Now"):
        with st.spinner("Downloading VCT DuckDB and processing 2026 maps..."):
            from src.data_pipeline import run_pipeline
            run_pipeline()
            st.success("Pipeline complete! Refreshing...")
            st.rerun()
    st.stop()

teams_by_league = get_teams_by_league()
team_leagues = get_team_leagues()
all_teams_list = sorted(model.teams) if model else []

TEAM_ALIASES = {
    "edg": "EDward Gaming",
    "fnc": "FNATIC",
    "sen": "Sentinels",
    "prx": "Paper Rex",
    "100t": "100 Thieves",
    "th": "Team Heretics",
    "tl": "Team Liquid",
    "vit": "Team Vitality",
    "kc": "Karmine Corp",
    "c9": "Cloud9",
    "eg": "Evil Geniuses",
    "kru": "KRÜ Esports",
    "blg": "Guangzhou Huadu Bilibili Gaming",
    "fpx": "FunPlus Phoenix",
    "te": "Trace Esports",
    "jdg": "JD Gaming",
    "tec": "Wuxi Titan Esports Club",
    "ag": "All Gamers",
    "tyl": "TYLOO",
    "rrq": "Rex Regum Qeon",
    "gen": "Gen.G",
    "geng": "Gen.G",
    "ts": "Team Secret",
    "dfm": "DetonatioN FocusMe",
    "ns": "Nongshim RedForce",
    "bbl": "BBL Esports",
    "m8": "Gentle Mates",
    "gx": "GIANTX",
    "fut": "FUT Esports",
    "drg": "Dragon Ranger Gaming",
    "kbg": "KeepBest Gaming",
    "xlg": "Xi Lai Gaming",
    "navi": "Natus Vincere",
    "nv": "Natus Vincere",
}


def render_team_selector(
    league_label: str,
    team_label: str,
    league_key: str,
    team_key: str,
    default_team: str,
    teams_by_league: dict,
    all_teams_list: list,
    team_leagues: dict,
) -> str:
    """Renders league dropdown, team dropdown, and a nearby search button with quick-select and alias matching."""
    pending_key = f"pending_select_{team_key}"
    if pending_key in st.session_state:
        chosen = st.session_state.pop(pending_key)
        primary_league = team_leagues.get(chosen)
        if primary_league and primary_league in ["Americas", "EMEA", "Pacific", "China"]:
            st.session_state[league_key] = primary_league
        else:
            st.session_state[league_key] = "All Leagues"
        st.session_state[team_key] = chosen
        st.session_state[f"search_input_{team_key}"] = ""

    league = st.selectbox(
        league_label,
        options=["Americas", "EMEA", "Pacific", "China", "Teams at Champions 2026", "All Leagues"],
        index=0,
        key=league_key,
    )
    teams_opts = teams_by_league.get(league, all_teams_list) if league != "All Leagues" else all_teams_list
    if default_team in teams_opts:
        def_idx = teams_opts.index(default_team)
    elif len(teams_opts) > 1:
        def_idx = 1
    else:
        def_idx = 0

    if team_key in st.session_state and st.session_state[team_key] not in teams_opts:
        st.session_state[team_key] = teams_opts[def_idx]

    col_sel, col_search = st.columns([3.8, 1.2], vertical_alignment="bottom")
    with col_sel:
        selected_team = st.selectbox(
            team_label,
            options=teams_opts,
            index=def_idx,
            key=team_key,
        )

    def _select_team(chosen: str):
        st.session_state[pending_key] = chosen
        st.rerun()

    with col_search:
        with st.popover("🔍 Search", help=f"Search and select {team_label}", use_container_width=True):
            st.markdown(f"**🔍 Search Team**")
            search_query = st.text_input(
                "Filter teams:",
                key=f"search_input_{team_key}",
                placeholder="Type name or tag (SEN, FNC, EDG)...",
            ).strip()

            q = search_query.lower()
            if q:
                matches = [t for t in all_teams_list if q in t.lower()]
                if q in TEAM_ALIASES:
                    alias_target = TEAM_ALIASES[q]
                    matching_alias = [t for t in all_teams_list if t.lower() == alias_target.lower() or alias_target.lower() in t.lower()]
                    for at in matching_alias:
                        if at in matches:
                            matches.remove(at)
                        matches.insert(0, at)

                if not matches:
                    st.warning("No teams found matching your search.")
                else:
                    st.caption(f"{len(matches)} team{'s' if len(matches) != 1 else ''} found")
                    for match in matches[:15]:
                        league_tag = team_leagues.get(match, "Other")
                        if st.button(f"{match}  •  {league_tag}", key=f"btn_{team_key}_{match}", use_container_width=True):
                            _select_team(match)
                    if len(matches) > 15:
                        st.caption(f"Showing first 15 of {len(matches)} matches. Type more letters to refine.")
            else:
                st.caption("Type a team name or tag (e.g. SEN, FNC, EDG) and press Enter to search.")

    return selected_team

# Navigation tabs
tab_rankings, tab_h2h, tab_picklist, tab_gm = st.tabs([
    "🏆 Map Power Rankings",
    "⚔️ Head-to-Head Matchup",
    "📋 Strategic Picklist",
    "🎮 Veto Simulator & \"Be the GM\" mode",
])

# ====================================================================
# TAB 1: MAP POWER RANKINGS
# ====================================================================
with tab_rankings:
    st.header("🏆 2026 VCT Map Power Rankings")
    st.markdown("""
    Power Scores (0–100) are evaluated on every 2026 match using **round differentials**, 
    **Bayesian shrinkage**, **60-day exponential recency decay (75% retention)**, and **two-degree opponent strength adjustment**.
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
            options=["All Leagues", "Teams at Champions 2026", "Americas", "EMEA", "Pacific", "China"],
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
            color_continuous_scale="RdYlGn",
            range_color=[20, 80],
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
                display_df.style.background_gradient(subset=["Power Score (0-100)"], cmap="RdYlGn", vmin=30, vmax=70),
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
        team_a = render_team_selector(
            league_label="Select League A:",
            team_label="Select Team A:",
            league_key="h2h_league_a",
            team_key="h2h_team_a",
            default_team="G2 Esports",
            teams_by_league=teams_by_league,
            all_teams_list=all_teams_list,
            team_leagues=team_leagues,
        )

    with col_t2:
        st.subheader("Team B")
        team_b = render_team_selector(
            league_label="Select League B:",
            team_label="Select Team B:",
            league_key="h2h_league_b",
            team_key="h2h_team_b",
            default_team="Sentinels",
            teams_by_league=teams_by_league,
            all_teams_list=all_teams_list,
            team_leagues=team_leagues,
        )

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

        # Veto & Permaban Comparison
        st.subheader("🎯 Veto Habits & Permaban Profile (Past 90 Days)")
        vcol_a, vcol_b = st.columns(2)
        prof_h2h_a = veto_analyzer.get_profile(team_a)
        prof_h2h_b = veto_analyzer.get_profile(team_b)
        with vcol_a:
            st.markdown(f"**{team_a} Veto Habits** ({prof_h2h_a.total_series if prof_h2h_a else 0} series in past 90d):")
            if prof_h2h_a and prof_h2h_a.permaban:
                st.markdown(f"- **Permaban:** <span class='ban-badge'>🔒 {prof_h2h_a.permaban} ({prof_h2h_a.permaban_rate*100:.0f}%)</span>", unsafe_allow_html=True)
            elif prof_h2h_a and prof_h2h_a.insufficient_sample:
                st.markdown(f"- **Top First-Cycle Ban:** {prof_h2h_a.top_ban} ({prof_h2h_a.top_ban_rate*100:.0f}%) *(Sample size too small: {prof_h2h_a.first_cycle_bans.get(prof_h2h_a.top_ban, 0)}/{prof_h2h_a.total_series} series in past 90d, min 3 required)*")
            elif prof_h2h_a and prof_h2h_a.top_ban:
                st.markdown(f"- **Top First-Cycle Ban:** {prof_h2h_a.top_ban} ({prof_h2h_a.top_ban_rate*100:.0f}%) *(No 80% permaban in past 90d)*")
            if prof_h2h_a and prof_h2h_a.comfort_pick:
                st.markdown(f"- **Comfort Pick:** <span class='pick-badge'>🎯 {prof_h2h_a.comfort_pick} ({prof_h2h_a.comfort_pick_rate*100:.0f}%)</span>", unsafe_allow_html=True)
        with vcol_b:
            st.markdown(f"**{team_b} Veto Habits** ({prof_h2h_b.total_series if prof_h2h_b else 0} series in past 90d):")
            if prof_h2h_b and prof_h2h_b.permaban:
                st.markdown(f"- **Permaban:** <span class='ban-badge'>🔒 {prof_h2h_b.permaban} ({prof_h2h_b.permaban_rate*100:.0f}%)</span>", unsafe_allow_html=True)
            elif prof_h2h_b and prof_h2h_b.insufficient_sample:
                st.markdown(f"- **Top First-Cycle Ban:** {prof_h2h_b.top_ban} ({prof_h2h_b.top_ban_rate*100:.0f}%) *(Sample size too small: {prof_h2h_b.first_cycle_bans.get(prof_h2h_b.top_ban, 0)}/{prof_h2h_b.total_series} series in past 90d, min 3 required)*")
            elif prof_h2h_b and prof_h2h_b.top_ban:
                st.markdown(f"- **Top First-Cycle Ban:** {prof_h2h_b.top_ban} ({prof_h2h_b.top_ban_rate*100:.0f}%) *(No 80% permaban in past 90d)*")
            if prof_h2h_b and prof_h2h_b.comfort_pick:
                st.markdown(f"- **Comfort Pick:** <span class='pick-badge'>🎯 {prof_h2h_b.comfort_pick} ({prof_h2h_b.comfort_pick_rate*100:.0f}%)</span>", unsafe_allow_html=True)

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
# TAB 3: STRATEGIC MAP PICKLIST
# ====================================================================
with tab_picklist:
    st.header("📋 Strategic Map Picklist")
    st.markdown("""
    Evaluate the optimal map pick order for any team against a specific opponent based on **Power Score differentials** ($\Delta PS = PS_{\\text{Focus}} - PS_{\\text{Opponent}}$).
    
    If the focus team has an identified **Permaban** (banned $\ge 80\%$ in first cycle in the past 90 days, min 3 series), that map is automatically placed at the **bottom** of the picklist, regardless of its score difference.
    """)

    teams_by_league = get_teams_by_league()
    all_teams_list = sorted(model.teams)

    col_pl_a, col_pl_b = st.columns(2)
    with col_pl_a:
        st.subheader("Focus Team (Viewing Picklist)")
        pl_team_a = render_team_selector(
            league_label="Select Focus League:",
            team_label="Select Focus Team:",
            league_key="pl_league_a",
            team_key="pl_a",
            default_team="G2 Esports",
            teams_by_league=teams_by_league,
            all_teams_list=all_teams_list,
            team_leagues=team_leagues,
        )
        prof_pl_a = veto_analyzer.get_profile(pl_team_a)
        if prof_pl_a:
            if prof_pl_a.permaban:
                st.markdown(f"<div style='margin-top: 4px;'><span class='ban-badge'>🔒 Permaban: {prof_pl_a.permaban} ({prof_pl_a.permaban_rate*100:.0f}%)</span></div>", unsafe_allow_html=True)
            elif prof_pl_a.insufficient_sample:
                st.caption(f"Top Ban: {prof_pl_a.top_ban} ({prof_pl_a.top_ban_rate*100:.0f}%) • Sample size too small ({prof_pl_a.first_cycle_bans.get(prof_pl_a.top_ban, 0)}/{prof_pl_a.total_series} series in past 90d, min 3 required)")
            elif prof_pl_a.top_ban:
                st.caption(f"Top Ban: {prof_pl_a.top_ban} ({prof_pl_a.top_ban_rate*100:.0f}%) • No 80% permaban")
            if prof_pl_a.comfort_pick:
                st.caption(f"Comfort Pick: {prof_pl_a.comfort_pick} ({prof_pl_a.comfort_pick_rate*100:.0f}%)")

    with col_pl_b:
        st.subheader("Opponent Team")
        pl_team_b = render_team_selector(
            league_label="Select Opponent League:",
            team_label="Select Opponent Team:",
            league_key="pl_league_b",
            team_key="pl_b",
            default_team="Sentinels",
            teams_by_league=teams_by_league,
            all_teams_list=all_teams_list,
            team_leagues=team_leagues,
        )
        prof_pl_b = veto_analyzer.get_profile(pl_team_b)
        if prof_pl_b:
            if prof_pl_b.permaban:
                st.markdown(f"<div style='margin-top: 4px;'><span class='ban-badge'>🔒 Permaban: {prof_pl_b.permaban} ({prof_pl_b.permaban_rate*100:.0f}%)</span></div>", unsafe_allow_html=True)
            elif prof_pl_b.insufficient_sample:
                st.caption(f"Top Ban: {prof_pl_b.top_ban} ({prof_pl_b.top_ban_rate*100:.0f}%) • Sample size too small ({prof_pl_b.first_cycle_bans.get(prof_pl_b.top_ban, 0)}/{prof_pl_b.total_series} series in past 90d, min 3 required)")
            elif prof_pl_b.top_ban:
                st.caption(f"Top Ban: {prof_pl_b.top_ban} ({prof_pl_b.top_ban_rate*100:.0f}%) • No 80% permaban")
            if prof_pl_b.comfort_pick:
                st.caption(f"Comfort Pick: {prof_pl_b.comfort_pick} ({prof_pl_b.comfort_pick_rate*100:.0f}%)")

    if pl_team_a == pl_team_b:
        st.info("Select two different teams to view the strategic picklist.")
    else:
        picklist = veto_analyzer.generate_picklist(pl_team_a, pl_team_b, model, ACTIVE_MAP_POOL)

        # Plotly Bar Chart: Difference in Power Score
        # Reverse list so Rank 1 appears at the top of a horizontal bar chart
        chart_data = list(reversed(picklist))
        map_labels = [
            f"🔒 {it['map']} (PERMABAN)" if it['is_permaban'] else f"#{it['rank']} {it['map']}"
            for it in chart_data
        ]
        diff_values = [it['diff'] for it in chart_data]
        bar_colors = [
            "#7f1d1d" if it['is_permaban'] else ("#00f2fe" if it['diff'] >= 0 else "#ff4b4b")
            for it in chart_data
        ]

        fig_pl = go.Figure()
        fig_pl.add_trace(go.Bar(
            y=map_labels,
            x=diff_values,
            orientation='h',
            marker=dict(color=bar_colors),
            text=[f"{v:+0.1f}" for v in diff_values],
            textposition='auto',
        ))
        fig_pl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f2f6"),
            title=f"Strategic Map Priority for {pl_team_a} vs {pl_team_b} (Δ Power Score)",
            xaxis_title=f"Power Score Advantage ({pl_team_a} minus {pl_team_b})",
            yaxis_title="Map Priority",
            height=380,
            xaxis=dict(zeroline=True, zerolinecolor="#64748b", zerolinewidth=2),
        )

        col_pl_chart, col_pl_table = st.columns([1, 1])
        with col_pl_chart:
            st.plotly_chart(fig_pl, use_container_width=True)

        with col_pl_table:
            st.subheader(f"Ordered Map Picklist")
            pl_df = pd.DataFrame([{
                "Rank": f"#{it['rank']}",
                "Map": it["map"],
                f"{pl_team_a}": it["focus_score"],
                f"{pl_team_b}": it["opponent_score"],
                "Δ Diff": f"{it['diff']:+0.1f}",
                "Win Prob %": f"{it['win_prob']}%",
                "Strategic Assessment": it["note"]
            } for it in picklist])

            st.dataframe(
                pl_df,
                use_container_width=True,
                height=350,
                hide_index=True,
            )

# ====================================================================
# TAB 4: VETO SIMULATOR & BE THE GM MODE
# ====================================================================
with tab_gm:
    st.header("🎮 Veto Simulator & \"Be the GM\" Mode")
    st.markdown("""
    Simulate official VCT BO3 map vetoes with the 7-map pool: 
    **Abyss, Ascent, Haven, Lotus, Split, Summit, Sunset**.
    """)

    teams_by_league = get_teams_by_league()
    all_teams_list = sorted(model.teams)

    col_sim_a, col_sim_b = st.columns(2)
    with col_sim_a:
        st.subheader("Team A (1st Ban & 1st Pick)")
        gm_team_a = render_team_selector(
            league_label="Select League A:",
            team_label="Select Team A:",
            league_key="gm_league_a",
            team_key="gm_a",
            default_team="G2 Esports",
            teams_by_league=teams_by_league,
            all_teams_list=all_teams_list,
            team_leagues=team_leagues,
        )
        prof_a = veto_analyzer.get_profile(gm_team_a)
        if prof_a:
            if prof_a.permaban:
                st.markdown(f"<div style='margin-top: 4px;'><span class='ban-badge'>🔒 Permaban: {prof_a.permaban} ({prof_a.permaban_rate*100:.0f}%)</span></div>", unsafe_allow_html=True)
            elif prof_a.insufficient_sample:
                st.caption(f"Top Ban: {prof_a.top_ban} ({prof_a.top_ban_rate*100:.0f}%) • Sample size too small ({prof_a.first_cycle_bans.get(prof_a.top_ban, 0)}/{prof_a.total_series} series in past 90d, min 3 required)")
            elif prof_a.top_ban:
                st.caption(f"Top Ban: {prof_a.top_ban} ({prof_a.top_ban_rate*100:.0f}%) • No 80% permaban")
            if prof_a.comfort_pick:
                st.caption(f"Comfort Pick: {prof_a.comfort_pick} ({prof_a.comfort_pick_rate*100:.0f}%)")

    with col_sim_b:
        st.subheader("Team B (2nd Ban & 2nd Pick)")
        gm_team_b = render_team_selector(
            league_label="Select League B:",
            team_label="Select Team B:",
            league_key="gm_league_b",
            team_key="gm_b",
            default_team="Sentinels",
            teams_by_league=teams_by_league,
            all_teams_list=all_teams_list,
            team_leagues=team_leagues,
        )
        prof_b = veto_analyzer.get_profile(gm_team_b)
        if prof_b:
            if prof_b.permaban:
                st.markdown(f"<div style='margin-top: 4px;'><span class='ban-badge'>🔒 Permaban: {prof_b.permaban} ({prof_b.permaban_rate*100:.0f}%)</span></div>", unsafe_allow_html=True)
            elif prof_b.insufficient_sample:
                st.caption(f"Top Ban: {prof_b.top_ban} ({prof_b.top_ban_rate*100:.0f}%) • Sample size too small ({prof_b.first_cycle_bans.get(prof_b.top_ban, 0)}/{prof_b.total_series} series in past 90d, min 3 required)")
            elif prof_b.top_ban:
                st.caption(f"Top Ban: {prof_b.top_ban} ({prof_b.top_ban_rate*100:.0f}%) • No 80% permaban")
            if prof_b.comfort_pick:
                st.caption(f"Comfort Pick: {prof_b.comfort_pick} ({prof_b.comfort_pick_rate*100:.0f}%)")

    sim_mode = st.radio(
        "Select Simulation Mode:",
        ["🤖 Fully Automated Simulation (AI vs AI)", "👔 'Be The GM' Interactive Mode"],
        horizontal=True,
    )

    simulator = VetoSimulator(model=model, map_pool=ACTIVE_MAP_POOL, analyzer=veto_analyzer)

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
                    is_pban = prompt["action_type"] == ActionType.BAN and veto_analyzer.is_permaban(gm_seat, map_name)
                    pban_tag = "🔒 " if is_pban else ""
                    btn_label = f"{pban_tag}{map_name}\n({prob_user*100:.0f}%)"
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


