# 🎯 VCT 2026 Map Analytics & BO3 Veto Simulator

An esports data analytics platform and interactive web application for the **Valorant Champions Tour (VCT) 2026** season. 

This project bridges **data engineering (DuckDB, SQLite)**, **applied statistics (Bayesian shrinkage, opponent-adjusted round margins, exponential recency decay)**, and **interactive web visualization (Streamlit, Plotly)** to provide map power rankings, head-to-head match projections, and a strategic **"Be The GM" BO3 Veto Simulator**.

---

## 🏗️ Architecture

```
                    VCT Reference Dataset
                        (vct.duckdb)
                             │
                             ▼ (DuckDB SQL)
                    data_pipeline.py
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       vct_2026_maps.csv            vct_analytics.db
       (Model Training)             (Portfolio SQL Queries)
              │                             │
              ▼                             ▼
       power_score.py                 queries.sql
   (45d Half-Life Decay,                    │
    Opponent Adjustment)                    │
              │                             │
              ▼                             │
      veto_simulator.py                     │
      (BO3 Pick/Ban Engine)                 │
              │                             │
              └──────────────┬──────────────┘
                             ▼
                          app.py
                  (Streamlit Web Application)
```

---

## 🧮 Statistical Methodology: Map Power Score (0–100)

Unlike series-level match scores, this model evaluates **individual map dominance** across all 2026 tier 1 tournaments (Kickoff, Masters, Stage 1, Stage 2, Champions).

### 1. Margin of Victory (Round Differential)
Every map produces two directional observations:
$$\text{Round Differential } (RD) = \text{Rounds For} - \text{Rounds Against}$$
To diminish extreme blowout noise (e.g. 13–1 vs 13–10), a sub-linear transform is applied:
$$\widetilde{RD} = \text{sign}(RD) \cdot |RD|^{0.85}$$

### 2. Exponential Recency Decay (45-Day Half-Life)
Older matches in the season contribute exponentially less than recent matches to account for meta, agent, and form shifts:
$$w_i = 2^{-\frac{\Delta\text{days}}{45}}$$

### 3. Bayesian Shrinkage & Opponent Adjustment (Two-Degree Iteration)
Teams with small sample sizes on a map are regressed towards league baseline using a prior pseudo-weight $C = 3.0$:
$$R_0(T, m) = \frac{\sum_i w_i \cdot \widetilde{RD}_i}{\sum_i w_i + C}$$
Teams then receive credit for beating elite teams on that map, and penalties for dropping maps to weaker opponents:
$$R_1(T, m) = \frac{\sum_i w_i \cdot [\widetilde{RD}_i + \alpha \cdot R_0(\text{Opponent}, m)]}{\sum_i w_i + C}$$
This is iterated across two degrees of separation.

### 4. 0–100 Power Score Normalization
Continuous ratings are converted into an intuitive 0–100 esports Power Score via logistic transformation centered at 50:
$$PS(T, m) = \frac{100}{1 + \exp(-0.25 \cdot R(T, m))}$$
- **80–100**: Elite Map Specialist (Tier 1 dominant pick)
- **60–79**: Strong Map Advantage
- **45–59**: Contested / Balanced Map
- **20–44**: Vulnerable Map
- **0–19**: Permaban Territory

---

## 🎮 BO3 Veto Simulator & "Be The GM"

Configured for the active 7-map competitive pool:
**`Abyss`**, **`Ascent`**, **`Haven`**, **`Lotus`**, **`Split`**, **`Summit`**, **`Sunset`**.

### Standard VCT Pick/Ban Sequence:
1. **Team A Ban** $\to$ 6 maps remain
2. **Team B Ban** $\to$ 5 maps remain
3. **Team A Pick** $\to$ Map 1 selected (Team B chooses side)
4. **Team B Pick** $\to$ Map 2 selected (Team A chooses side)
5. **Team A Ban** $\to$ 2 maps remain
6. **Team B Ban** $\to$ 1 map remains
7. **Decider Map** $\to$ Map 3 selected (Higher seed / coin toss chooses side)

### Features:
- **AI Advisor**: Evaluates expected win probability on every remaining map and recommends strategic bans (removing opponent power maps) and picks.
- **Series Win Odds**: Computes live BO3 series win projection after each pick/ban:
  $$P(\text{Series}) = P_1 P_2 + P_1(1 - P_2)P_3 + (1 - P_1)P_2 P_3$$

---

## 🗄️ SQL Portfolio Queries

The project maintains an optimized SQLite database (`data/vct_analytics.db`) populated with 3,000+ map observations. Key portfolio queries in `sql/queries.sql` demonstrate:
- **Aggregations & Grouping**: Win rates, average round margins, map volume.
- **Filtering & Joins**: Direct head-to-head match histories.
- **Window Functions**: Rolling 3-game round margin momentum (`AVG() OVER (PARTITION BY team ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`).
- **Outlier Analysis**: Top 10 blowout map victories in 2026.

---

## 🚀 Quick Start Guide

### 1. Clone & Setup Virtual Environment
```powershell
# Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Pipeline
Downloads latest daily DuckDB warehouse and populates local SQLite database:
```powershell
python src/data_pipeline.py
```

### 3. Run Test Suite
```powershell
pytest -v
```

### 4. Launch Streamlit Web Application
```powershell
streamlit run app.py
```

---

## 📂 Project Structure
```
VCT-Map-Analytics/
├── app.py                     # Streamlit web application
├── requirements.txt           # Python package dependencies
├── pyproject.toml             # Pytest & package configuration
├── sql/
│   ├── schema.sql             # SQLite relational schema
│   └── queries.sql            # Data analyst portfolio queries
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py       # DuckDB extraction & ETL pipeline
│   ├── database.py            # SQLite connection & query helpers
│   ├── power_score.py         # Opponent-adjusted recency statistical model
│   └── veto_simulator.py      # BO3 pick/ban engine & GM simulator
├── tests/
│   ├── test_data_pipeline.py  # Pipeline & DB tests
│   ├── test_power_score.py    # Statistical model unit tests
│   └── test_veto_simulator.py # Pick/ban logic tests
└── data/
    ├── raw/                   # vct.duckdb (downloaded daily)
    ├── processed/             # vct_2026_maps.csv
    └── vct_analytics.db       # SQLite application database
```