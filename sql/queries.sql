-- ====================================================================
-- Portfolio SQL Queries: VCT 2026 Map Analytics
-- Demonstrating aggregation, filtering, window functions, and analytics
-- ====================================================================

-- 1. Team Performance by Map (Record, Win Rate, Round Differential)
-- Analyzes how dominant each team is on each map played in 2026.
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
ORDER BY win_rate_pct DESC, avg_round_margin DESC;

-- 2. Active Map Pool Popularity & Balance
-- Shows overall pick frequency and parity across the map pool in 2026.
SELECT 
    map,
    COUNT(*) / 2 AS total_times_played,
    ROUND(AVG(rounds_for + rounds_against), 2) AS avg_total_rounds,
    SUM(CASE WHEN rounds_for > 13 OR rounds_against > 13 THEN 1 ELSE 0 END) / 2 AS overtime_games
FROM map_results
GROUP BY map
ORDER BY total_times_played DESC;

-- 3. Head-to-Head Comparison Between Two Specific Teams
-- Example: G2 Esports vs Sentinels
SELECT 
    date,
    event,
    map,
    team,
    rounds_for,
    rounds_against,
    opponent,
    result
FROM map_results
WHERE team = 'G2 Esports' AND opponent = 'Sentinels'
ORDER BY date DESC;

-- 4. Recent Form / Rolling Round Margin (SQL Window Function)
-- Calculates 3-game rolling average round margin per team to measure momentum.
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
ORDER BY team, date;

-- 5. Top 10 Most Dominant Map Wins in 2026
SELECT 
    date,
    event,
    team AS winner,
    opponent AS loser,
    map,
    rounds_for || '-' || rounds_against AS final_score,
    round_differential
FROM map_results
WHERE result = 'W'
ORDER BY round_differential DESC, date DESC
LIMIT 10;
