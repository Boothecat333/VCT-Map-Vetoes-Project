"""
Power Score statistical model for VCT 2026.
Calculates opponent-adjusted, recency-weighted map power scores (0-100)
incorporating round differentials, Bayesian shrinkage, and iterative opponent quality.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Active competitive map pool
ACTIVE_MAP_POOL = ["Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset"]
DEFAULT_DECAY_DAYS = 60.0       # 60-day decay interval
DEFAULT_RETENTION_RATE = 0.75   # 75% weight retained after 60 days
DEFAULT_PRIOR_WEIGHT = 2.5      # Bayesian shrinkage pseudo-matches
DEFAULT_OPPONENT_ALPHA = 0.5    # Opponent adjustment factor
DEFAULT_K = 0.95                # Sigmoid steepness (spreads Power Score across 0-100)


@dataclass
class TeamMapRating:
    team: str
    map_name: str
    power_score: float         # Scaled 0 - 100
    raw_rating: float          # Unbounded opponent-adjusted rating
    base_rating: float         # Rating before opponent adjustment
    maps_played: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_round_diff: float
    effective_matches: float   # Sum of recency weights


class PowerScoreModel:
    """
    Computes VCT map ratings using:
    1. Margin of victory (round differential with diminishing returns)
    2. Exponential recency decay (75% weight after 60 days)
    3. Bayesian shrinkage towards league average
    4. Two-degree opponent strength adjustment
    5. Normalization to 0–100 Power Score
    """

    def __init__(
        self,
        decay_days: float = DEFAULT_DECAY_DAYS,
        retention_rate: float = DEFAULT_RETENTION_RATE,
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
        opponent_alpha: float = DEFAULT_OPPONENT_ALPHA,
        k: float = DEFAULT_K,
        reference_date: Optional[date] = None,
    ):
        self.decay_days = decay_days
        self.retention_rate = retention_rate
        self.prior_weight = prior_weight
        self.opponent_alpha = opponent_alpha
        self.k = k
        self.reference_date = reference_date
        self.ratings: Dict[Tuple[str, str], TeamMapRating] = {}
        self.teams: List[str] = []
        self.maps: List[str] = []

    def compute_recency_weight(self, match_date: date, ref_date: date) -> float:
        """Computes exponential decay weight: w = retention_rate^(delta_days / decay_days)."""
        delta_days = (ref_date - match_date).days
        if delta_days <= 0:
            return 1.0
        return float(self.retention_rate ** (delta_days / self.decay_days))

    def _transform_round_differential(self, round_diff: int) -> float:
        """Applies sub-linear transformation to round diff to prevent extreme blowouts from skewing."""
        sign = 1.0 if round_diff >= 0 else -1.0
        return float(sign * (abs(round_diff) ** 0.85))

    def fit(self, df: pd.DataFrame) -> "PowerScoreModel":
        """
        Fits the Power Score model on a DataFrame of map observations.
        Required columns: ['date', 'team', 'opponent', 'map', 'rounds_for', 'rounds_against', 'result']
        """
        if df.empty:
            return self

        # Prepare clean working copy
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"]).dt.date
        ref_date = self.reference_date or data["date"].max()

        data["weight"] = data["date"].apply(lambda d: self.compute_recency_weight(d, ref_date))
        data["round_diff"] = data["rounds_for"] - data["rounds_against"]
        data["transformed_diff"] = data["round_diff"].apply(self._transform_round_differential)

        teams = sorted(data["team"].unique().tolist())
        maps = sorted(data["map"].unique().tolist())
        self.teams = teams
        self.maps = maps

        # Step 1: Base Rating R0 with Bayesian shrinkage
        # R0(T, m) = sum(w * transformed_diff) / (sum(w) + prior_weight)
        base_ratings: Dict[Tuple[str, str], float] = {}
        match_stats: Dict[Tuple[str, str], dict] = {}

        for (team, map_name), group in data.groupby(["team", "map"]):
            w_sum = group["weight"].sum()
            weighted_diff = (group["weight"] * group["transformed_diff"]).sum()
            r0 = weighted_diff / (w_sum + self.prior_weight)
            base_ratings[(team, map_name)] = r0

            wins = int((group["result"] == "W").sum())
            losses = int((group["result"] == "L").sum())
            total = len(group)
            match_stats[(team, map_name)] = {
                "maps_played": total,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round(100.0 * wins / total, 1) if total > 0 else 0.0,
                "avg_round_diff": round(float(group["round_diff"].mean()), 2),
                "effective_matches": round(float(w_sum), 2),
            }

        # Ensure all combinations exist in base_ratings with 0 if unplayed
        for t in teams:
            for m in maps:
                if (t, m) not in base_ratings:
                    base_ratings[(t, m)] = 0.0
                    match_stats[(t, m)] = {
                        "maps_played": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_rate_pct": 0.0,
                        "avg_round_diff": 0.0,
                        "effective_matches": 0.0,
                    }

        # Step 2 & 3: Opponent-Adjusted Iterations (Degree 1 and Degree 2)
        # Match perf = transformed_diff + alpha * opponent_rating
        curr_ratings = base_ratings.copy()

        for iteration in range(2):
            next_ratings: Dict[Tuple[str, str], float] = {}
            for (team, map_name), group in data.groupby(["team", "map"]):
                w_sum = group["weight"].sum()
                adj_diffs = []
                for _, row in group.iterrows():
                    opp = row["opponent"]
                    opp_r = curr_ratings.get((opp, map_name), 0.0)
                    adj_diff = row["transformed_diff"] + (self.opponent_alpha * opp_r)
                    adj_diffs.append(row["weight"] * adj_diff)
                
                r_adj = sum(adj_diffs) / (w_sum + self.prior_weight)
                next_ratings[(team, map_name)] = r_adj

            for t in teams:
                for m in maps:
                    if (t, m) not in next_ratings:
                        next_ratings[(t, m)] = 0.0
            curr_ratings = next_ratings

        # Step 4: Scale to 0-100 Power Score
        # Using a logistic sigmoid centered at 50 with steepness factor self.k
        self.ratings = {}
        for (team, map_name), raw_r in curr_ratings.items():
            ps = 100.0 / (1.0 + np.exp(-self.k * raw_r))
            stats = match_stats.get((team, map_name), {
                "maps_played": 0, "wins": 0, "losses": 0,
                "win_rate_pct": 0.0, "avg_round_diff": 0.0, "effective_matches": 0.0
            })
            self.ratings[(team, map_name)] = TeamMapRating(
                team=team,
                map_name=map_name,
                power_score=round(float(ps), 1),
                raw_rating=round(float(raw_r), 3),
                base_rating=round(float(base_ratings.get((team, map_name), 0.0)), 3),
                maps_played=stats["maps_played"],
                wins=stats["wins"],
                losses=stats["losses"],
                win_rate_pct=stats["win_rate_pct"],
                avg_round_diff=stats["avg_round_diff"],
                effective_matches=stats["effective_matches"],
            )

        return self

    def get_rating(self, team: str, map_name: str) -> TeamMapRating:
        """Retrieves rating for team on map, returning unplayed baseline if not found."""
        if (team, map_name) in self.ratings:
            return self.ratings[(team, map_name)]
        return TeamMapRating(
            team=team,
            map_name=map_name,
            power_score=50.0,
            raw_rating=0.0,
            base_rating=0.0,
            maps_played=0,
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            avg_round_diff=0.0,
            effective_matches=0.0,
        )

    def get_team_ratings(self, team: str, map_pool: Optional[List[str]] = None) -> List[TeamMapRating]:
        """Returns ratings for all maps for a given team, optionally filtered to a map pool."""
        target_maps = map_pool or ACTIVE_MAP_POOL
        return [self.get_rating(team, m) for m in target_maps]

    def predict_map_win_probability(self, team_a: str, team_b: str, map_name: str) -> float:
        """
        Calculates win probability for Team A vs Team B on a specific map.
        Uses logistic curve based on Power Score difference.
        """
        ps_a = self.get_rating(team_a, map_name).power_score
        ps_b = self.get_rating(team_b, map_name).power_score
        diff = ps_a - ps_b
        prob_a = 1.0 / (1.0 + (10.0 ** (-diff / 40.0)))
        return round(float(prob_a), 3)

    def to_dataframe(self, map_pool: Optional[List[str]] = None) -> pd.DataFrame:
        """Converts all computed ratings into a clean tabular DataFrame for Streamlit display."""
        target_maps = map_pool or self.maps or ACTIVE_MAP_POOL
        rows = []
        for (team, map_name), r in self.ratings.items():
            if map_name in target_maps:
                rows.append({
                    "team": team,
                    "map": map_name,
                    "power_score": r.power_score,
                    "record": f"{r.wins}-{r.losses}",
                    "win_rate_pct": r.win_rate_pct,
                    "avg_round_diff": r.avg_round_diff,
                    "maps_played": r.maps_played,
                    "effective_matches": r.effective_matches,
                })
        return pd.DataFrame(rows)
