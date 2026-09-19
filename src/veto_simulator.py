"""
BO3 Veto Simulator and 'Be The GM' Engine for VCT 2026.
Implements official VCT BO3 pick/ban sequences with strategic recommendation logic
and series outcome projections.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
from src.power_score import PowerScoreModel, ACTIVE_MAP_POOL
from src.veto_analysis import VetoTendencyAnalyzer


class ActionType(str, Enum):
    BAN = "BAN"
    PICK = "PICK"
    DECIDER = "DECIDER"


@dataclass
class VetoStep:
    step_number: int
    actor_team: str           # "Team A" or "Team B" or team name
    action_type: ActionType
    map_name: Optional[str] = None
    side_choice_team: Optional[str] = None
    reasoning: Optional[str] = None
    win_prob_team_a: Optional[float] = None


@dataclass
class VetoState:
    team_a: str
    team_b: str
    map_pool: List[str]
    available_maps: List[str] = field(default_factory=list)
    banned_maps: Dict[str, str] = field(default_factory=dict)  # map -> banned_by_team
    picked_maps: List[Tuple[str, str]] = field(default_factory=list)  # (map, picked_by_team)
    decider_map: Optional[str] = None
    history: List[VetoStep] = field(default_factory=list)
    current_step_index: int = 0
    is_complete: bool = False

    def __post_init__(self):
        if not self.available_maps:
            self.available_maps = list(self.map_pool)


class VetoSimulator:
    """
    Simulates VCT BO3 Map Vetoes:
    Turn 1: Team A Ban
    Turn 2: Team B Ban
    Turn 3: Team A Pick (Map 1)
    Turn 4: Team B Pick (Map 2)
    Turn 5: Team A Ban
    Turn 6: Team B Ban
    Turn 7: Remaining Decider Map (Map 3)
    """

    STEPS_DEFINITION = [
        (1, "A", ActionType.BAN, "Bans 1st map"),
        (2, "B", ActionType.BAN, "Bans 1st map"),
        (3, "A", ActionType.PICK, "Picks Map 1"),
        (4, "B", ActionType.PICK, "Picks Map 2"),
        (5, "A", ActionType.BAN, "Bans 2nd map"),
        (6, "B", ActionType.BAN, "Bans 2nd map"),
        (7, "NONE", ActionType.DECIDER, "Decider Map"),
    ]

    def __init__(
        self,
        model: PowerScoreModel,
        map_pool: Optional[List[str]] = None,
        analyzer: Optional[VetoTendencyAnalyzer] = None,
    ):
        self.model = model
        self.map_pool = list(map_pool or ACTIVE_MAP_POOL)
        self.analyzer = analyzer or VetoTendencyAnalyzer()

    def create_session(self, team_a: str, team_b: str) -> VetoState:
        """Initializes a new veto session for Team A vs Team B."""
        return VetoState(
            team_a=team_a,
            team_b=team_b,
            map_pool=list(self.map_pool),
            available_maps=list(self.map_pool),
        )

    def get_current_prompt(self, state: VetoState) -> Optional[dict]:
        """Returns details about the next action required in the veto."""
        if state.is_complete or state.current_step_index >= len(self.STEPS_DEFINITION):
            return None

        step_num, team_code, action_type, desc = self.STEPS_DEFINITION[state.current_step_index]
        actor = state.team_a if team_code == "A" else (state.team_b if team_code == "B" else "Decider")

        return {
            "step_number": step_num,
            "team_code": team_code,
            "actor": actor,
            "action_type": action_type,
            "description": desc,
            "available_maps": list(state.available_maps),
        }

    def recommend_action(self, state: VetoState) -> Tuple[str, str]:
        """
        Recommends optimal map selection for the current actor based on Permabans and Power Scores.
        Returns: (recommended_map, rationale_explanation)
        """
        prompt = self.get_current_prompt(state)
        if not prompt or not state.available_maps:
            return ("", "Veto complete")

        step_num = prompt["step_number"]
        actor = prompt["actor"]
        action_type = prompt["action_type"]

        if action_type == ActionType.DECIDER or len(state.available_maps) == 1:
            decider = state.available_maps[0]
            p_a = self.model.predict_map_win_probability(state.team_a, state.team_b, decider)
            return (decider, f"Remaining map becomes the Decider ({round(p_a*100, 1)}% win chance for {state.team_a}).")

        # Evaluate win probabilities for all remaining maps
        map_probs = {}
        for m in state.available_maps:
            prob_a = self.model.predict_map_win_probability(state.team_a, state.team_b, m)
            map_probs[m] = prob_a if actor == state.team_a else (1.0 - prob_a)

        if action_type == ActionType.PICK:
            # Pick highest win probability map
            best_map = max(map_probs.keys(), key=lambda m: map_probs[m])
            best_prob = round(map_probs[best_map] * 100, 1)
            ps_actor = self.model.get_rating(actor, best_map).power_score
            opp = state.team_b if actor == state.team_a else state.team_a
            ps_opp = self.model.get_rating(opp, best_map).power_score
            explanation = (
                f"{actor} holds a {ps_actor:.1f} Power Score on {best_map} vs {opp}'s {ps_opp:.1f}. "
                f"Projected {best_prob}% win probability makes this their strongest pick."
            )
            return (best_map, explanation)

        elif action_type == ActionType.BAN:
            # 1. Check for verified permaban in First Cycle (Turns 1 & 2)
            if step_num in (1, 2) and self.analyzer:
                prof = self.analyzer.get_profile(actor)
                if prof and prof.permaban and prof.permaban in state.available_maps:
                    rate_pct = round((prof.permaban_rate or 0) * 100, 1)
                    explanation = (
                        f"🔒 Permaban: {actor} bans {prof.permaban} in {rate_pct}% "
                        f"of matches ({prof.total_series} series sample). Recommended ban adheres to team philosophy."
                    )
                    return (prof.permaban, explanation)

            # 2. Fallback to lowest win probability map (opponents' greatest advantage)
            worst_map = min(map_probs.keys(), key=lambda m: map_probs[m])
            worst_prob = round(map_probs[worst_map] * 100, 1)
            opp = state.team_b if actor == state.team_a else state.team_a
            ps_opp = self.model.get_rating(opp, worst_map).power_score
            ps_actor = self.model.get_rating(actor, worst_map).power_score

            tendency_note = ""
            if step_num in (1, 2) and self.analyzer:
                prof = self.analyzer.get_profile(actor)
                if prof and prof.top_ban and prof.top_ban in state.available_maps:
                    top_pct = round((prof.top_ban_rate or 0) * 100, 1)
                    tendency_note = f" (Note: {actor}'s top historical ban is {prof.top_ban} at {top_pct}%)."

            explanation = (
                f"Banning {worst_map} removes {opp}'s strong {ps_opp:.1f} Power Score "
                f"({actor} is {ps_actor:.1f}). Protects against an estimated {100-worst_prob:.1f}% opponent map edge.{tendency_note}"
            )
            return (worst_map, explanation)

        return (state.available_maps[0], "Neutral selection")

    def execute_step(self, state: VetoState, chosen_map: str, reasoning: Optional[str] = None) -> VetoState:
        """Applies a user or AI chosen map to the current veto step."""
        if state.is_complete or chosen_map not in state.available_maps:
            return state

        step_num, team_code, action_type, _ = self.STEPS_DEFINITION[state.current_step_index]
        actor = state.team_a if team_code == "A" else (state.team_b if team_code == "B" else "Decider")

        win_prob_a = self.model.predict_map_win_probability(state.team_a, state.team_b, chosen_map)
        side_picker = state.team_b if actor == state.team_a else state.team_a

        step = VetoStep(
            step_number=step_num,
            actor_team=actor,
            action_type=action_type,
            map_name=chosen_map,
            side_choice_team=side_picker if action_type == ActionType.PICK else None,
            reasoning=reasoning,
            win_prob_team_a=win_prob_a,
        )
        state.history.append(step)
        state.available_maps.remove(chosen_map)

        if action_type == ActionType.BAN:
            state.banned_maps[chosen_map] = actor
        elif action_type == ActionType.PICK:
            state.picked_maps.append((chosen_map, actor))

        state.current_step_index += 1

        # Check if only 1 map remains -> Automatically assign Decider
        if state.current_step_index == 6 and len(state.available_maps) == 1:
            decider = state.available_maps[0]
            decider_prob_a = self.model.predict_map_win_probability(state.team_a, state.team_b, decider)
            decider_step = VetoStep(
                step_number=7,
                actor_team="Decider",
                action_type=ActionType.DECIDER,
                map_name=decider,
                side_choice_team="Coin Toss / Higher Seed",
                reasoning=f"Final unbanned map remaining from pool",
                win_prob_team_a=decider_prob_a,
            )
            state.history.append(decider_step)
            state.decider_map = decider
            state.available_maps.remove(decider)
            state.current_step_index = 7
            state.is_complete = True

        elif state.current_step_index >= len(self.STEPS_DEFINITION):
            state.is_complete = True

        return state

    def simulate_full_veto(self, team_a: str, team_b: str) -> VetoState:
        """Simulates a complete automated BO3 veto between two teams using optimal AI decisions."""
        state = self.create_session(team_a, team_b)
        while not state.is_complete:
            rec_map, rec_reason = self.recommend_action(state)
            state = self.execute_step(state, rec_map, rec_reason)
        return state

    def calculate_series_odds(self, state: VetoState) -> Dict[str, float]:
        """
        Calculates projected BO3 series win probabilities given the maps selected.
        If veto is not finished, returns odds over picked maps + expected remaining pool.
        """
        if not state.is_complete or len(state.picked_maps) < 2 or not state.decider_map:
            return {"team_a_pct": 50.0, "team_b_pct": 50.0}

        map1 = state.picked_maps[0][0]
        map2 = state.picked_maps[1][0]
        map3 = state.decider_map

        p1 = self.model.predict_map_win_probability(state.team_a, state.team_b, map1)
        p2 = self.model.predict_map_win_probability(state.team_a, state.team_b, map2)
        p3 = self.model.predict_map_win_probability(state.team_a, state.team_b, map3)

        # Team A wins 2-0: P1 * P2
        # Team A wins 2-1: P1 * (1-P2) * P3 + (1-P1) * P2 * P3
        prob_a_series = (p1 * p2) + (p1 * (1.0 - p2) * p3) + ((1.0 - p1) * p2 * p3)
        prob_b_series = 1.0 - prob_a_series

        return {
            "team_a_pct": round(prob_a_series * 100, 1),
            "team_b_pct": round(prob_b_series * 100, 1),
            "map_1": {"map": map1, "picker": state.picked_maps[0][1], "team_a_win_pct": round(p1 * 100, 1)},
            "map_2": {"map": map2, "picker": state.picked_maps[1][1], "team_a_win_pct": round(p2 * 100, 1)},
            "map_3": {"map": map3, "picker": "Decider", "team_a_win_pct": round(p3 * 100, 1)},
        }
