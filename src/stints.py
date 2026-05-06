"""Parse play-by-play data into lineup stints for RAPM modeling.

A stint is a continuous stretch of game time where no substitutions occur.
Each stint records which 10 players are on court and the point differential.
"""

from __future__ import annotations

import re
from itertools import groupby

import pandas as pd
import numpy as np

from src.data import GameData


def _clock_to_seconds(clock_str: str) -> float:
    """Parse ISO 8601 duration clock (e.g. 'PT07M09.00S') to seconds remaining."""
    m = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def _elapsed_seconds(period: int, clock_str: str) -> float:
    """Convert period + game clock to cumulative elapsed seconds."""
    remaining = _clock_to_seconds(clock_str)
    period_length = 12 * 60 if period <= 4 else 5 * 60
    elapsed_in_period = period_length - remaining

    prior = sum(12 * 60 for _ in range(1, min(period, 5)))
    if period > 4:
        prior += sum(5 * 60 for _ in range(5, period))

    return prior + elapsed_in_period


def _apply_sub_batch(subs: list[dict], home_lineup: set, away_lineup: set, home_team_id: int):
    """Apply a batch of simultaneous substitutions to lineups."""
    for sub in subs:
        pid = sub["personId"]
        tid = sub.get("teamId")
        is_home = tid == home_team_id
        lineup = home_lineup if is_home else away_lineup

        if sub["subType"] == "out":
            lineup.discard(pid)
        elif sub["subType"] == "in":
            lineup.add(pid)


def parse_game_stints(game: GameData) -> list[dict]:
    """Parse a single game's CDN play-by-play actions into stints."""
    home_lineup = set(game.home_starters)
    away_lineup = set(game.away_starters)

    if len(home_lineup) != 5 or len(away_lineup) != 5:
        return []

    actions = sorted(game.actions, key=lambda a: a["orderNumber"])

    stints = []
    stint_start = 0.0
    current_period = 1
    home_score = away_score = 0
    stint_start_home = stint_start_away = 0

    # Group substitutions by their timestamp so they're applied atomically
    i = 0
    while i < len(actions):
        action = actions[i]
        period = action["period"]
        clock = action.get("clock", "PT00M00.00S")

        # Update running score
        sh = action.get("scoreHome")
        sa = action.get("scoreAway")
        if sh is not None and sa is not None:
            try:
                home_score = int(sh)
                away_score = int(sa)
            except (ValueError, TypeError):
                pass

        # Period transition
        if period != current_period:
            t = _elapsed_seconds(current_period, "PT00M00.00S")
            duration = t - stint_start
            if duration > 0 and len(home_lineup) == 5 and len(away_lineup) == 5:
                stints.append({
                    "game_id": game.game_id,
                    "home_players": frozenset(home_lineup),
                    "away_players": frozenset(away_lineup),
                    "duration_seconds": duration,
                    "home_points": home_score - stint_start_home,
                    "away_points": away_score - stint_start_away,
                    "margin": (home_score - stint_start_home) - (away_score - stint_start_away),
                })
            current_period = period
            period_clock = "PT12M00.00S" if period <= 4 else "PT05M00.00S"
            stint_start = _elapsed_seconds(period, period_clock)
            stint_start_home = home_score
            stint_start_away = away_score

        # Collect all substitutions at the same (period, clock)
        if action.get("actionType") == "substitution":
            sub_batch = []
            sub_period = period
            sub_clock = clock
            t = _elapsed_seconds(period, clock)

            while i < len(actions) and actions[i].get("actionType") == "substitution" \
                    and actions[i]["period"] == sub_period and actions[i].get("clock") == sub_clock:
                sub_batch.append(actions[i])
                # Update score from sub events too
                sh2 = actions[i].get("scoreHome")
                sa2 = actions[i].get("scoreAway")
                if sh2 is not None and sa2 is not None:
                    try:
                        home_score = int(sh2)
                        away_score = int(sa2)
                    except (ValueError, TypeError):
                        pass
                i += 1

            # Close current stint
            duration = t - stint_start
            if duration > 0 and len(home_lineup) == 5 and len(away_lineup) == 5:
                stints.append({
                    "game_id": game.game_id,
                    "home_players": frozenset(home_lineup),
                    "away_players": frozenset(away_lineup),
                    "duration_seconds": duration,
                    "home_points": home_score - stint_start_home,
                    "away_points": away_score - stint_start_away,
                    "margin": (home_score - stint_start_home) - (away_score - stint_start_away),
                })

            # Apply all subs atomically
            _apply_sub_batch(sub_batch, home_lineup, away_lineup, game.home_team_id)

            stint_start = t
            stint_start_home = home_score
            stint_start_away = away_score
            continue

        i += 1

    # Close final stint
    t = _elapsed_seconds(current_period, "PT00M00.00S")
    duration = t - stint_start
    if duration > 0 and len(home_lineup) == 5 and len(away_lineup) == 5:
        stints.append({
            "game_id": game.game_id,
            "home_players": frozenset(home_lineup),
            "away_players": frozenset(away_lineup),
            "duration_seconds": duration,
            "home_points": home_score - stint_start_home,
            "away_points": away_score - stint_start_away,
            "margin": (home_score - stint_start_home) - (away_score - stint_start_away),
        })

    return stints


def build_stint_dataset(game_data_list: list[GameData]) -> pd.DataFrame:
    """Parse all games into a single stints DataFrame."""
    all_stints = []
    for game in game_data_list:
        all_stints.extend(parse_game_stints(game))

    df = pd.DataFrame(all_stints)
    df = df[df["duration_seconds"] >= 10].reset_index(drop=True)
    return df
