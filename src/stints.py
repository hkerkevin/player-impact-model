"""Parse play-by-play data into lineup stints for RAPM modeling.

A stint is a continuous stretch of game time where no substitutions occur.
Each stint records which 10 players are on court and the point differential.
"""

import pandas as pd
import numpy as np

from src.data import GameData

EVENT_MADE_SHOT = 1
EVENT_FREE_THROW = 3
EVENT_SUBSTITUTION = 8


def _elapsed_seconds(period: int, pctimestring: str) -> float:
    """Convert period + game clock to cumulative elapsed seconds."""
    parts = pctimestring.split(":")
    mins, secs = int(parts[0]), int(parts[1])
    remaining = mins * 60 + secs

    period_length = 12 * 60 if period <= 4 else 5 * 60
    elapsed_in_period = period_length - remaining

    prior = sum(12 * 60 for _ in range(1, min(period, 5)))
    if period > 4:
        prior += sum(5 * 60 for _ in range(5, period))

    return prior + elapsed_in_period


def _get_starters(box_df: pd.DataFrame, team_id: int) -> set[int]:
    """Extract starting five from box score by START_POSITION field."""
    team = box_df[box_df["TEAM_ID"] == team_id]
    starters = team[
        team["START_POSITION"].notna() & (team["START_POSITION"] != "")
    ]
    return set(starters["PLAYER_ID"].astype(int).tolist())


def _score_from_event(row, home_team_id: int) -> tuple[int, int]:
    """Return (home_points, away_points) for a single scoring event."""
    etype = row["EVENTMSGTYPE"]

    if etype == EVENT_MADE_SHOT:
        team_id = row.get("PLAYER1_TEAM_ID")
        if pd.isna(team_id):
            return 0, 0
        team_id = int(team_id)
        desc = str(row.get("HOMEDESCRIPTION") or "") + str(row.get("VISITORDESCRIPTION") or "")
        pts = 3 if "3PT" in desc else 2
        return (pts, 0) if team_id == home_team_id else (0, pts)

    if etype == EVENT_FREE_THROW:
        desc = str(row.get("HOMEDESCRIPTION") or "") + str(row.get("VISITORDESCRIPTION") or "")
        if "MISS" in desc.upper():
            return 0, 0
        team_id = row.get("PLAYER1_TEAM_ID")
        if pd.isna(team_id):
            return 0, 0
        team_id = int(team_id)
        return (1, 0) if team_id == home_team_id else (0, 1)

    return 0, 0


def _close_stint(
    game_id: str,
    home_lineup: set[int],
    away_lineup: set[int],
    start_time: float,
    end_time: float,
    home_pts: int,
    away_pts: int,
) -> dict | None:
    duration = end_time - start_time
    if duration <= 0 or len(home_lineup) != 5 or len(away_lineup) != 5:
        return None
    return {
        "game_id": game_id,
        "home_players": frozenset(home_lineup),
        "away_players": frozenset(away_lineup),
        "duration_seconds": duration,
        "home_points": home_pts,
        "away_points": away_pts,
        "margin": home_pts - away_pts,
    }


def parse_game_stints(game: GameData) -> list[dict]:
    """Parse a single game's play-by-play into stints."""
    home_lineup = _get_starters(game.box, game.home_team_id)
    away_lineup = _get_starters(game.box, game.away_team_id)

    if len(home_lineup) != 5 or len(away_lineup) != 5:
        return []

    pbp = game.pbp.sort_values(["PERIOD", "EVENTNUM"]).reset_index(drop=True)

    stints = []
    stint_start = 0.0
    h_pts = a_pts = 0
    current_period = 1

    for _, row in pbp.iterrows():
        period = row["PERIOD"]
        etype = row["EVENTMSGTYPE"]

        # Period transition — close current stint, carry lineups forward
        if period != current_period:
            end_t = _elapsed_seconds(current_period, "0:00")
            s = _close_stint(game.game_id, home_lineup, away_lineup, stint_start, end_t, h_pts, a_pts)
            if s:
                stints.append(s)
            current_period = period
            clock = "12:00" if period <= 4 else "5:00"
            stint_start = _elapsed_seconds(period, clock)
            h_pts = a_pts = 0

        # Substitution — close stint, update lineup
        if etype == EVENT_SUBSTITUTION:
            t = _elapsed_seconds(period, row["PCTIMESTRING"])
            s = _close_stint(game.game_id, home_lineup, away_lineup, stint_start, t, h_pts, a_pts)
            if s:
                stints.append(s)
            stint_start = t
            h_pts = a_pts = 0

            p_in = row.get("PLAYER1_ID")
            p_out = row.get("PLAYER2_ID")
            if pd.notna(p_in) and pd.notna(p_out):
                p_in, p_out = int(p_in), int(p_out)
                # Determine direction by checking who is currently on court
                if p_out in home_lineup:
                    home_lineup.discard(p_out)
                    home_lineup.add(p_in)
                elif p_out in away_lineup:
                    away_lineup.discard(p_out)
                    away_lineup.add(p_in)
                elif p_in in home_lineup:
                    home_lineup.discard(p_in)
                    home_lineup.add(p_out)
                elif p_in in away_lineup:
                    away_lineup.discard(p_in)
                    away_lineup.add(p_out)
            continue

        # Scoring
        dh, da = _score_from_event(row, game.home_team_id)
        h_pts += dh
        a_pts += da

    # Close final stint
    end_t = _elapsed_seconds(current_period, "0:00")
    s = _close_stint(game.game_id, home_lineup, away_lineup, stint_start, end_t, h_pts, a_pts)
    if s:
        stints.append(s)

    return stints


def build_stint_dataset(game_data_list: list[GameData]) -> pd.DataFrame:
    """Parse all games into a single stints DataFrame."""
    all_stints = []
    for game in game_data_list:
        all_stints.extend(parse_game_stints(game))

    df = pd.DataFrame(all_stints)
    # Drop stints shorter than 10 seconds (noise from rapid substitutions)
    df = df[df["duration_seconds"] >= 10].reset_index(drop=True)
    return df
