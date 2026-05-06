"""Fetch NBA play-by-play and box score data via nba_api with local parquet caching."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    BoxScoreTraditionalV2,
    LeagueGameFinder,
    PlayByPlayV2,
)

CACHE_DIR = Path("data/raw")

CUSTOM_HEADERS = {
    "Host": "stats.nba.com",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
}

API_TIMEOUT = 120


@dataclass
class GameData:
    game_id: str
    home_team_id: int
    away_team_id: int
    pbp: pd.DataFrame
    box: pd.DataFrame


def get_season_games(season: str = "2023-24") -> pd.DataFrame:
    """Fetch regular season game metadata with home/away team identification."""
    cache_path = CACHE_DIR / f"games_{season}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    finder = LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
        headers=CUSTOM_HEADERS,
        timeout=API_TIMEOUT,
    )
    raw = finder.get_data_frames()[0]

    home = raw[raw["MATCHUP"].str.contains(" vs. ")][["GAME_ID", "TEAM_ID"]].rename(
        columns={"TEAM_ID": "HOME_TEAM_ID"}
    )
    away = raw[raw["MATCHUP"].str.contains(" @ ")][["GAME_ID", "TEAM_ID"]].rename(
        columns={"TEAM_ID": "AWAY_TEAM_ID"}
    )
    games = home.merge(away, on="GAME_ID")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(cache_path, index=False)
    return games


def _fetch_cached(endpoint_cls, game_id: str, prefix: str, **kwargs) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{prefix}_{game_id}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    ep = endpoint_cls(game_id=game_id, headers=CUSTOM_HEADERS, timeout=API_TIMEOUT, **kwargs)
    df = ep.get_data_frames()[0]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def fetch_season_data(
    season: str = "2023-24",
    max_games: int | None = None,
    delay: float = 1.0,
    retries: int = 3,
) -> list[GameData]:
    """Fetch play-by-play and box score data for a season.

    Returns a list of GameData objects, one per successfully fetched game.
    Data is cached locally as parquet files to avoid redundant API calls.
    """
    games = get_season_games(season)
    if max_games:
        games = games.head(max_games)

    results: list[GameData] = []
    total = len(games)

    for _, row in games.iterrows():
        gid = row["GAME_ID"]
        print(f"  [{len(results) + 1}/{total}] {gid}", end="", flush=True)

        success = False
        for attempt in range(retries):
            try:
                pbp = _fetch_cached(PlayByPlayV2, gid, "pbp")
                time.sleep(delay)
                box = _fetch_cached(BoxScoreTraditionalV2, gid, "box")
                time.sleep(delay)

                results.append(
                    GameData(
                        game_id=gid,
                        home_team_id=int(row["HOME_TEAM_ID"]),
                        away_team_id=int(row["AWAY_TEAM_ID"]),
                        pbp=pbp,
                        box=box,
                    )
                )
                print(" OK", flush=True)
                success = True
                break
            except Exception as e:
                if attempt < retries - 1:
                    wait = delay * (attempt + 2)
                    print(f" retry({attempt + 1})...", end="", flush=True)
                    time.sleep(wait)
                else:
                    print(f" ERROR: {e}", flush=True)

    return results
