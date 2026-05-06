"""Fetch NBA play-by-play and box score data from the NBA CDN with local caching."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import LeagueGameFinder

CACHE_DIR = Path("data/raw")
CDN_PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
CDN_BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
CDN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
}


@dataclass
class GameData:
    game_id: str
    home_team_id: int
    away_team_id: int
    actions: list[dict]
    home_starters: set[int]
    away_starters: set[int]
    player_names: dict[int, str]


def get_season_games(season: str = "2023-24") -> pd.DataFrame:
    """Fetch regular season game metadata with home/away team identification."""
    cache_path = CACHE_DIR / f"games_{season}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    finder = LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
        timeout=60,
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


def _fetch_cdn_json(url: str, cache_key: str) -> dict:
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    r = requests.get(url, headers=CDN_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def _parse_boxscore(box_data: dict) -> tuple[int, int, set[int], set[int], dict[int, str]]:
    """Extract team IDs, starters, and player names from boxscore JSON."""
    game = box_data["game"]
    home = game["homeTeam"]
    away = game["awayTeam"]

    home_id = home["teamId"]
    away_id = away["teamId"]

    home_starters = {p["personId"] for p in home["players"] if p.get("starter") == "1"}
    away_starters = {p["personId"] for p in away["players"] if p.get("starter") == "1"}

    names = {}
    for p in home["players"] + away["players"]:
        names[p["personId"]] = p["name"]

    return home_id, away_id, home_starters, away_starters, names


def fetch_season_data(
    season: str = "2023-24",
    max_games: int | None = None,
    delay: float = 0.3,
) -> list[GameData]:
    """Fetch play-by-play and box score data for a season via NBA CDN.

    Returns a list of GameData objects, one per successfully fetched game.
    Data is cached locally as JSON files.
    """
    games = get_season_games(season)
    if max_games:
        games = games.head(max_games)

    results: list[GameData] = []
    total = len(games)

    for _, row in games.iterrows():
        gid = row["GAME_ID"]
        print(f"  [{len(results) + 1}/{total}] {gid}", end="", flush=True)

        try:
            pbp_data = _fetch_cdn_json(CDN_PBP_URL.format(game_id=gid), f"pbp_{gid}")
            box_data = _fetch_cdn_json(CDN_BOX_URL.format(game_id=gid), f"box_{gid}")
            time.sleep(delay)

            home_id, away_id, home_starters, away_starters, names = _parse_boxscore(box_data)
            actions = pbp_data["game"]["actions"]

            results.append(
                GameData(
                    game_id=gid,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    actions=actions,
                    home_starters=home_starters,
                    away_starters=away_starters,
                    player_names=names,
                )
            )
            print(" OK", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue

    return results
