"""Quick demo: run the full RAPM pipeline on synthetic NBA data.

Generates realistic stint data for the 2023-24 Lakers and opponents,
fits the RAPM model, and produces visualizations — proving the pipeline
works end-to-end without waiting on the NBA API.

Player impact values are seeded from actual 2023-24 on/off splits so
the model should recover a plausible ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.rapm import fit_rapm, display_results
from src.viz import plot_rapm_rankings, plot_rapm_distribution

np.random.seed(42)

# -------------------------------------------------------------------
# Real 2023-24 players with approximate true RAPM-range impacts
# -------------------------------------------------------------------
PLAYERS = {
    # Lakers
    "LeBron James": 4.5, "Anthony Davis": 5.2, "Austin Reaves": 2.1,
    "D'Angelo Russell": -0.8, "Rui Hachimura": 0.5,
    "Taurean Prince": -0.3, "Gabe Vincent": -1.5, "Jaxson Hayes": -0.9,
    "Cam Reddish": -1.2, "Christian Wood": -1.8, "Max Christie": 0.2,
    "Jarred Vanderbilt": 1.0, "Spencer Dinwiddie": -0.5,
    # Nuggets
    "Nikola Jokic": 7.8, "Jamal Murray": 2.5, "Michael Porter Jr.": 1.2,
    "Aaron Gordon": 2.0, "Kentavious Caldwell-Pope": 1.8,
    "Reggie Jackson": -1.0, "Christian Braun": 0.8, "Peyton Watson": -0.4,
    # Celtics
    "Jayson Tatum": 5.5, "Jaylen Brown": 3.8, "Jrue Holiday": 3.2,
    "Derrick White": 4.0, "Kristaps Porzingis": 4.8,
    "Al Horford": 2.2, "Sam Hauser": 0.7, "Payton Pritchard": 1.0,
    # Thunder
    "Shai Gilgeous-Alexander": 6.5, "Jalen Williams": 3.5,
    "Chet Holmgren": 3.0, "Lu Dort": 1.5, "Josh Giddey": -0.2,
    "Cason Wallace": 0.5, "Isaiah Joe": 0.3, "Kenrich Williams": 0.4,
    # Wolves
    "Anthony Edwards": 3.2, "Karl-Anthony Towns": 2.5,
    "Rudy Gobert": 3.8, "Jaden McDaniels": 2.0, "Mike Conley": 1.5,
    "Naz Reid": 1.8, "Nickeil Alexander-Walker": -0.3, "Kyle Anderson": 0.5,
    # Clippers
    "Kawhi Leonard": 5.0, "Paul George": 3.5, "James Harden": 2.0,
    "Russell Westbrook": -1.5, "Ivica Zubac": 1.2,
    "Norman Powell": 1.8, "Terance Mann": -0.2, "Bones Hyland": -0.8,
    # Bucks
    "Giannis Antetokounmpo": 6.0, "Damian Lillard": 2.8,
    "Khris Middleton": 1.5, "Brook Lopez": 3.0, "Bobby Portis": 0.5,
    "Pat Connaughton": -0.5, "MarJon Beauchamp": -1.0, "AJ Green": 0.3,
    # Suns
    "Kevin Durant": 4.2, "Devin Booker": 2.5, "Bradley Beal": 0.5,
    "Jusuf Nurkic": 0.8, "Grayson Allen": 1.5,
    "Eric Gordon": -0.5, "Drew Eubanks": -1.0, "Nassir Little": -0.8,
}

PLAYER_IDS = {name: 1000 + i for i, name in enumerate(PLAYERS)}

TEAMS = {
    "LAL": ["LeBron James", "Anthony Davis", "Austin Reaves", "D'Angelo Russell",
            "Rui Hachimura", "Taurean Prince", "Gabe Vincent", "Jaxson Hayes",
            "Cam Reddish", "Christian Wood", "Max Christie", "Jarred Vanderbilt",
            "Spencer Dinwiddie"],
    "DEN": ["Nikola Jokic", "Jamal Murray", "Michael Porter Jr.", "Aaron Gordon",
            "Kentavious Caldwell-Pope", "Reggie Jackson", "Christian Braun", "Peyton Watson"],
    "BOS": ["Jayson Tatum", "Jaylen Brown", "Jrue Holiday", "Derrick White",
            "Kristaps Porzingis", "Al Horford", "Sam Hauser", "Payton Pritchard"],
    "OKC": ["Shai Gilgeous-Alexander", "Jalen Williams", "Chet Holmgren", "Lu Dort",
            "Josh Giddey", "Cason Wallace", "Isaiah Joe", "Kenrich Williams"],
    "MIN": ["Anthony Edwards", "Karl-Anthony Towns", "Rudy Gobert", "Jaden McDaniels",
            "Mike Conley", "Naz Reid", "Nickeil Alexander-Walker", "Kyle Anderson"],
    "LAC": ["Kawhi Leonard", "Paul George", "James Harden", "Russell Westbrook",
            "Ivica Zubac", "Norman Powell", "Terance Mann", "Bones Hyland"],
    "MIL": ["Giannis Antetokounmpo", "Damian Lillard", "Khris Middleton", "Brook Lopez",
            "Bobby Portis", "Pat Connaughton", "MarJon Beauchamp", "AJ Green"],
    "PHX": ["Kevin Durant", "Devin Booker", "Bradley Beal", "Jusuf Nurkic",
            "Grayson Allen", "Eric Gordon", "Drew Eubanks", "Nassir Little"],
}


def _pick_lineup(roster: list[str], rng: np.random.RandomState) -> list[str]:
    """Pick 5 players from a roster weighted toward the top of the rotation."""
    weights = np.array([1.0 / (i + 1) for i in range(len(roster))])
    weights /= weights.sum()
    return list(rng.choice(roster, size=5, replace=False, p=weights))


def generate_stints(n_games: int = 200, stints_per_game: int = 40) -> pd.DataFrame:
    """Generate synthetic stint data reflecting real player impacts."""
    rng = np.random.RandomState(42)
    opp_teams = [t for t in TEAMS if t != "LAL"]
    stints = []

    for g in range(n_games):
        opp = rng.choice(opp_teams)
        home_is_lal = g % 2 == 0
        home_roster = TEAMS["LAL"] if home_is_lal else TEAMS[opp]
        away_roster = TEAMS[opp] if home_is_lal else TEAMS["LAL"]

        for s in range(stints_per_game):
            home5 = _pick_lineup(home_roster, rng)
            away5 = _pick_lineup(away_roster, rng)
            duration = rng.uniform(20, 300)

            home_impact = sum(PLAYERS[p] for p in home5)
            away_impact = sum(PLAYERS[p] for p in away5)
            net_impact = home_impact - away_impact

            possessions = duration / 14.5
            expected_margin = net_impact * possessions / 100
            noise = rng.normal(0, max(1.0, np.sqrt(possessions) * 3))
            margin = round(expected_margin + noise)

            stints.append({
                "game_id": f"00223{g:05d}",
                "home_players": frozenset(PLAYER_IDS[p] for p in home5),
                "away_players": frozenset(PLAYER_IDS[p] for p in away5),
                "duration_seconds": duration,
                "home_points": max(0, margin) if margin > 0 else 0,
                "away_points": max(0, -margin) if margin < 0 else 0,
                "margin": margin,
            })

    return pd.DataFrame(stints)


def main():
    print("=" * 60)
    print("  NBA Player Impact Model — Demo (2023-24)")
    print("=" * 60)

    print("\n[1/4] Generating synthetic stint data...")
    stints_df = generate_stints(n_games=500, stints_per_game=40)
    print(f"  {len(stints_df)} stints across {stints_df['game_id'].nunique()} games")
    print(f"  Avg stint duration: {stints_df['duration_seconds'].mean():.1f}s")

    print("\n[2/4] Fitting RAPM model...")
    id_to_name = {v: k for k, v in PLAYER_IDS.items()}
    results = fit_rapm(stints_df, alpha=5000, player_names=id_to_name)

    print("\n[3/4] Results")
    display_results(results, n=15)

    print("\n[4/4] Generating visualizations...")
    plot_rapm_rankings(results, "output", "2023-24 (Demo)", n=15, min_minutes=200)
    plot_rapm_distribution(results, "output", "2023-24 (Demo)", min_minutes=200)

    results.to_csv("output/rapm_results_demo.csv", index=False)
    print(f"  Saved: output/rapm_results_demo.csv")

    # Validation: compare recovered RAPM to seeded true impacts
    print("\n" + "=" * 60)
    print("  MODEL VALIDATION")
    print("=" * 60)
    merged = results.merge(
        pd.DataFrame({"player_name": PLAYERS.keys(), "true_impact": PLAYERS.values()}),
        on="player_name",
    )
    corr = merged["rapm"].corr(merged["true_impact"])
    mae = (merged["rapm"] - merged["true_impact"]).abs().mean()
    print(f"  Correlation (recovered vs true): {corr:.3f}")
    print(f"  Mean Absolute Error:             {mae:.2f}")
    print(f"  Top recovered: {merged.sort_values('rapm', ascending=False).iloc[0]['player_name']}")
    print(f"  Top true:      {merged.sort_values('true_impact', ascending=False).iloc[0]['player_name']}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
