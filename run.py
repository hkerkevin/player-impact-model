"""End-to-end NBA Player Impact Model pipeline.

Usage:
    python run.py --season 2023-24 --max-games 100 --alpha 5000
"""

import argparse

from src.data import fetch_season_data
from src.stints import build_stint_dataset
from src.rapm import fit_rapm, display_results
from src.viz import plot_rapm_rankings, plot_rapm_distribution


def _build_player_name_map(game_data_list) -> dict[int, str]:
    """Build player_id → player_name mapping from box score data."""
    names = {}
    for game in game_data_list:
        for _, row in game.box.iterrows():
            pid = int(row["PLAYER_ID"])
            if pid not in names:
                names[pid] = row["PLAYER_NAME"]
    return names


def main():
    parser = argparse.ArgumentParser(description="NBA Player Impact Model (RAPM)")
    parser.add_argument("--season", default="2023-24", help="NBA season (e.g. 2023-24)")
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games (for testing)")
    parser.add_argument("--alpha", type=float, default=None, help="Ridge regularization strength (default: cross-validated)")
    parser.add_argument("--output", default="output", help="Output directory for figures")
    parser.add_argument("--min-minutes", type=int, default=500, help="Minimum minutes for qualified players")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  NBA Player Impact Model (RAPM) — {args.season}")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/4] Fetching play-by-play data...")
    game_data = fetch_season_data(args.season, args.max_games)
    player_names = _build_player_name_map(game_data)
    print(f"  Fetched {len(game_data)} games, {len(player_names)} unique players")

    # 2. Parse stints
    print("\n[2/4] Parsing lineup stints...")
    stints_df = build_stint_dataset(game_data)
    print(f"  {len(stints_df)} stints from {stints_df['game_id'].nunique()} games")
    print(f"  Avg stint duration: {stints_df['duration_seconds'].mean():.1f}s")

    # 3. Fit RAPM
    print("\n[3/4] Fitting RAPM model...")
    results = fit_rapm(stints_df, alpha=args.alpha, player_names=player_names)
    display_results(results, n=15)

    # 4. Visualize
    print("\n[4/4] Generating visualizations...")
    plot_rapm_rankings(results, args.output, args.season, min_minutes=args.min_minutes)
    plot_rapm_distribution(results, args.output, args.season, min_minutes=args.min_minutes)

    # Save full results
    results.to_csv(f"{args.output}/rapm_results_{args.season}.csv", index=False)
    print(f"  Saved: {args.output}/rapm_results_{args.season}.csv")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
