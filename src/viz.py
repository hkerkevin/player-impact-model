"""Publication-ready visualizations for RAPM player impact results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rapm_rankings(
    results: pd.DataFrame,
    output_dir: str = "output",
    season: str = "2023-24",
    n: int = 15,
    min_minutes: int = 500,
) -> Path:
    """Horizontal bar chart of top and bottom players by RAPM.

    Returns path to saved figure.
    """
    qualified = results[results["minutes"] >= min_minutes].copy()
    top = qualified.head(n).iloc[::-1]
    bottom = qualified.tail(n)
    combined = pd.concat([bottom, top])

    fig, ax = plt.subplots(figsize=(10, 12))

    colors = ["#C9082A" if v < 0 else "#552583" for v in combined["rapm"]]
    bars = ax.barh(range(len(combined)), combined["rapm"], color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined["player_name"], fontsize=9)

    for i, (val, mins) in enumerate(zip(combined["rapm"], combined["minutes"])):
        offset = 0.15 if val >= 0 else -0.15
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:+.1f}", va="center", ha=ha, fontsize=8, fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(n - 0.5, color="gray", linewidth=0.5, linestyle="--")

    ax.set_xlabel("RAPM (points per 100 possessions)", fontsize=11)
    ax.set_title(
        f"NBA Player Impact — Regularized Adjusted Plus-Minus\n{season} Regular Season (min. {min_minutes} minutes)",
        fontsize=13,
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rapm_rankings_{season}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_rapm_distribution(
    results: pd.DataFrame,
    output_dir: str = "output",
    season: str = "2023-24",
    min_minutes: int = 500,
) -> Path:
    """Histogram of RAPM distribution across qualified players."""
    qualified = results[results["minutes"] >= min_minutes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(qualified["rapm"], bins=30, color="#552583", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.axvline(qualified["rapm"].mean(), color="#FDB927", linewidth=2, label=f"Mean: {qualified['rapm'].mean():.2f}")

    ax.set_xlabel("RAPM (points per 100 possessions)", fontsize=11)
    ax.set_ylabel("Number of Players", fontsize=11)
    ax.set_title(f"RAPM Distribution — {season}", fontsize=13, fontweight="bold")
    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rapm_distribution_{season}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
