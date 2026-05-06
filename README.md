# NBA Player Impact Model (RAPM)

A production-grade implementation of **Regularized Adjusted Plus-Minus (RAPM)** — the gold standard for isolating individual player contributions in basketball.

RAPM answers the question: **"How many points per 100 possessions does this player add (or subtract) when on the court, controlling for teammates and opponents?"**

## How It Works

Traditional plus-minus conflates a player's value with their teammates' and opponents' quality. RAPM solves this with a regression framework:

1. **Stint parsing**: Break each game into stints — continuous stretches of play between substitutions
2. **Design matrix**: For each stint, encode which 10 players are on court (+1 for home, -1 for away)
3. **Ridge regression**: Regress stint-level point differentials onto the player indicator matrix with L2 regularization
4. **Output**: Each player's coefficient = their estimated impact per 100 possessions, isolated from context

The L2 penalty (Ridge) shrinks low-minute players toward zero, preventing extreme coefficients from small sample sizes.

## Project Structure

```
├── run.py              # End-to-end pipeline CLI
├── src/
│   ├── data.py         # NBA API fetching with parquet caching
│   ├── stints.py       # Play-by-play → lineup stint parsing
│   ├── rapm.py         # RAPM model (sparse Ridge regression)
│   └── viz.py          # Publication-ready visualizations
├── data/raw/           # Cached API responses (gitignored)
└── output/             # Results and figures (gitignored)
```

## Results — 2023-24 Full Season (1,230 games)

![RAPM Rankings](output/rapm_rankings_2023-24.png)

**34,353 stints** parsed from all 1,230 regular season games across **540 players**.

| Metric | Value |
|--------|-------|
| 5-Fold CV R² | 0.0065 |
| 5-Fold CV RMSE | 127.28 |
| In-sample R² | 0.0165 |
| Qualified players (500+ min) | 360 |

> **Note on R²:** A low R² is expected and correct for RAPM. Individual players explain <5% of stint-level scoring variance — basketball is inherently noisy. The value is in the *coefficients* (player rankings), not the overall fit.

**Key findings:**
- **Jokic (+3.1)** leads the league — consistent with his 2023-24 MVP runner-up season
- **SGA (+2.9), OG Anunoby (+2.9), Paul George (+2.8)** round out the top tier
- **LeBron (+2.1)** at #14 — strong positive impact in his 21st season
- Bottom-15 flags known negative-impact players on losing teams (Osman, Scoot Henderson, Wiseman)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full season (fetches ~1230 games, takes ~25 min first run, cached after)
python run.py --season 2023-24

# Quick test with 50 games
python run.py --season 2023-24 --max-games 50

# Custom regularization strength
python run.py --season 2023-24 --alpha 5000

# Cross-validated alpha (default)
python run.py --season 2023-24
```

### Output

- `output/rapm_rankings_{season}.png` — Top/bottom players bar chart
- `output/rapm_distribution_{season}.png` — RAPM distribution histogram
- `output/rapm_results_{season}.csv` — Full results table

## Technical Details

| Component | Detail |
|-----------|--------|
| Data source | NBA CDN (play-by-play + box scores), `nba_api` (game index) |
| Stint parsing | Substitution-event-driven, per-period lineup tracking |
| Possessions proxy | `stint_duration / 14.5s` (avg NBA possession length) |
| Target variable | Point margin per 100 possessions |
| Sample weights | Estimated possessions per stint |
| Regularization | Ridge (L2), alpha selected via cross-validation or user-specified |
| Sparse matrix | `scipy.sparse.coo_matrix` for memory-efficient design matrix |

## Methodology Notes

- **Minimum minutes filter** (default 500): Players below this threshold are included in the model but excluded from rankings to avoid noisy estimates.
- **Stint filtering**: Stints shorter than 10 seconds are dropped (rapid substitution noise).
- **Period transitions**: Lineups carry forward across periods; substitutions at period starts are handled via play-by-play events.
- **Caching**: All API responses are cached as JSON files in `data/raw/`. Delete this directory to force a re-fetch.
