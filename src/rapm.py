"""Regularized Adjusted Plus-Minus (RAPM) model.

RAPM isolates individual player impact by regressing lineup-level point
differentials onto a player indicator matrix with L2 regularization.

Design matrix X: rows = stints, columns = players
    +1 if player is on court for the home team
    -1 if player is on court for the away team
    0  otherwise

Target y: point margin per 100 possessions (approximated via stint duration)
Weights:  estimated possessions per stint (duration / avg_possession_length)

The L2 penalty (Ridge) shrinks low-minute players toward zero, preventing
extreme coefficients from small sample sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold

AVG_POSSESSION_SECONDS = 14.5


def build_design_matrix(
    stints_df: pd.DataFrame,
) -> tuple[coo_matrix, list[int], np.ndarray, np.ndarray]:
    """Build the sparse RAPM design matrix from stint data.

    Returns:
        X:       sparse design matrix (n_stints x n_players)
        players: ordered list of player IDs (column index → player ID)
        y:       target vector (margin per 100 possessions)
        w:       sample weights (estimated possessions)
    """
    all_players: set[int] = set()
    for hp, ap in zip(stints_df["home_players"], stints_df["away_players"]):
        all_players.update(hp)
        all_players.update(ap)

    players = sorted(all_players)
    pid_to_idx = {pid: i for i, pid in enumerate(players)}

    rows, cols, vals = [], [], []
    for i, (_, stint) in enumerate(stints_df.iterrows()):
        for pid in stint["home_players"]:
            rows.append(i)
            cols.append(pid_to_idx[pid])
            vals.append(1.0)
        for pid in stint["away_players"]:
            rows.append(i)
            cols.append(pid_to_idx[pid])
            vals.append(-1.0)

    n = len(stints_df)
    p = len(players)
    X = coo_matrix((vals, (rows, cols)), shape=(n, p)).tocsr()

    possessions = stints_df["duration_seconds"].values / AVG_POSSESSION_SECONDS
    y = (stints_df["margin"].values / possessions) * 100
    w = possessions

    return X, players, y, w


def fit_rapm(
    stints_df: pd.DataFrame,
    alpha: float | None = None,
    player_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Fit RAPM model and return player impact rankings.

    Args:
        stints_df:    DataFrame from build_stint_dataset
        alpha:        Ridge regularization strength. If None, uses cross-validation.
        player_names: optional mapping of player_id → player_name

    Returns:
        DataFrame with columns: player_id, player_name, rapm, minutes
    """
    X, players, y, w = build_design_matrix(stints_df)

    # Replace any inf/nan from zero-duration stints
    mask = np.isfinite(y)
    X, y, w = X[mask], y[mask], w[mask]

    if alpha is not None:
        model = Ridge(alpha=alpha, fit_intercept=False)
    else:
        model = RidgeCV(
            alphas=[100, 500, 1000, 2500, 5000, 10000],
            fit_intercept=False,
        )

    model.fit(X, y, sample_weight=w)

    if hasattr(model, "alpha_"):
        print(f"  Cross-validated alpha: {model.alpha_}")

    # Evaluation: 5-fold cross-validation with proper sample weights
    eval_alpha = model.alpha_ if hasattr(model, "alpha_") else alpha
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2, cv_rmse = [], []
    for train_idx, test_idx in kf.split(X):
        fold_model = Ridge(alpha=eval_alpha, fit_intercept=False)
        fold_model.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
        y_pred_fold = fold_model.predict(X[test_idx])
        w_test = w[test_idx]
        ss_res = np.sum(w_test * (y[test_idx] - y_pred_fold) ** 2)
        ss_tot = np.sum(w_test * (y[test_idx] - np.average(y[test_idx], weights=w_test)) ** 2)
        cv_r2.append(1 - ss_res / ss_tot)
        cv_rmse.append(np.sqrt(np.mean(w_test * (y[test_idx] - y_pred_fold) ** 2)))
    print(f"  5-Fold CV R²:   {np.mean(cv_r2):.4f} (+/-{np.std(cv_r2):.4f})")
    print(f"  5-Fold CV RMSE: {np.mean(cv_rmse):.2f} (+/-{np.std(cv_rmse):.2f})")

    # In-sample R²
    y_pred = model.predict(X)
    ss_res = np.sum(w * (y - y_pred) ** 2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r2_train = 1 - ss_res / ss_tot
    print(f"  In-sample R²:   {r2_train:.4f}")

    # Compute total minutes per player
    all_players_set = {pid: 0.0 for pid in players}
    for _, stint in stints_df.iterrows():
        mins = stint["duration_seconds"] / 60
        for pid in stint["home_players"]:
            all_players_set[pid] += mins
        for pid in stint["away_players"]:
            all_players_set[pid] += mins

    if player_names is None:
        player_names = {}

    results = pd.DataFrame(
        {
            "player_id": players,
            "player_name": [player_names.get(pid, str(pid)) for pid in players],
            "rapm": model.coef_,
            "minutes": [all_players_set[pid] for pid in players],
        }
    )
    results = results.sort_values("rapm", ascending=False).reset_index(drop=True)
    return results


def display_results(results: pd.DataFrame, n: int = 15) -> None:
    """Print top and bottom players by RAPM."""
    min_minutes = 500
    qualified = results[results["minutes"] >= min_minutes]

    print(f"\n  Qualified players (>= {min_minutes} min): {len(qualified)}")
    print(f"\n  {'TOP ' + str(n) + ' PLAYERS':^50}")
    print(f"  {'—' * 50}")
    print(f"  {'Rank':<6}{'Player':<28}{'RAPM':>8}{'Minutes':>8}")
    print(f"  {'—' * 50}")
    for i, (_, row) in enumerate(qualified.head(n).iterrows()):
        print(f"  {i + 1:<6}{row['player_name']:<28}{row['rapm']:>+8.2f}{row['minutes']:>8.0f}")

    print(f"\n  {'BOTTOM ' + str(n) + ' PLAYERS':^50}")
    print(f"  {'—' * 50}")
    print(f"  {'Rank':<6}{'Player':<28}{'RAPM':>8}{'Minutes':>8}")
    print(f"  {'—' * 50}")
    bottom = qualified.tail(n).iloc[::-1]
    for i, (_, row) in enumerate(bottom.iterrows()):
        rank = len(qualified) - i
        print(f"  {rank:<6}{row['player_name']:<28}{row['rapm']:>+8.2f}{row['minutes']:>8.0f}")
