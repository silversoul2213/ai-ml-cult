"""
Exploratory Data Analysis.

Produces the figures and numeric summary the technical report needs:
  * rating distribution
  * user-activity (ratings per user) distribution
  * movie-popularity (ratings per movie) distribution
  * sparsity / coverage statistics
  * temporal volume trend

All figures are saved to <out>/figures and a machine-readable summary to
<out>/eda_insights.json so the report's numbers come straight from the data.
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_eda(dataset, out_dir: str) -> dict:
    ratings = dataset.ratings
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    n_users = ratings["user_id"].nunique()
    n_movies = ratings["movie_id"].nunique()
    n_ratings = len(ratings)
    sparsity = 1.0 - n_ratings / (n_users * n_movies)

    per_user = ratings.groupby("user_id").size()
    per_movie = ratings.groupby("movie_id").size()

    insights = {
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "n_ratings": int(n_ratings),
        "sparsity": round(float(sparsity), 6),
        "density_pct": round(float((1 - sparsity) * 100), 4),
        "mean_rating": round(float(ratings["rating"].mean()), 4),
        "median_rating": float(ratings["rating"].median()),
        "rating_value_counts": ratings["rating"].value_counts().sort_index().to_dict(),
        "ratings_per_user": {
            "min": int(per_user.min()), "median": float(per_user.median()),
            "mean": round(float(per_user.mean()), 2), "max": int(per_user.max()),
        },
        "ratings_per_movie": {
            "min": int(per_movie.min()), "median": float(per_movie.median()),
            "mean": round(float(per_movie.mean()), 2), "max": int(per_movie.max()),
        },
        "pct_movies_covering_80pct_ratings": _long_tail_stat(per_movie),
    }

    # --- figures ---
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = ratings["rating"].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color="#4C72B0")
    ax.set(title="Rating distribution", xlabel="Star rating", ylabel="Count")
    _save(fig, os.path.join(fig_dir, "rating_distribution.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(per_user.values), bins=40, color="#55A868")
    ax.set(title="User activity (log10 ratings/user)",
           xlabel="log10(ratings per user)", ylabel="Number of users")
    _save(fig, os.path.join(fig_dir, "user_activity.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(per_movie.values), bins=40, color="#C44E52")
    ax.set(title="Movie popularity (log10 ratings/movie)",
           xlabel="log10(ratings per movie)", ylabel="Number of movies")
    _save(fig, os.path.join(fig_dir, "movie_popularity.png"))

    if "date" in ratings.columns:
        monthly = ratings.set_index("date").resample("ME").size()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(monthly.index, monthly.values, color="#8172B3")
        ax.set(title="Rating volume over time", xlabel="Date", ylabel="Ratings/month")
        _save(fig, os.path.join(fig_dir, "temporal_volume.png"))

    # popularity Lorenz-style cumulative curve (illustrates the long tail)
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_pop = np.sort(per_movie.values)[::-1]
    cum = np.cumsum(sorted_pop) / sorted_pop.sum()
    ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum * 100, color="#CCB974")
    ax.set(title="Cumulative rating share by movie popularity",
           xlabel="Top % of movies", ylabel="% of all ratings")
    _save(fig, os.path.join(fig_dir, "long_tail.png"))

    with open(os.path.join(out_dir, "eda_insights.json"), "w") as fh:
        json.dump(insights, fh, indent=2, default=str)
    return insights


def _long_tail_stat(per_movie) -> float:
    sorted_pop = np.sort(per_movie.values)[::-1]
    cum = np.cumsum(sorted_pop) / sorted_pop.sum()
    idx = int(np.searchsorted(cum, 0.80)) + 1
    return round(idx / len(sorted_pop) * 100, 2)
