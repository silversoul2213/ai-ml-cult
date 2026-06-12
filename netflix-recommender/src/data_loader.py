"""
Data loading and parsing for the Netflix Prize dataset.

The raw `combined_data_*.txt` files use a compact format where a line ending in
':' introduces a movie id, and subsequent lines are `user_id,rating,date` for
that movie. This module turns that into a tidy DataFrame:

    user_id | movie_id | rating | date

It also loads `movie_titles.csv` (movie_id, year, title) and offers a
`synthetic_ratings()` helper so the whole pipeline can be smoke-tested without
the multi-GB download.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    ratings: pd.DataFrame          # columns: user_id, movie_id, rating, date
    titles: pd.DataFrame | None    # columns: movie_id, year, title
    # contiguous index maps (built lazily for the models)
    user_index: dict | None = None
    movie_index: dict | None = None

    def build_indexes(self) -> "Dataset":
        users = self.ratings["user_id"].unique()
        movies = self.ratings["movie_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.movie_index = {m: i for i, m in enumerate(movies)}
        return self

    @property
    def n_users(self) -> int:
        return self.ratings["user_id"].nunique()

    @property
    def n_movies(self) -> int:
        return self.ratings["movie_id"].nunique()

    def title_for(self, movie_id) -> str:
        if self.titles is None:
            return f"Movie {movie_id}"
        row = self.titles.loc[self.titles["movie_id"] == movie_id]
        if row.empty:
            return f"Movie {movie_id}"
        t = row.iloc[0]
        year = "" if pd.isna(t["year"]) else f" ({int(t['year'])})"
        return f"{t['title']}{year}"


def parse_combined_file(path: str) -> pd.DataFrame:
    """Parse a single combined_data_*.txt file into a long DataFrame.

    Streaming line-by-line keeps memory modest even for the 24M-row files.
    """
    user_ids, movie_ids, ratings, dates = [], [], [], []
    current_movie = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.endswith(":"):
                current_movie = int(line[:-1])
            else:
                uid, rating, date = line.split(",")
                user_ids.append(int(uid))
                movie_ids.append(current_movie)
                ratings.append(np.int8(int(rating)))
                dates.append(date)
    df = pd.DataFrame(
        {
            "user_id": np.asarray(user_ids, dtype=np.int32),
            "movie_id": np.asarray(movie_ids, dtype=np.int32),
            "rating": np.asarray(ratings, dtype=np.int8),
            "date": pd.to_datetime(dates),
        }
    )
    return df


def load_titles(data_dir: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "movie_titles.csv")
    if not os.path.exists(path):
        return None
    # latin-1 + python engine because some titles contain stray commas;
    # we cap the split at 2 so commas inside the title survive.
    rows = []
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(",", 2)
            if len(parts) < 3:
                continue
            mid, year, title = parts
            year_val = np.nan if year in ("", "NULL") else int(year)
            rows.append((int(mid), year_val, title))
    return pd.DataFrame(rows, columns=["movie_id", "year", "title"])


def load_dataset(
    data_dir: str,
    files: list[str] | None = None,
    sample_users: int | None = None,
    min_ratings_per_user: int = 1,
    seed: int = 42,
) -> Dataset:
    """Load and optionally subsample the dataset.

    sample_users keeps ALL ratings for a random subset of users, which preserves
    realistic per-user density (unlike row sampling, which destroys it).
    """
    if files is None:
        files = [f for f in os.listdir(data_dir) if f.startswith("combined_data")]
        files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No combined_data_*.txt files found in {data_dir!r}. "
            "Download the dataset from Kaggle (see README)."
        )

    frames = [parse_combined_file(os.path.join(data_dir, f)) for f in files]
    ratings = pd.concat(frames, ignore_index=True)

    if sample_users is not None:
        rng = np.random.default_rng(seed)
        all_users = ratings["user_id"].unique()
        if sample_users < len(all_users):
            keep = rng.choice(all_users, size=sample_users, replace=False)
            ratings = ratings[ratings["user_id"].isin(keep)].reset_index(drop=True)

    if min_ratings_per_user > 1:
        counts = ratings["user_id"].value_counts()
        keep = counts[counts >= min_ratings_per_user].index
        ratings = ratings[ratings["user_id"].isin(keep)].reset_index(drop=True)

    titles = load_titles(data_dir)
    return Dataset(ratings=ratings, titles=titles)


def synthetic_ratings(
    n_users: int = 400,
    n_movies: int = 120,
    n_ratings: int = 12000,
    n_latent: int = 6,
    seed: int = 0,
) -> Dataset:
    """Generate a small synthetic dataset with genuine latent structure.

    Used for unit/smoke testing the full pipeline without the real download.
    Ratings are produced from a low-rank user/movie factor model so that
    collaborative methods can actually learn something.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (n_users, n_latent))
    V = rng.normal(0, 1, (n_movies, n_latent))
    ub = rng.normal(0, 0.5, n_users)
    vb = rng.normal(0, 0.5, n_movies)
    global_mean = 3.5

    users = rng.integers(0, n_users, n_ratings)
    movies = rng.integers(0, n_movies, n_ratings)
    raw = global_mean + ub[users] + vb[movies] + (U[users] * V[movies]).sum(1) * 0.4
    raw += rng.normal(0, 0.4, n_ratings)
    ratings = np.clip(np.round(raw), 1, 5).astype(np.int8)

    dates = pd.to_datetime("2005-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n_ratings), unit="D"
    )
    df = pd.DataFrame(
        {
            "user_id": users.astype(np.int32) + 1,
            "movie_id": movies.astype(np.int32) + 1,
            "rating": ratings,
            "date": dates,
        }
    ).drop_duplicates(subset=["user_id", "movie_id"]).reset_index(drop=True)

    titles = pd.DataFrame(
        {
            "movie_id": np.arange(1, n_movies + 1),
            "year": rng.integers(1990, 2005, n_movies),
            "title": [f"Synthetic Title {i}" for i in range(1, n_movies + 1)],
        }
    )
    return Dataset(ratings=df, titles=titles)


def train_test_split_temporal(
    ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
):
    """Per-user leave-out split.

    For every user we move the most recent `test_frac` of their ratings into the
    test set. This mimics deployment (predict the future from the past) and
    guarantees every test user is also seen in training (no cold-start leakage
    into the held-out evaluation, which we handle separately).
    """
    ratings = ratings.sort_values(["user_id", "date"]).reset_index(drop=True)
    test_mask = np.zeros(len(ratings), dtype=bool)
    for _, idx in ratings.groupby("user_id").groups.items():
        idx = np.asarray(idx)
        n_test = max(1, int(round(len(idx) * test_frac))) if len(idx) > 1 else 0
        if n_test:
            test_mask[idx[-n_test:]] = True
    train = ratings[~test_mask].reset_index(drop=True)
    test = ratings[test_mask].reset_index(drop=True)
    return train, test


if __name__ == "__main__":
    # quick self-check on synthetic data
    ds = synthetic_ratings()
    print(f"users={ds.n_users} movies={ds.n_movies} ratings={len(ds.ratings)}")
    tr, te = train_test_split_temporal(ds.ratings)
    print(f"train={len(tr)} test={len(te)}")
