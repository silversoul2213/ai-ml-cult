"""
Item-based collaborative filtering with adjusted-cosine similarity.

For a target (user u, item i) we predict from the items u already rated,
weighted by how similar those items are to i:

    r_hat(u,i) = b_ui + sum_j sim(i,j) * (r_uj - b_uj) / sum_j |sim(i,j)|

where b are baseline (bias) estimates. Similarities are computed on a sparse
user-item matrix and truncated to the top-`k_neighbors` most similar items per
item to keep memory and prediction time bounded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .baseline import BaselineModel


class ItemItemCF:
    name = "Item-Item CF"

    def __init__(self, k_neighbors: int = 50, min_overlap: int = 3, shrinkage: float = 25.0):
        self.k_neighbors = k_neighbors
        self.min_overlap = min_overlap
        self.shrinkage = shrinkage
        self.baseline = BaselineModel()
        self.movie_ids: np.ndarray | None = None
        self.movie_pos: dict = {}
        self.user_pos: dict = {}
        self.sim_topk: dict = {}     # movie_id -> list[(neighbor_movie_id, sim)]
        self.user_rated: dict = {}   # user_id -> dict(movie_id -> rating)

    def fit(self, train: pd.DataFrame) -> "ItemItemCF":
        self.baseline.fit(train)

        self.movie_ids = np.sort(train["movie_id"].unique())
        self.movie_pos = {m: i for i, m in enumerate(self.movie_ids)}
        users = np.sort(train["user_id"].unique())
        self.user_pos = {u: i for i, u in enumerate(users)}

        rows = train["movie_id"].map(self.movie_pos).to_numpy()
        cols = train["user_id"].map(self.user_pos).to_numpy()

        # mean-center each item's ratings (adjusted cosine) before similarity
        item_mean = train.groupby("movie_id")["rating"].transform("mean").to_numpy()
        vals = train["rating"].to_numpy() - item_mean

        mat = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.movie_ids), len(users))
        )
        sims = cosine_similarity(mat, dense_output=False).tolil()

        # co-rating counts: how many users rated BOTH items. A high cosine built
        # on only 2-3 shared users is unreliable, so we (a) drop pairs below
        # `min_overlap` and (b) apply significance shrinkage
        # sim' = sim * n_ij / (n_ij + shrinkage), pulling thin overlaps toward 0.
        binary = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(self.movie_ids), len(users)),
        )
        co = (binary @ binary.T).tocsr()

        # keep top-k neighbours per item (shrunk + overlap-filtered)
        for i in range(sims.shape[0]):
            co_row = co.getrow(i)
            co_map = dict(zip(co_row.indices, co_row.data))
            pairs = []
            for j, s in zip(sims.rows[i], sims.data[i]):
                if j == i or s <= 0:
                    continue
                n_ij = co_map.get(j, 0)
                if n_ij < self.min_overlap:
                    continue
                pairs.append((j, s * n_ij / (n_ij + self.shrinkage)))
            pairs.sort(key=lambda p: p[1], reverse=True)
            self.sim_topk[self.movie_ids[i]] = [
                (self.movie_ids[j], float(s)) for j, s in pairs[: self.k_neighbors]
            ]

        # cache each user's rated items for fast lookup at predict time
        self.user_rated = (
            train.groupby("user_id")
            .apply(lambda g: dict(zip(g["movie_id"], g["rating"])))
            .to_dict()
        )
        return self

    def predict(self, user_id, movie_id) -> float:
        base = self.baseline.predict(user_id, movie_id)
        rated = self.user_rated.get(user_id)
        neighbours = self.sim_topk.get(movie_id)
        if not rated or not neighbours:
            return base

        num = 0.0
        den = 0.0
        for nb_movie, sim in neighbours:
            if nb_movie in rated:
                b_uj = self.baseline.predict(user_id, nb_movie)
                num += sim * (rated[nb_movie] - b_uj)
                den += abs(sim)
        if den == 0:
            return base
        return float(np.clip(base + num / den, 1.0, 5.0))

    def predict_batch(self, users, movies) -> np.ndarray:
        return np.array([self.predict(u, m) for u, m in zip(users, movies)])
