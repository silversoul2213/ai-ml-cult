"""
Matrix factorization via stochastic gradient descent (the "SVD" of the Netflix
Prize, a la Funk/Koren). Predicts:

    r_hat(u,i) = mu + b_u + b_i + p_u . q_i

Parameters learned by minimizing regularized squared error over observed
ratings:

    min  sum_(u,i) (r_ui - r_hat)^2 + lambda(||p_u||^2+||q_i||^2+b_u^2+b_i^2)

Pure NumPy so there's no scikit-surprise / external dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MatrixFactorization:
    name = "Matrix Factorization (SGD)"

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.verbose = verbose

        self.global_mean = 0.0
        self.user_index: dict = {}
        self.item_index: dict = {}
        self.bu = None
        self.bi = None
        self.P = None
        self.Q = None
        self.train_rmse_history: list[float] = []

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        rng = np.random.default_rng(self.seed)
        users = train["user_id"].unique()
        items = train["movie_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {m: i for i, m in enumerate(items)}
        n_users, n_items = len(users), len(items)

        self.global_mean = float(train["rating"].mean())
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        scale = 0.1
        self.P = rng.normal(0, scale, (n_users, self.n_factors))
        self.Q = rng.normal(0, scale, (n_items, self.n_factors))

        u_arr = np.array([self.user_index[u] for u in train["user_id"]])
        i_arr = np.array([self.item_index[m] for m in train["movie_id"]])
        r_arr = train["rating"].to_numpy(dtype=np.float64)
        n = len(r_arr)

        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            sq_err = 0.0
            for idx in order:
                u, i, r = u_arr[idx], i_arr[idx], r_arr[idx]
                pred = self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err = r - pred
                sq_err += err * err

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu, qi = self.P[u].copy(), self.Q[i]
                self.P[u] += self.lr * (err * qi - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * qi)

            rmse = np.sqrt(sq_err / n)
            self.train_rmse_history.append(rmse)
            if self.verbose:
                msg = f"  epoch {epoch + 1:2d}/{self.n_epochs}  train RMSE={rmse:.4f}"
                if valid is not None:
                    msg += f"  valid RMSE={self._rmse(valid):.4f}"
                print(msg)
        return self

    def _rmse(self, df: pd.DataFrame) -> float:
        preds = self.predict_batch(df["user_id"].to_numpy(), df["movie_id"].to_numpy())
        return float(np.sqrt(np.mean((preds - df["rating"].to_numpy()) ** 2)))

    def predict(self, user_id, movie_id) -> float:
        pred = self.global_mean
        u = self.user_index.get(user_id)
        i = self.item_index.get(movie_id)
        if u is not None:
            pred += self.bu[u]
        if i is not None:
            pred += self.bi[i]
        if u is not None and i is not None:
            pred += self.P[u] @ self.Q[i]
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, users, movies) -> np.ndarray:
        out = np.full(len(users), self.global_mean, dtype=np.float64)
        for k, (uid, mid) in enumerate(zip(users, movies)):
            u = self.user_index.get(uid)
            i = self.item_index.get(mid)
            val = self.global_mean
            if u is not None:
                val += self.bu[u]
            if i is not None:
                val += self.bi[i]
            if u is not None and i is not None:
                val += self.P[u] @ self.Q[i]
            out[k] = val
        return np.clip(out, 1.0, 5.0)

    def recommend(self, user_id, known_items: set, n: int = 10):
        """Score all items for a user and return Top-n unseen ones."""
        u = self.user_index.get(user_id)
        if u is None:
            return []
        scores = self.global_mean + self.bu[u] + self.bi + self.Q @ self.P[u]
        order = np.argsort(scores)[::-1]
        inv_item = {v: k for k, v in self.item_index.items()}
        out = []
        for i in order:
            mid = inv_item[i]
            if mid in known_items:
                continue
            out.append((mid, float(np.clip(scores[i], 1.0, 5.0))))
            if len(out) >= n:
                break
        return out
