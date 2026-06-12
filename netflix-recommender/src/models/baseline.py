"""
Baseline predictor: global mean + user bias + item bias.

    r_hat(u, i) = mu + b_u + b_i

Biases are estimated with damped means (shrinkage toward 0 for users/items with
few ratings), which is the classic Netflix baseline and a surprisingly strong
reference point. Any fancier model should beat this on RMSE to justify itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BaselineModel:
    name = "Baseline (bias)"

    def __init__(self, reg_user: float = 10.0, reg_item: float = 25.0):
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.global_mean = 0.0
        self.user_bias: dict = {}
        self.item_bias: dict = {}

    def fit(self, train: pd.DataFrame) -> "BaselineModel":
        self.global_mean = float(train["rating"].mean())

        # item bias first (damped)
        item_grp = train.groupby("movie_id")["rating"]
        item_sum = (item_grp.sum() - self.global_mean * item_grp.count())
        self.item_bias = (item_sum / (item_grp.count() + self.reg_item)).to_dict()

        # then user bias on the residual after removing item bias
        ib = train["movie_id"].map(self.item_bias).fillna(0.0).to_numpy()
        resid = train["rating"].to_numpy() - self.global_mean - ib
        tmp = pd.DataFrame({"user_id": train["user_id"].to_numpy(), "resid": resid})
        user_grp = tmp.groupby("user_id")["resid"]
        user_sum = user_grp.sum()
        self.user_bias = (user_sum / (user_grp.count() + self.reg_user)).to_dict()
        return self

    def predict(self, user_id, movie_id) -> float:
        pred = (
            self.global_mean
            + self.user_bias.get(user_id, 0.0)
            + self.item_bias.get(movie_id, 0.0)
        )
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, users, movies) -> np.ndarray:
        ub = np.array([self.user_bias.get(u, 0.0) for u in users])
        ib = np.array([self.item_bias.get(m, 0.0) for m in movies])
        return np.clip(self.global_mean + ub + ib, 1.0, 5.0)
