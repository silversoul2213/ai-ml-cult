"""
Evaluation metrics.

Mandatory:
  * RMSE  — rating-prediction accuracy on held-out (user, item, rating) triples.
  * MAP@10 — Mean Average Precision over the Top-10 recommended items, where an
             item is RELEVANT iff its true held-out rating >= RELEVANCE (3.5).

Procedure for MAP@10
--------------------
For each test user:
  1. The model scores every candidate item the user has NOT seen in TRAIN.
  2. We rank those candidates and take the Top-10.
  3. Relevant items = test items the user actually rated >= 3.5.
  4. Average Precision (AP@10) rewards placing relevant items high in the list.
MAP@10 is the mean of AP@10 across evaluated users.

Optional metrics (MAE, Precision@K, Recall@K, NDCG@K) are also provided.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------- rating accuracy ----------------------------- #
def rmse(model, test: pd.DataFrame) -> float:
    preds = model.predict_batch(
        test["user_id"].to_numpy(), test["movie_id"].to_numpy()
    )
    return float(np.sqrt(np.mean((preds - test["rating"].to_numpy()) ** 2)))


def mae(model, test: pd.DataFrame) -> float:
    preds = model.predict_batch(
        test["user_id"].to_numpy(), test["movie_id"].to_numpy()
    )
    return float(np.mean(np.abs(preds - test["rating"].to_numpy())))


# ----------------------------- ranking helpers ----------------------------- #
def _average_precision_at_k(ranked_items, relevant_set, k: int) -> float:
    if not relevant_set:
        return np.nan  # undefined; excluded from the mean
    hits = 0
    score = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in relevant_set:
            hits += 1
            score += hits / rank
    return score / min(len(relevant_set), k)


def _dcg(rels) -> float:
    rels = np.asarray(rels, dtype=float)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def _score_candidates(model, user_id, candidates):
    users = [user_id] * len(candidates)
    return model.predict_batch(users, candidates)


def ranking_metrics(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    relevance: float = 3.5,
    n_candidates: int = 200,
    max_users: int | None = 2000,
    seed: int = 42,
) -> dict:
    """Compute MAP@k (+ optional Precision/Recall/NDCG@k).

    To keep evaluation tractable on 480K users we (a) optionally evaluate a
    random sample of `max_users` test users, and (b) rank each user's true test
    items against a sampled pool of `n_candidates` unseen negatives — the
    standard sampled-ranking protocol used in recommender benchmarks.
    """
    rng = np.random.default_rng(seed)

    train_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    test_items = (
        test.groupby("user_id")
        .apply(lambda g: dict(zip(g["movie_id"], g["rating"])))
        .to_dict()
    )
    all_items = np.array(sorted(set(train["movie_id"]) | set(test["movie_id"])))

    eval_users = list(test_items.keys())
    if max_users is not None and len(eval_users) > max_users:
        eval_users = list(rng.choice(eval_users, size=max_users, replace=False))

    aps, precisions, recalls, ndcgs = [], [], [], []
    for user_id in eval_users:
        truth = test_items[user_id]
        relevant = {m for m, r in truth.items() if r >= relevance}

        seen = train_seen.get(user_id, set())
        # candidate pool: the user's true test items + sampled unseen negatives
        neg_pool = np.setdiff1d(all_items, np.array(list(seen | set(truth.keys()))))
        n_neg = min(n_candidates, len(neg_pool))
        negatives = rng.choice(neg_pool, size=n_neg, replace=False) if n_neg else np.array([])
        candidates = np.concatenate([np.array(list(truth.keys())), negatives]).astype(int)
        if candidates.size == 0:
            continue

        scores = _score_candidates(model, user_id, candidates)
        order = np.argsort(scores)[::-1]
        ranked = candidates[order]

        ap = _average_precision_at_k(ranked, relevant, k)
        if not np.isnan(ap):
            aps.append(ap)

        topk = ranked[:k]
        n_rel_in_topk = sum(1 for m in topk if m in relevant)
        precisions.append(n_rel_in_topk / k)
        if relevant:
            recalls.append(n_rel_in_topk / len(relevant))
            gains = [1.0 if m in relevant else 0.0 for m in topk]
            ideal = sorted([1.0] * len(relevant), reverse=True)[:k]
            ndcg = _dcg(gains) / _dcg(ideal) if _dcg(ideal) > 0 else 0.0
            ndcgs.append(ndcg)

    return {
        f"MAP@{k}": float(np.mean(aps)) if aps else float("nan"),
        f"Precision@{k}": float(np.mean(precisions)) if precisions else float("nan"),
        f"Recall@{k}": float(np.mean(recalls)) if recalls else float("nan"),
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "n_eval_users": len(eval_users),
    }


def evaluate_model(
    model, train, test, k=10, relevance=3.5, optional=True, **kwargs
) -> dict:
    out = {"RMSE": rmse(model, test)}
    if optional:
        out["MAE"] = mae(model, test)
    out.update(ranking_metrics(model, train, test, k=k, relevance=relevance, **kwargs))
    return out
