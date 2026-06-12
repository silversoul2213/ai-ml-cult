"""
Top-K recommendation generation + explanations.

Given a fitted model, produce ranked recommendations for sample users, attach
human-readable explanations ("because you liked X and Y"), and surface
success / failure cases for the report's qualitative analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _user_top_rated(train: pd.DataFrame, user_id, n=3):
    g = train[train["user_id"] == user_id].sort_values("rating", ascending=False)
    return g.head(n)["movie_id"].tolist()


def generate_recommendations(model, dataset, train, user_id, k=10):
    """Top-K unseen recommendations for a single user, with explanation."""
    seen = set(train[train["user_id"] == user_id]["movie_id"])
    all_items = np.array(sorted(set(train["movie_id"])))
    candidates = np.array([m for m in all_items if m not in seen])
    if candidates.size == 0:
        return []

    scores = model.predict_batch([user_id] * len(candidates), candidates)
    order = np.argsort(scores)[::-1][:k]

    liked = _user_top_rated(train, user_id, n=2)
    liked_titles = [dataset.title_for(m) for m in liked]
    explanation = (
        f"Because you rated {' and '.join(liked_titles)} highly"
        if liked_titles
        else "Based on overall popularity and your profile"
    )

    recs = []
    for i in order:
        mid = int(candidates[i])
        recs.append(
            {
                "movie_id": mid,
                "title": dataset.title_for(mid),
                "predicted_rating": round(float(scores[i]), 3),
                "explanation": explanation,
            }
        )
    return recs


def success_and_failure_cases(model, dataset, train, test, k=10, relevance=3.5,
                              max_users=500, seed=42):
    """Find one clear success and one clear failure case for the report.

    Success: a high-ranked recommendation that the user really did rate >= 3.5.
    Failure: a high-ranked recommendation the user actually disliked (< 3.5).
    """
    rng = np.random.default_rng(seed)
    test_truth = (
        test.groupby("user_id")
        .apply(lambda g: dict(zip(g["movie_id"], g["rating"])))
        .to_dict()
    )
    users = list(test_truth.keys())
    if len(users) > max_users:
        users = list(rng.choice(users, size=max_users, replace=False))

    success, failure = None, None
    for user_id in users:
        truth = test_truth[user_id]
        seen = set(train[train["user_id"] == user_id]["movie_id"])
        candidates = np.array([m for m in truth if m not in seen])
        if candidates.size == 0:
            continue
        scores = model.predict_batch([user_id] * len(candidates), candidates)
        top = candidates[int(np.argmax(scores))]
        actual = truth[top]
        rec = {
            "user_id": int(user_id),
            "recommended": dataset.title_for(top),
            "predicted_rating": round(float(np.max(scores)), 3),
            "actual_rating": int(actual),
        }
        if success is None and actual >= relevance:
            success = rec
        if failure is None and actual < relevance:
            failure = rec
        if success and failure:
            break
    return {"success_case": success, "failure_case": failure}


def sample_recommendation_report(model, dataset, train, test, n_users=3, k=10,
                                 relevance=3.5, seed=42):
    rng = np.random.default_rng(seed)
    users = train["user_id"].unique()
    sample = rng.choice(users, size=min(n_users, len(users)), replace=False)
    out = {"top_k_examples": [], **success_and_failure_cases(
        model, dataset, train, test, k=k, relevance=relevance, seed=seed)}
    for u in sample:
        out["top_k_examples"].append(
            {"user_id": int(u),
             "recommendations": generate_recommendations(model, dataset, train, u, k=k)}
        )
    return out
