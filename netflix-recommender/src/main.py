"""
End-to-end pipeline orchestrator.

    python -m src.main --data-dir data --files combined_data_1.txt \
        --sample-users 20000 --models baseline itemcf mf --top-k 10

Use --synthetic to run the whole thing on generated data (no download needed) —
useful for verifying the install and reading example outputs.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import platform
import subprocess

import numpy as np
from scipy.optimize import nnls

from .data_loader import load_dataset, synthetic_ratings, train_test_split_temporal
from .eda import run_eda
from .models import MODEL_REGISTRY
from .evaluate import evaluate_model
from .recommend import sample_recommendation_report


class EnsembleModel:
    """Stacked blend of base models with non-negative weights.

    Weights are learned on a held-out validation slice (NNLS), so the ensemble
    is a proper out-of-fold stack rather than a fit on the training residual.
    """
    name = "Ensemble (stacked blend)"

    def __init__(self, base_models, weights):
        self.base_models = base_models
        self.weights = np.asarray(weights, dtype=float)

    def predict_batch(self, users, movies):
        cols = [m.predict_batch(users, movies) for m in self.base_models]
        return np.clip(np.column_stack(cols) @ self.weights, 1.0, 5.0)

    def predict(self, user_id, movie_id):
        return float(self.predict_batch([user_id], [movie_id])[0])


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_model(name: str, seed: int):
    cls = MODEL_REGISTRY[name]
    if name == "mf":
        return cls(seed=seed)
    return cls()


def main(argv=None):
    p = argparse.ArgumentParser(description="Netflix Prize recommender pipeline")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--files", nargs="*", default=None,
                   help="specific combined_data_*.txt files; default = all found")
    p.add_argument("--sample-users", type=int, default=None)
    p.add_argument("--min-ratings-per-user", type=int, default=5)
    p.add_argument("--models", nargs="+", default=["baseline", "itemcf", "mf"],
                   choices=list(MODEL_REGISTRY))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--relevance", type=float, default=3.5)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--eval-max-users", type=int, default=2000)
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true",
                   help="use generated data instead of the real download")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # 1. load
    if args.synthetic:
        print("[1/6] Loading SYNTHETIC dataset ...")
        dataset = synthetic_ratings(seed=args.seed)
    else:
        print("[1/6] Loading dataset ...")
        dataset = load_dataset(
            args.data_dir, files=args.files, sample_users=args.sample_users,
            min_ratings_per_user=args.min_ratings_per_user, seed=args.seed,
        )
    print(f"      users={dataset.n_users:,}  movies={dataset.n_movies:,}  "
          f"ratings={len(dataset.ratings):,}")

    # 2. EDA
    print("[2/6] Running EDA ...")
    insights = run_eda(dataset, args.out)
    print(f"      sparsity={insights['sparsity']:.5f}  "
          f"mean_rating={insights['mean_rating']}")

    # 3. split (train/test), then carve a validation slice from train used for
    #    MF early-stopping and for learning the ensemble's blend weights.
    print("[3/6] Train/test split (per-user temporal hold-out) ...")
    train, test = train_test_split_temporal(
        dataset.ratings, test_frac=args.test_frac, seed=args.seed)
    train_fit, val = train_test_split_temporal(train, test_frac=0.1, seed=args.seed)
    print(f"      train={len(train):,}  (fit={len(train_fit):,} val={len(val):,})  "
          f"test={len(test):,}")

    # 4. train + evaluate each base model (fitted on train_fit; the held-out val
    #    slice is reserved for early-stopping/blending, evaluation is on test).
    print("[4/6] Training & evaluating base models ...")
    metrics = {}
    fitted = {}
    for name in args.models:
        print(f"  -> {name}")
        model = build_model(name, args.seed)
        ts = time.time()
        if name == "mf":
            model.fit(train_fit, valid=val)
        else:
            model.fit(train_fit)
        fit_time = time.time() - ts
        res = evaluate_model(
            model, train, test, k=args.top_k, relevance=args.relevance,
            max_users=args.eval_max_users, seed=args.seed,
        )
        res["fit_time_sec"] = round(fit_time, 2)
        res["model_name"] = model.name
        metrics[name] = res
        fitted[name] = model
        print(f"     RMSE={res['RMSE']:.4f}  "
              f"MAP@{args.top_k}={res[f'MAP@{args.top_k}']:.4f}  "
              f"({fit_time:.1f}s)")

    # 4b. stacked ensemble: learn non-negative blend weights on the val slice.
    if len(args.models) >= 2:
        print("[5/6] Building stacked ensemble (NNLS blend on val) ...")
        ts = time.time()
        base_list = [fitted[n] for n in args.models]
        vu, vm = val["user_id"].to_numpy(), val["movie_id"].to_numpy()
        vr = val["rating"].to_numpy(dtype=float)
        preds_val = np.column_stack([m.predict_batch(vu, vm) for m in base_list])
        weights, _ = nnls(preds_val, vr)
        ens = EnsembleModel(base_list, weights)
        res = evaluate_model(
            ens, train, test, k=args.top_k, relevance=args.relevance,
            max_users=args.eval_max_users, seed=args.seed,
        )
        res["fit_time_sec"] = round(time.time() - ts, 2)
        res["model_name"] = ens.name
        res["weights"] = {args.models[i]: round(float(weights[i]), 4)
                          for i in range(len(args.models))}
        metrics["ensemble"] = res
        fitted["ensemble"] = ens
        print(f"     weights={res['weights']}")
        print(f"     RMSE={res['RMSE']:.4f}  "
              f"MAP@{args.top_k}={res[f'MAP@{args.top_k}']:.4f}")

    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    # 6. recommendations from the best model (lowest RMSE)
    print("[6/6] Generating sample recommendations ...")
    best = min(metrics, key=lambda m: metrics[m]["RMSE"])
    print(f"      best-by-RMSE model: {best}")
    rec_report = sample_recommendation_report(
        fitted[best], dataset, train, test,
        k=args.top_k, relevance=args.relevance, seed=args.seed)
    rec_report["model_used"] = best
    with open(os.path.join(args.out, "sample_recommendations.json"), "w") as fh:
        json.dump(rec_report, fh, indent=2, default=str)

    # run config for reproducibility
    config = {
        "argv": sys.argv,
        "args": vars(args),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "run_config.json"), "w") as fh:
        json.dump(config, fh, indent=2, default=str)

    print(f"\nDone in {config['elapsed_sec']}s. Artifacts in '{args.out}/'.")
    print("Summary:")
    for name, m in metrics.items():
        print(f"  {name:10s} RMSE={m['RMSE']:.4f}  MAP@{args.top_k}={m[f'MAP@{args.top_k}']:.4f}")


if __name__ == "__main__":
    main()
