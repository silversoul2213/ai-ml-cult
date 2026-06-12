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

from .data_loader import load_dataset, synthetic_ratings, train_test_split_temporal
from .eda import run_eda
from .models import MODEL_REGISTRY
from .evaluate import evaluate_model
from .recommend import sample_recommendation_report


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
        print("[1/5] Loading SYNTHETIC dataset ...")
        dataset = synthetic_ratings(seed=args.seed)
    else:
        print("[1/5] Loading dataset ...")
        dataset = load_dataset(
            args.data_dir, files=args.files, sample_users=args.sample_users,
            min_ratings_per_user=args.min_ratings_per_user, seed=args.seed,
        )
    print(f"      users={dataset.n_users:,}  movies={dataset.n_movies:,}  "
          f"ratings={len(dataset.ratings):,}")

    # 2. EDA
    print("[2/5] Running EDA ...")
    insights = run_eda(dataset, args.out)
    print(f"      sparsity={insights['sparsity']:.5f}  "
          f"mean_rating={insights['mean_rating']}")

    # 3. split
    print("[3/5] Train/test split (per-user temporal hold-out) ...")
    train, test = train_test_split_temporal(
        dataset.ratings, test_frac=args.test_frac, seed=args.seed)
    print(f"      train={len(train):,}  test={len(test):,}")

    # 4. train + evaluate each model
    print("[4/5] Training & evaluating models ...")
    metrics = {}
    fitted = {}
    for name in args.models:
        print(f"  -> {name}")
        model = build_model(name, args.seed)
        ts = time.time()
        model.fit(train)
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

    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    # 5. recommendations from the best model (lowest RMSE)
    print("[5/5] Generating sample recommendations ...")
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
