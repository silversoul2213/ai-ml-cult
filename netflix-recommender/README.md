# Personalized Content Discovery — Netflix Prize Recommender

A reproducible recommendation system built on the **Netflix Prize Dataset**
(100M ratings, 480K users, 17.7K movies). It learns user preferences, predicts
unseen ratings, and generates personalized Top-K recommendations.

The pipeline implements and **compares multiple recommendation approaches**, and
evaluates them with the two mandatory metrics: **RMSE** (rating accuracy) and
**MAP@10** (ranking quality, relevance threshold = 3.5).

---

## 1. What's implemented

| Stage | Module | Output |
|-------|--------|--------|
| Data loading & parsing | `src/data_loader.py` | tidy ratings DataFrame + movie titles |
| Exploratory Data Analysis | `src/eda.py` | figures in `results/figures/`, `results/eda_insights.json` |
| Models | `src/models/` | Global/Bias baseline, Item-Item CF, Matrix Factorization (SGD) |
| Evaluation | `src/evaluate.py` | RMSE + MAP@10 (+ optional MAE, Precision@K, Recall@K, NDCG) |
| Recommendations | `src/recommend.py` | Top-K lists + human-readable explanations |
| Orchestrator | `src/main.py` | runs the full pipeline end to end |

Models (at least two, as required):
- **BaselineModel** — global mean + user/item bias terms (strong, fast reference).
- **ItemItemCF** — item-based collaborative filtering (cosine / adjusted-cosine).
- **MatrixFactorization** — latent-factor SGD (the SVD-style approach that won the Prize era), with bias terms and L2 regularization.

---

## 2. Getting the data

The dataset is **not** included (it is ~2 GB). Download it from Kaggle:

> https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

Unzip so the project sees:

```
data/
  combined_data_1.txt
  combined_data_2.txt
  combined_data_3.txt
  combined_data_4.txt
  movie_titles.csv
```

Each `combined_data_*.txt` block looks like:

```
1:                 <- movie id, terminated by a colon
1488844,3,2005-09-06   <- user_id, rating, date
822109,5,2005-05-13
...
```

`movie_titles.csv` is `movie_id,year,title` (latin-1 encoded; titles may contain commas).

> **Tip — the full set is huge.** Use `--sample-users` to train on a random
> subset of users (the loader keeps every rating *for those users*, preserving
> per-user density). Start with `--sample-users 20000` to develop, then scale up.

---

## 3. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Run

```bash
# Full pipeline on a 20k-user subset of files 1 & 2
python -m src.main \
    --data-dir data \
    --files combined_data_1.txt combined_data_2.txt \
    --sample-users 20000 \
    --models baseline itemcf mf \
    --top-k 10 --relevance 3.5 \
    --out results
```

Outputs land in `results/`:
- `figures/` — EDA charts (rating distribution, sparsity, activity, popularity).
- `eda_insights.json` — numeric EDA summary you can quote in the report.
- `metrics.json` — RMSE and MAP@10 (and optional metrics) per model.
- `sample_recommendations.json` — Top-10 lists with explanations, success/failure cases.

You can also run stages individually — see `python -m src.main --help`.

## 5. Reproducibility

Every random operation is seeded (`--seed`, default 42). The exact command,
git commit, and library versions are written to `results/run_config.json` so a
run can be reproduced exactly.

## 6. Repo layout

```
netflix-recommender/
├── README.md
├── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── recommend.py
│   ├── main.py
│   └── models/
│       ├── __init__.py
│       ├── baseline.py
│       ├── item_cf.py
│       └── matrix_factorization.py
├── report/REPORT_TEMPLATE.md        <- 10-page technical report scaffold
├── slides/PRESENTATION_TEMPLATE.md  <- 8-slide deck scaffold
└── results/                         <- generated artifacts (gitignored except samples)
```

## 7. Note on results in the report

Numbers (RMSE, MAP@10, charts) are produced by **running the code on the real
dataset**. The report and slide templates have clearly marked `‹fill in›`
placeholders for these — run the pipeline, then paste the values from
`results/metrics.json` and the figures from `results/figures/`.
