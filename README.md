# Personalized Content Discovery — Netflix Prize Recommender

**Open Projects 2026 · AI / ML** &nbsp;·&nbsp; **Team rukdimaxx**
Somil Agrawal (23112099) · S R Nivedhitha (22411030)

A reproducible recommendation system built on the **Netflix Prize dataset**
(100M ratings · 480K users · 17.7K movies). It learns user preferences, predicts
unseen ratings, and generates personalized Top-K recommendations — comparing four
approaches and evaluating them on the two mandatory metrics, **RMSE** and
**MAP@10** (relevance ≥ 3.5).

## Deliverables
| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Technical Report (PDF) | [`netflix-recommender/deliverables/Technical-Report.pdf`](netflix-recommender/deliverables/Technical-Report.pdf) |
| 2 | Source code & pipeline | [`netflix-recommender/`](netflix-recommender/) |
| 3 | Presentation (PDF) | [`netflix-recommender/deliverables/Presentation.pdf`](netflix-recommender/deliverables/Presentation.pdf) |

## Headline result
| Model | RMSE ↓ | MAP@10 ↑ |
|-------|-------:|---------:|
| Baseline (bias) | 0.9399 | **0.1843** |
| Item-Item CF | 0.9496 | 0.0854 |
| Matrix Factorization | 0.9118 | 0.1177 |
| **Ensemble (stacked)** | **0.9053** | 0.1419 |

The stacked **Ensemble wins on rating accuracy**; the trivial **Baseline wins on
ranking**. Accuracy and discovery are different objectives — and our ensemble
proves it by giving the baseline a blend weight of exactly zero while minimizing
error.

## What's implemented
| Stage | Module | Output |
|-------|--------|--------|
| Data loading & parsing | `src/data_loader.py` | tidy ratings DataFrame + movie titles |
| Exploratory Data Analysis | `src/eda.py` | figures + `results/eda_insights.json` |
| Models | `src/models/` + `src/main.py` | Baseline, Item-Item CF, Matrix Factorization (SGD), stacked Ensemble |
| Evaluation | `src/evaluate.py` | RMSE + MAP@10 (+ MAE, Precision@K, Recall@K, NDCG) |
| Recommendations | `src/recommend.py` | Top-K lists + plain-language explanations |
| Orchestrator | `src/main.py` | runs the full pipeline end to end |

**Models (four; at least two required):** `BaselineModel` (μ + user/item bias),
`ItemItemCF` (adjusted cosine with significance shrinkage + min co-rating
overlap), `MatrixFactorization` (50-factor SGD, L2, validation early-stopping),
and a non-negative least-squares **stacked Ensemble** with weights learned on a
held-out validation slice.

## Get the data
The dataset (~2 GB) is **not** committed. Download it from Kaggle and unzip into a
`data/` folder **at the repo root**:

> https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

```
ai-ml-cult/
└── data/
    ├── combined_data_1.txt   # "1:" introduces a movie id; then "user_id,rating,date"
    ├── combined_data_2.txt
    ├── combined_data_3.txt
    ├── combined_data_4.txt
    └── movie_titles.csv       # movie_id,year,title  (latin-1)
```

## Setup & run
```bash
cd netflix-recommender
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Reproduce the reported numbers (20k-user subsample of file 1, seed 42):
python -m src.main --data-dir ../data --files combined_data_1.txt \
    --sample-users 20000 --models baseline itemcf mf --top-k 10 --relevance 3.5 \
    --out results
```
Sampling *users* (not rows) preserves per-user density; subsetting is explicitly
allowed by the brief. Use `python -m src.main --synthetic` to smoke-test the whole
pipeline with no download. See `python -m src.main --help` for all options.

Outputs land in `netflix-recommender/results/` (git-ignored): `figures/` (EDA
charts), `eda_insights.json`, `metrics.json`, `sample_recommendations.json`, and
`run_config.json`.

## Reproducibility
Every random operation is seeded (`--seed`, default 42). The exact command, git
commit, and library versions are written to `results/run_config.json`, so any run
reproduces exactly.

## Repository layout
```
ai-ml-cult/
├── README.md
└── netflix-recommender/
    ├── requirements.txt
    ├── src/
    │   ├── data_loader.py · eda.py · evaluate.py · recommend.py
    │   ├── main.py                 # orchestrator + stacked ensemble
    │   └── models/                 # baseline.py · item_cf.py · matrix_factorization.py
    ├── deliverables/
    │   ├── Technical-Report.pdf    # Deliverable 1
    │   └── Presentation.pdf        # Deliverable 3
    └── results/                    # generated artifacts (git-ignored)
```
