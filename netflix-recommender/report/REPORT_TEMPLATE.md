# Personalized Content Discovery on the Netflix Prize Dataset
### Technical Report — Open Projects 2026 (AI/ML)

**Team:** ‹name(s)› · **Department/Year:** ‹dept, year›
**Repository:** ‹GitHub URL›

> **How to finalise this report.** Run `python -m src.main ...` on the real
> dataset, then replace every `‹…›` placeholder with the value from
> `results/metrics.json`, `results/eda_insights.json`, and the figures in
> `results/figures/`. Keep it to **10 pages**. Delete these quote-blocks before
> exporting to PDF.

---

## 1. Problem Understanding
Streaming platforms succeed or fail on content discovery. We frame the task as
learning a function `r̂(u, i)` that predicts how user *u* would rate item *i*,
then ranking unseen items to produce personalised Top-K lists. Two things matter
and they are not the same: **rating accuracy** (does the predicted star value
match reality?) and **ranking quality** (are the items we surface actually the
ones the user will love?). We optimise and report both — RMSE for the former,
MAP@10 for the latter — and discuss the trade-off explicitly in §7.

## 2. Exploratory Data Analysis
Dataset: ‹n_ratings› ratings from ‹n_users› users over ‹n_movies› movies;
density ‹density_pct›% (sparsity ‹sparsity›).

- **Rating distribution.** Mean ‹mean_rating›, median ‹median_rating›; the
  distribution is skewed toward 3–4 stars. *(Fig: rating_distribution.png)*
- **User activity.** Ratings per user range ‹min›–‹max› (median ‹median›). A
  small group of power-users contributes a disproportionate share. *(Fig:
  user_activity.png)*
- **Content popularity / long tail.** The top ‹pct_movies_covering_80pct_ratings›%
  of movies account for 80% of all ratings — a steep long tail. *(Fig:
  long_tail.png, movie_popularity.png)*
- **Temporal trend.** Rating volume grows over the dataset window. *(Fig:
  temporal_volume.png)*

**Implications.** High sparsity motivates latent-factor models; the long tail
warns that a popularity-only baseline will look deceptively strong on common
items but fail on discovery; power-user skew motivates per-user bias terms.

## 3. Methodology
**Train/test split.** Per-user *temporal* hold-out: the most recent
‹test_frac›% of each user's ratings form the test set (`train_test_split_temporal`).
This mimics deployment and avoids look-ahead leakage.

**Relevance definition.** For MAP@10, an item is relevant iff the user's true
held-out rating ≥ **3.5**.

**Top-10 procedure.** For each test user, score all unseen items (real run) or a
sampled candidate pool (large-scale eval), rank, and take the top 10.

**MAP@10 computation.** Average Precision@10 per user (rewarding relevant items
ranked high), averaged over evaluated users; users with no relevant items are
excluded. See `src/evaluate.py`.

## 4. Model Design
We implement and compare three approaches of increasing sophistication:

1. **Baseline (bias):** `μ + b_u + b_i` with shrinkage. Fast, strong reference.
2. **Item-Item CF:** adjusted-cosine item similarity, top-‹k› neighbours, bias-corrected.
3. **Matrix Factorization (SGD):** `μ + b_u + b_i + pᵤ·qᵢ`, ‹n_factors› factors,
   L2 reg ‹reg›, lr ‹lr›, ‹n_epochs› epochs.

## 5. Evaluation Metrics
- **RMSE** — rating-prediction accuracy (mandatory).
- **MAP@10** — ranking quality, relevance ≥ 3.5 (mandatory).
- Optional: MAE, Precision@10, Recall@10, NDCG@10.

## 6. Experimental Results

| Model | RMSE ↓ | MAP@10 ↑ | MAE ↓ | Fit time |
|-------|-------:|---------:|------:|---------:|
| Baseline (bias)       | ‹› | ‹› | ‹› | ‹› |
| Item-Item CF          | ‹› | ‹› | ‹› | ‹› |
| Matrix Factorization  | ‹› | ‹› | ‹› | ‹› |

*(Numbers from `results/metrics.json`.)* Briefly: which model wins on RMSE, which
on MAP@10, and at what computational cost.

## 7. Recommendation Examples
Show 1–2 users' Top-10 lists (from `results/sample_recommendations.json`), plus:
- **Success case:** ‹recommended title› predicted ‹x›, actual ‹y› (≥3.5). ✔
- **Failure case:** ‹recommended title› predicted ‹x›, actual ‹y› (<3.5). �’
Discuss *why* (e.g. popularity bias, sparse user history).

## 8. Key Insights
- RMSE and MAP@10 do **not** rank models identically — accurate rating
  prediction ≠ good ranking. ‹state what you observed›
- Item-CF excels for users with rich histories; MF generalises better under sparsity.
- The long tail makes coverage/diversity worth watching (optional metrics).

## 9. Future Improvements
Neural collaborative filtering, ALS at scale (Spark), hybrid models using movie
metadata (year/genre), implicit-feedback signals, and a proper cold-start
strategy (§ optional task).

## 10. Reproducibility
Exact command, git commit, and library versions are saved to
`results/run_config.json`. `README.md` documents setup and the single command to
reproduce every number above.
