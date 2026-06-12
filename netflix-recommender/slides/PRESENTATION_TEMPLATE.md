# Presentation Scaffold — Personalized Content Discovery (max 8 slides)

> Build this in your tool of choice and export to **PDF**. One `##` block = one
> slide. Replace `‹…›` with real numbers/figures after running the pipeline.

---

## Slide 1 — Title
**Personalized Content Discovery on the Netflix Prize Dataset**
Team ‹names› · ‹dept/year› · Open Projects 2026
One-line hook: *"Turning 100M ratings into the right next watch."*

## Slide 2 — Problem Overview
- Discovery drives engagement & retention on streaming platforms.
- Goal: predict preferences **and** rank the best unseen content.
- Two success axes: rating accuracy (RMSE) vs ranking quality (MAP@10).

## Slide 3 — The Data
- ‹n_ratings› ratings · ‹n_users› users · ‹n_movies› movies · sparsity ‹sparsity›.
- Key visual: rating distribution + long-tail curve.
- Insight callout: top ‹x›% of movies = 80% of ratings.

## Slide 4 — Approach
- Three models compared: Baseline bias → Item-Item CF → Matrix Factorization.
- Per-user temporal split; relevance threshold = 3.5 for MAP@10.
- Diagram: data → EDA → models → eval → Top-K recs.

## Slide 5 — Results
- Results table (RMSE, MAP@10) for all three models.
- Headline: best RMSE = ‹model/value›, best MAP@10 = ‹model/value›.
- Bar chart comparing the two metrics across models.

## Slide 6 — Recommendation Examples
- One user's Top-10 with explanations ("because you liked …").
- Success case ✔ and failure case ✗ side by side.

## Slide 7 — Key Insights
- RMSE-best ≠ MAP-best: accuracy and ranking are different goals.
- Sparsity favours latent factors; rich histories favour item-CF.
- Long tail → coverage/diversity matter for true discovery.

## Slide 8 — Takeaways & Next Steps
- What we'd ship and why.
- Next: neural CF, hybrid metadata model, cold-start strategy, deployment.
- Repo + reproducibility one-liner.
