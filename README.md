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

## Reproduce
```bash
cd netflix-recommender
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the Netflix Prize files in ../data (see netflix-recommender/README.md), then:
python -m src.main --data-dir ../data --files combined_data_1.txt \
    --sample-users 20000 --models baseline itemcf mf --top-k 10 --relevance 3.5
```

Full setup, data instructions, and design notes: [`netflix-recommender/README.md`](netflix-recommender/README.md).
