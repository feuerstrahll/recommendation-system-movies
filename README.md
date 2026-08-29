# Movie Recommendation System

Lab 5: comparing recommendation approaches on a movie dataset and choosing
a strategy for production use.

The active project lives in
[`recommendation-system-lab5/`](recommendation-system-lab5/) — start there.
It contains the data pipeline (raw → cleaned CSVs → PostgreSQL/SQLite),
four recommendation models, a cold-start router, and unified evaluation.
See [`recommendation-system-lab5/README.md`](recommendation-system-lab5/README.md)
for setup and usage.

## Lab requirements coverage

| # | Requirement | Status |
|---|---|---|
| 1 | Theory: content-based, collaborative, LightFM, GNN | Covered — see `models/README.md` |
| 2 | Dataset: users, items, interactions, metadata | Covered — see `database/database_architecture.md` |
| 3 | Implement 4 model types | Covered — content-based, LightFM (collaborative), LightFM (hybrid), LightGCN, all in `models/` |
| 4 | Evaluation: Precision@K, Recall@K, NDCG@K, RMSE | Covered — `evaluation/compare_models.py` runs all four models on one split and reports ranking metrics for all of them plus RMSE/MAE for the two LightFM variants |
| 5 | Production-fit analysis (quality, speed, scalability, real-time, update cost) | Partial — training/inference speed and update cost are discussed per-model in `models/README.md`; no load-testing or scalability benchmarks exist |
| 6 | Model selection + UI integration | Partial — routing logic (`models/recommendation_router.py`) picks a strategy per user; `docs/demo/` holds exported HTML UI mockups, not a live backend-connected app |

`FULL_LAB_STATUS.py` in this folder is an earlier, now partly outdated
self-check script — it predates `evaluation/compare_models.py` and lists
RMSE support as missing, which is no longer accurate.

## Repository layout

```
README.md                       this file
recommendation-system-lab5/     active project — see its own README.md
movie-recommendation-system.html  standalone UI export (superseded by docs/demo/ inside the project folder)
```
