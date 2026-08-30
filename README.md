# Movie Recommendation System

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

Movie dataset preparation, a PostgreSQL/SQLite schema, and four recommendation
approaches (content-based, SVD, LightFM, LightGCN) built on
[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

This is the active development branch: model implementations, database
tooling, evaluation, and a Streamlit UI. `main` was merged to match this
branch; see [CONTRIBUTING.md](CONTRIBUTING.md) for the current workflow.

## Table of Contents

- [Structure](#structure)
- [Architecture](#architecture)
- [Data Contract](#data-contract)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Demo & Screenshots](#demo--screenshots)
- [Future Work](#future-work)
- [License](#license)

## Structure

```
app.py          Streamlit UI — login/profile, search, and all four recommendation strategies
data/           check.py, dataloader.py — local data utilities (raw/processed CSVs are not committed)
database/       schema, cleaning script, integrity checks, ER diagram — see database/README.MD
etl/            etl_pipeline.py — loads processed CSVs into database/recommender.db (SQLite)
models/         4 recommendation models + cold-start questionnaire + router — see models/README.md
evaluation/     shared metrics (Precision@K, Recall@K, NDCG@K) + evaluation/compare_models.py
UI_screens/     screenshots of the Streamlit app (auth, search, cold start, collaborative, hybrid)
docs/demo/      exported standalone HTML builds of the UI
```

## Architecture

```
Raw Movies Dataset (Kaggle)
        │
        ▼
database/clean_movies_data.py  ──▶  data/processed/*.csv  (TMDB-keyed)
        │
        ▼
etl/etl_pipeline.py  ──▶  database/recommender.db (SQLite) / PostgreSQL (database/schema.sql)
        │
        ▼
   models/  ──┬── content_based.py     (TF-IDF / SentenceTransformer + FAISS)
              ├── svd_model.py         (truncated SVD)
              ├── lightfm_model.py     (factorization machines, WARP loss)
              └── lightgcn_model.py    (graph convolution, BPR loss)
        │
        ├──▶ evaluation/compare_models.py  ──▶  Precision@K / Recall@K / NDCG@K
        │
        └──▶ models/recommendation_router.py  ──▶  app.py (Streamlit UI)
```

New users are routed by rating count — new → cold start → warm start → mature
— through [`models/recommendation_router.py`](models/recommendation_router.py),
which picks a strategy per lifecycle stage. See
[models/README.md](models/README.md) for the full routing table.

## Data Contract

Movie identity across the whole project is the TMDB id:

```text
movies.movie_id  = TMDB id
ratings.movie_id = TMDB id
ratings.movielens_id = original MovieLens movieId (kept for traceability only)
```

See [database/database_architecture.md](database/database_architecture.md) for the full table layout.

## Setup

```powershell
pip install -r requirements.txt
```

`requirements.txt` covers the classic stack (pandas, scikit-learn, LightFM,
Streamlit) plus PyTorch + PyTorch Geometric for LightGCN.

## Data Preparation

```powershell
python database/clean_movies_data.py --input data/raw --output data/processed
```

Reads the raw Movies Dataset CSVs (`movies_metadata.csv`, `credits.csv`,
`keywords.csv`, `ratings.csv`, `links.csv`) and writes the cleaned,
TMDB-keyed CSVs consumed by every model and by the SQL schema.

## Models

Four approaches live in `models/`, meant to be compared rather than picked
as a single "winner":

| Model | File | Approach |
|---|---|---|
| Content-Based | `models/content_based.py` | SentenceTransformer embeddings over title + genres + year, cosine similarity / FAISS |
| SVD | `models/svd_model.py` | Classic matrix factorization (scipy truncated SVD) |
| LightFM | `models/lightfm_model.py` | Factorization machines, WARP loss, interaction data only |
| LightGCN | `models/lightgcn_model.py` | Graph convolution over the user-item bipartite graph, BPR loss |

`evaluation/compare_models.py` additionally benchmarks a lightweight
TF-IDF content-based baseline and a LightFM + item-features hybrid variant
alongside these four — see [Evaluation](#evaluation) below.

Cold-start users are handled separately by `models/cold_start_questionnaire.py`
and routed through the lifecycle stages (new → cold start → warm start →
mature) in `models/recommendation_router.py`. Details, parameters, and
recorded results: [models/README.md](models/README.md).

## Evaluation

```powershell
python evaluation/compare_models.py
```

Runs Content-Based (TF-IDF), SVD, LightFM Collaborative, LightFM Hybrid, and
LightGCN on the exact same preprocessed data (implicit interactions, rating
>= 4.0, 80/20 split per user) and reports Precision@K, Recall@K, NDCG@K for
K = 10, 20. Training exposure per epoch is equalized across models — see
[models/README.md](models/README.md) for the full protocol and methodology
notes (how LightGCN's epoch budget was aligned with LightFM's, and why the
benchmarked "Content-Based (TF-IDF)" differs from `models/content_based.py`).

Metric implementations: `evaluation/metrics.py`.

## Results

| Model | P@10 | R@10 | NDCG@10 | Train time |
|---|---:|---:|---:|---:|
| Content-Based (TF-IDF) | 0.0030 | 0.0050 | 0.0037 | 0.0s |
| SVD | 0.1387 | **0.1766** | 0.2137 | **4.9s** |
| **LightFM Collab** | **0.1510** | 0.1857 | **0.2177** | 41.1s |
| LightFM Hybrid | 0.1410 | 0.1559 | 0.2025 | 108.4s |
| LightGCN | 0.0173 | 0.0108 | 0.0192 | 13051.5s |

LightFM Collaborative is numerically best, though its margin over SVD is
close to the run-to-run noise Collab itself shows across the two runs
recorded so far — SVD trails by only ~2% on NDCG@10/MAP@10 (up to ~8% on
other metrics) at a fraction of the training/inference cost. LightGCN is
not production-ready: even its best configuration trails SVD by a wide
margin at roughly 790x the training cost, and a recent round of tuning
(embedding normalization + a smaller batch size, changed together) made it
*worse*, not better, including zero correct recommendations in the eval's
lowest interaction-count band (5-9 interactions — not true cold-start
users) this run; see RESULTS.md for the full regression analysis. Full
metrics (P/R/NDCG/MAP@10/20, catalog coverage, interaction-count segments,
LightFM score-distribution diagnostics, methodology, and known
limitations): **[RESULTS.md](RESULTS.md)**. Raw per-run numbers:
[results/metrics.csv](results/metrics.csv).

**Score distribution diagnostic (LightFM):** ROC-AUC alone can hide *why*
a model separates positives from negatives poorly — a step-like or
clustered ROC curve usually means score ties or clusters somewhere in the
distribution, which a single AUC number won't show. These histograms plot
each model's raw predicted scores for held-out positive items against a
sampled set of negative items, so separation quality is visible directly:

| LightFM Collab | LightFM Hybrid |
|---|---|
| ![LightFM Collab score distribution](results/score_hist_lightfm_collab.png) | ![LightFM Hybrid score distribution](results/score_hist_lightfm_hybrid.png) |

Both show clean, well-separated, unimodal positive/negative distributions
with a modest overlap region — no clustering or step artifacts. This was
checked specifically because an older, unrelated exploratory run
(`models/lightfm_roc_curve.png`, predating the current evaluation protocol)
had a visibly step-like ROC curve; these histograms confirm that artifact
doesn't reproduce under the current pipeline.

## Demo & Screenshots

`docs/demo/` contains exported standalone HTML builds of the UI
(`movie-recommendation-system.html` and a compressed variant).

| | |
|---|---|
| ![Login](UI_screens/login.png) Login | ![Search](UI_screens/search.png) Search |
| ![Cold start](UI_screens/coldStart.png) Cold-start questionnaire | ![Collaborative](UI_screens/collab_sys.png) Collaborative recommendations |
| ![Hybrid](UI_screens/hybridSys1.png) Hybrid recommendations | ![Personal account](UI_screens/personal_acc.png) Personal account / rating history |

More screenshots (auth flow, content-based results, before/after
collaborative filtering comparisons) are in [UI_screens/](UI_screens/).

## Future Work

- Load-testing and scalability benchmarks for the production-fit analysis
  (currently qualitative only — see `models/README.md`)
- Wire the SentenceTransformer + FAISS content engine (`models/content_based.py`)
  into `evaluation/compare_models.py` for a stronger content-based baseline
- Hyperparameter tuning pass for LightGCN and LightFM Hybrid beyond the
  current fixed configuration

## Data sources

- [MovieLens](https://grouplens.org/datasets/movielens/)
- [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Review the terms and licensing of the code, datasets, and third-party
dependencies before redistribution or deployment. This documentation package
does not introduce a software license.

## License

[MIT](LICENSE)
