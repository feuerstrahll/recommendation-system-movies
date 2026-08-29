# Movie Recommendation System

Movie dataset preparation, a PostgreSQL/SQLite schema, and four recommendation
approaches (content-based, SVD, LightFM, LightGCN) built on
[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

This branch (`feature/dev`) holds active development: model implementations,
database tooling, evaluation, and a Streamlit UI. The `main` branch holds an
earlier baseline (SVD + FAISS hybrid recommender) and is kept as-is.

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

## Data contract

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

## Data preparation

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

Cold-start users are handled separately by `models/cold_start_questionnaire.py`
and routed through the lifecycle stages (new → cold start → warm start →
mature) in `models/recommendation_router.py`. Details, parameters, and
recorded results: [models/README.md](models/README.md).

## Evaluation

```powershell
python evaluation/compare_models.py
```

Runs Content-Based, SVD, LightFM, and LightGCN on the exact same
preprocessed data (implicit interactions, rating >= 4.0, 80/20 split per
user) and reports Precision@K, Recall@K, NDCG@K for K = 10, 20 — the same
protocol every model's own script uses standalone, so results are directly
comparable. Metric implementations: `evaluation/metrics.py`.

Details and the shared protocol: [models/README.md](models/README.md).

## Demo

`docs/demo/` contains exported standalone HTML builds of the UI
(`movie-recommendation-system.html` and a compressed variant). Screenshots
of the live Streamlit app are in `UI_screens/`.
