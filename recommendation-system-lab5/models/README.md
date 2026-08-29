# Models

Recommendation models for Lab 5 & Lab 6, plus cold-start handling and a
router that picks a strategy per user.

## Identifiers

Processed CSVs key movies by `movie_id` (TMDB id). `ratings.csv` also has
`movielens_id`, but every model here joins on `movie_id`.

## Files

| File | What it does |
|---|---|
| `content_lightFM.py` | Content-based embeddings (SentenceTransformer over title + genres) combined with LightFM as a hybrid model, implicit feedback. |
| `lightfm_model.py` | LightFM collaborative filtering — factorization machines on interaction data only, WARP loss. |
| `lightgcn_model.py` | LightGCN — graph convolution over the user-item bipartite graph, BPR loss with negative sampling. |
| `cold_start_questionnaire.py` | Mini-questionnaire (genres, favorite movies, release-era preference) → content-based recommendations for users with no rating history. |
| `recommendation_router.py` | Routes a user to a strategy based on how many ratings they have. |
| `cold_start_examples.py` | Runnable examples exercising the questionnaire and router. |
| `model_creation.ipynb` | Exploratory notebook: SVD baseline, then FAISS-based content retrieval. |
| `lightfm_roc_curve.png` | ROC curve from a LightFM evaluation run. |

## User lifecycle and routing

`recommendation_router.py` (`UnifiedRecommendationRouter`) switches strategy
by rating count:

| Stage | Ratings | Strategy | Why |
|---|---|---|---|
| New | 0 | Questionnaire → content-based | No history to collaborate on |
| Cold start | 1–5 | Content-based | Still not enough signal for collaborative filtering |
| Warm start | 5–20 | Hybrid (content + collaborative) | Some history, content still fills gaps |
| Mature | 20+ | Collaborative (LightGCN / LightFM) | Enough history for full collaborative filtering |

```python
from recommendation_router import UnifiedRecommendationRouter
import pandas as pd

movies = pd.read_csv("data/processed/movies.csv")
ratings = pd.read_csv("data/processed/ratings.csv")
router = UnifiedRecommendationRouter(movies, ratings)

router.register_new_user(user_id=12345)
result = router.get_recommendations(user_id=12345, n_recommendations=10)
print(result.strategy, result.user_stage)

router.record_user_rating(user_id=12345, movie_id=123, rating=4.5)
```

## Running the models

```powershell
python models/cold_start_questionnaire.py     # interactive: genres, favorite movies, era preference
python models/recommendation_router.py         # demo of the full lifecycle
python models/cold_start_examples.py           # non-interactive usage examples
python models/lightfm_model.py                 # LightFM collaborative filtering
python models/content_lightFM.py               # content-based + LightFM hybrid
python models/lightgcn_model.py                # LightGCN
```

All scripts expect `data/processed/ratings.csv` (columns `userId`/`user_id`,
`movieId`/`movie_id`, `rating` — LightGCN auto-renames `user_id`→`userId` if
needed) and `data/processed/movies.csv`.

## LightGCN details

`lightgcn_model.py` provides:

- `prepare_lightgcn_data()` — builds the user-item bipartite graph from ratings (filtered to `rating >= 4.0`).
- `LightGCN` — the model: K graph-convolution layers, embeddings averaged across layers.
- `train_lightgcn()` — BPR loss with negative sampling.
- `evaluate_model()` — Precision@K, Recall@K, NDCG@K.

Default parameters (in `main()`): `emb_dim=64`, `n_layers=3`, `lr=0.001`,
`epochs=20`, `batch_size=1024`. Requires `torch>=2.0.0` and
`torch-geometric>=2.3.0` (in `requirements.txt`); install with:

```powershell
install_lightgcn_deps.bat
```

or manually:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric tqdm
```

Runnable usage examples (basic run, single-user recommendations, comparison
against LightFM, embedding inspection) are in `lightgcn_examples.py` at the
repo root.

## Recorded results

These are results from prior runs, not guaranteed to reproduce exactly
(dataset sampling and training are stochastic). See
[README_lightFMresults.md](README_lightFMresults.md) for the full LightFM
run log.

**LightFM** — ROC AUC (sampled 500 users): **0.7277**.

**LightGCN** — on a run over ~600K ratings (610 evaluated users):

| K | Precision@K | Recall@K | NDCG@K |
|---|---|---|---|
| 10 | 0.0324 | 0.0652 | 0.0421 |
| 20 | 0.0198 | 0.0823 | 0.0527 |

For an apples-to-apples comparison across all four models on the same
split, use `evaluation/compare_models.py` — see the root README.

## Evaluation strategy notes

- Content-based and LightGCN are ranking models; they are scored with
  Precision@K, Recall@K, NDCG@K only.
- LightFM outputs raw scores that can be normalized to a 1–5 scale, so it's
  additionally scored with RMSE/MAE (see `evaluation/compare_models.py`).
- LightGCN training is CPU-slow relative to LightFM (minutes vs. seconds on
  this dataset size) because it iterates graph convolutions per epoch;
  LightFM converges faster but does not model the graph structure directly.
