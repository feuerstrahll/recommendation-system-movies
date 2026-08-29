# Models

Recommendation models for Lab 5 & Lab 6, plus cold-start handling and a
router that picks a strategy per user.

## Identifiers

Processed CSVs key movies by `movie_id` (TMDB id). `ratings.csv` also has
`movielens_id`, but every model here joins on `movie_id`.

## Files

| File | What it does |
|---|---|
| `content_based.py` | Content-based strategy — SentenceTransformer embeddings over title + genres + release year, ranked by cosine similarity (FAISS if installed, brute-force otherwise). No collaborative signal, so it can recommend cold-start items. |
| `svd_model.py` | SVD collaborative filtering — classic matrix factorization (scipy truncated SVD) on implicit interactions. |
| `lightfm_model.py` | LightFM collaborative filtering — factorization machines on interaction data only, WARP loss. |
| `lightgcn_model.py` | LightGCN — graph convolution over the user-item bipartite graph, BPR loss with negative sampling. |
| `cold_start_questionnaire.py` | Mini-questionnaire (genres, favorite movies, release-era preference) → content-based recommendations for users with no rating history. |
| `recommendation_router.py` | Routes a user to a strategy based on how many ratings they have. |
| `cold_start_examples.py` | Runnable examples exercising the questionnaire and router. |
| `model_creation.ipynb` | Exploratory notebook: SVD baseline, then FAISS-based content retrieval. |
| `lightfm_roc_curve.png` | ROC curve from an earlier, exploratory LightFM run (see `README_lightFMresults.md`). |

The four collaborative/content models (`content_based.py`, `svd_model.py`,
`lightfm_model.py`, `lightgcn_model.py`) each have their own runnable
`train_and_evaluate()`/`main()` and share one evaluation protocol — see
below.

## User lifecycle and routing

`recommendation_router.py` (`UnifiedRecommendationRouter`) switches strategy
by rating count:

| Stage | Ratings | Strategy | Why |
|---|---|---|---|
| New | 0 | Questionnaire → content-based | No history to collaborate on |
| Cold start | 1–5 | Content-based | Still not enough signal for collaborative filtering |
| Warm start | 5–20 | Hybrid (content + collaborative) | Some history, content still fills gaps |
| Mature | 20+ | Collaborative (LightGCN / LightFM / SVD) | Enough history for full collaborative filtering |

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
python models/content_based.py                 # content-based (SentenceTransformer + cosine/FAISS)
python models/svd_model.py                     # SVD collaborative filtering
python models/lightfm_model.py                 # LightFM collaborative filtering
python models/lightgcn_model.py                # LightGCN
```

All four scripts expect `data/processed/ratings.csv` (`user_id`, `movie_id`,
`rating` — LightGCN's internal helper renames these to `userId`/`movieId`)
and `data/processed/movies.csv`. `content_based.py` also needs the FAISS
package to build an index (`pip install faiss-cpu`, in `requirements.txt`);
without it, it falls back to brute-force cosine similarity automatically.

## Shared evaluation protocol (SVD / LightFM / LightGCN)

`svd_model.py`, `lightfm_model.py`, and `lightgcn_model.py` each implement
the same preprocessing and evaluation pipeline independently (so each is
runnable on its own), and `evaluation/compare_models.py` reuses the same
functions to compare all three (plus content-based) on one shared split:

- **Binarization**: `rating >= 4.0` counts as a positive interaction;
  everything else is dropped. All three models train on implicit 0/1
  feedback, not raw explicit ratings.
- **Filtering**: only users with at least 5 positive interactions are kept.
- **Split**: a strict 80/20 split **per user** (not a global random split),
  so every user has both train and test rows.
- **Evaluation**: items already in a user's train set are excluded from
  their top-K at scoring time; test interactions on items never seen in
  train (cold-start items with no learned embedding/factor) are dropped
  from ground truth before computing metrics.
- **Metrics**: Precision@K, Recall@K, NDCG@K for K = 10, 20 — the only
  metrics reported. RMSE/MAE were dropped: with binarized interactions
  there's no rating scale left to score predictions against.

## LightGCN details

`lightgcn_model.py` provides:

- `prepare_lightgcn_data()` — builds the user-item bipartite graph from ratings.
- `LightGCN` — the model: K graph-convolution layers, embeddings averaged across layers.
- `train_lightgcn()` — BPR loss with negative sampling; graph propagation runs once
  per epoch (not once per mini-batch), with BPR loss computed in batches only to
  bound memory, and one `backward()`/`optimizer.step()` per epoch.
- `evaluate_model()` — Precision@K, Recall@K, NDCG@K on held-out data.

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

`README_lightFMresults.md` documents an early, exploratory LightFM run (ROC
AUC ~0.73) that predates the shared evaluation protocol above and used a
different, non-per-user split — it's kept as a historical log, not a
number to compare against current runs.

For current, directly comparable Precision@K/Recall@K/NDCG@K across all
four models on the same split, run:

```powershell
python evaluation/compare_models.py
```
