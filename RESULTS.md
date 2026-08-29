# Results

**Last updated:** 2026-08-30
**Run commit:** [`9c33b17`](https://github.com/feuerstrahll/recommendation-system-movies/commit/9c33b17) — post training-loop fixes, pre catalog-coverage feature (see [Limitations](#limitations--future-work))
**Raw numbers:** [results/metrics.csv](results/metrics.csv)

## Executive Summary

Across five recommendation approaches evaluated on 131K users and 26K movies
from the combined MovieLens/TMDB dataset, **LightFM Collaborative Filtering**
delivered the best ranking accuracy (NDCG@10 = 0.218), narrowly ahead of a
much simpler and roughly 9x cheaper-to-train **SVD** baseline (NDCG@10 =
0.214) — the closeness of that gap is itself a notable finding. The bigger
surprise is that graph-based **LightGCN** underperformed both factorization
approaches (NDCG@10 = 0.159) despite training exposure now matched to
LightFM's; its loss was still declining at the final epoch, which points to
undertraining rather than an architectural ceiling. **LightFM Hybrid**
(item-feature-augmented) trailed plain Collaborative filtering by a small,
consistent margin on these accuracy metrics — expected, since most
evaluation users already have substantial rating history, so the hybrid's
real advantage (serving cold-start items) doesn't show up in aggregate
ranking numbers computed mostly over warm users. For production, we'd
recommend **LightFM Collaborative** for accuracy-critical, established-user
segments, **LightFM Hybrid** specifically for cold/light users, and **SVD**
as a strong, much cheaper fallback where infrastructure cost matters more
than the last few points of NDCG.

## Methodology

**Dataset:** 131,351 users, 26,461 items, drawn from 7,608,221 positive
interactions (rating ≥ 4.0) after binarizing MovieLens ratings joined
against TMDB movie metadata.

**Split:** an 80/20 split applied *per user* (not a global random split),
so every user has both train and test rows: 5,984,633 train interactions,
1,561,736 test interactions. 1,653 test interactions were dropped from
ground truth because they involve items with zero training interactions
(no model here can recommend an item it never saw) — this is a real
evaluation limitation, not something to gloss over; see
[Limitations](#limitations--future-work).

**Evaluation sample:** 300 users sampled with a fixed seed (42), evaluated
identically across all five models so differences reflect the model, not
the eval sample.

**Metrics** (K = 10, 20):

- **Precision@K** — of the K items shown, what fraction were actually
  relevant (present in the user's held-out test interactions)?
- **Recall@K** — of the user's relevant held-out items, what fraction did
  the top-K surface?
- **NDCG@K** — like Precision, but rewards relevant items ranked *higher*
  within the top-K more than ones near the bottom; the single number we
  lean on most for a one-line comparison since it captures ranking quality,
  not just hit/miss.

Full protocol detail (binarization threshold, per-model preprocessing,
LightGCN/LightFM epoch equalization): [models/README.md](models/README.md).

## Results

| Model | P@10 | R@10 | NDCG@10 | NDCG@20 | Train (s) | Infer (ms) | NDCG@10 / train-sec (×1000) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Content-Based (TF-IDF) | 0.0030 | 0.0050 | 0.0037 | 0.0045 | 0.00 | 2.34 | n/a — no training step |
| SVD | 0.1387 | **0.1766** | 0.2137 | 0.2273 | **4.19** | **0.07** | **51.05** |
| **LightFM Collab** | **0.1490** | 0.1760 | **0.2178** | **0.2374** | 36.26 | 1.70 | 6.01 |
| LightFM Hybrid | 0.1440 | 0.1652 | 0.2038 | 0.2196 | 99.89 | 2.56 | 2.04 |
| LightGCN | 0.1133 | 0.1393 | 0.1592 | 0.1760 | 3310.70 | 0.10 | 0.05 |

Bold marks the best value per column. Full K=10/20 P/R/NDCG matrix:
[results/metrics.csv](results/metrics.csv).

**Note on Content-Based:** its near-zero accuracy here is expected, not a
bug — this is a pure TF-IDF/genre similarity baseline with no collaborative
signal, evaluated on warm users where collaborative methods have a strong
advantage. Its actual value is serving users collaborative models can't
(see [Production Readiness](#production-readiness) and
[models/README.md](models/README.md) for the cold-start-specific numbers).

## Production Readiness

| Model | Cold Start | Scalability | Real-Time | Update Cost | Complexity |
|---|---|---|---|---|---|
| Content-Based (TF-IDF) | Excellent | Good — O(items) | Yes | Simple (metadata refresh) | Low |
| SVD | Poor | Medium — O(users × items) | No | Full retrain | Low |
| LightFM Collab | Poor | Medium — O(users × items) | No | Full retrain | Medium |
| LightFM Hybrid | Good (item features) | Medium | Partial | Full retrain | Medium |
| LightGCN | Poor | Medium — O(edges) | No | Expensive (graph rebuild) | High |

## Discussion

**SVD is nearly as accurate as LightFM Collab at a fraction of the cost.**
The NDCG@10 gap is 1.9% relative, but SVD trains ~8.7x faster and predicts
~23x faster per user. If infrastructure cost or latency budget is tight,
SVD is a legitimate production choice, not just a baseline to beat.

**LightGCN likely needs more than 10 epochs.** Loss dropped from 0.208 to
0.078 across training and was still falling at the last logged epoch —
there's no sign of a plateau. Three candidate explanations, ranked by how
much we believe each:
1. **Undertrained (most likely).** The loss curve gives direct evidence:
   it hadn't converged. LightGCN's per-epoch exposure was equalized against
   LightFM in this run, but factorization models like LightFM/SVD have a
   much shorter path to a good solution than a 3-layer graph convolution
   with randomly initialized embeddings — it plausibly needs more epochs at
   the same per-epoch exposure, not just "the same number of epochs."
2. **Hyperparameters not tuned for this graph.** `lr=0.001`, `emb_dim=64`,
   `n_layers=3` are reasonable defaults, not a result of any search on this
   dataset's user/item graph.
3. **Oversmoothing across 3 layers** — a known LightGCN failure mode on
   sparser graphs — is possible but less supported here, since a smoothed,
   converged model would show a flattening loss curve, which we don't see.

**LightFM Hybrid's accuracy gap vs. Collab is expected given this eval's
user mix, not a sign the item features are broken.** After fixing the
identity-embedding bug (see git history — the hybrid used to collapse to
near-random), genre features add a real but modest signal on an eval
sample of mostly warm/mature users, where LightFM Collab's pure
collaborative signal is already strong. The hybrid's actual selling point —
recommending items with sparse or no interaction history — isn't visible
in an aggregate NDCG number computed mostly over users who don't need it.

## Recommendation

- **Mature users (established interaction history):** LightFM
  Collaborative — best NDCG@10, and a real-time item-features path isn't
  needed since sparse content-only ranking isn't the concern for this
  segment.
- **Cold / light users:** LightFM Hybrid or Content-Based, per
  `models/recommendation_router.py`'s lifecycle routing — this is where
  their design advantage (serving items collaborative models can't) is
  supposed to pay off, which this benchmark's aggregate numbers don't
  directly measure (see [Limitations](#limitations--future-work)).
- **Cost-constrained deployments:** SVD — within 2% of LightFM Collab's
  NDCG@10 at a fraction of the training and inference cost.
- **LightGCN:** not recommended for production as currently tuned; revisit
  after a longer training run and a hyperparameter search (see below).

## Limitations & Future Work

- **10 epochs only, no hyperparameter search** for any model. Reported
  numbers are for one fixed configuration each, not each model's ceiling.
- **LightGCN's loss curve was still declining** at the final epoch — its
  true accuracy ceiling is unmeasured here; a longer run is the single
  highest-value follow-up before drawing conclusions about LightGCN's
  fitness for this problem.
- **Single train/test split, no cross-validation** — these numbers have
  unknown variance; a second seed or k-fold run would show how stable
  the ranking between models actually is.
- **300 sampled eval users, not the full user base**, for evaluation
  speed. Directionally reliable but not the last word on exact values.
- **1,653 test interactions on cold-start items were dropped** from every
  model's ground truth. This is standard practice (no model here has an
  embedding for an item it never trained on), but it also means none of
  these numbers say anything about cold-start-item performance — that's
  a separate, qualitative claim in the Production Readiness table above,
  not something measured by P/R/NDCG here.
- **Catalog coverage and cold/light/heavy user-segment breakdowns are
  implemented** (`evaluation/compare_models.py`'s `compute_coverage` /
  `compute_segment_metrics`) **but this report predates that run.**
  <!-- TODO: replace this bullet with real coverage/segment numbers once
  the next full run (commit 28632c8 or later) completes. -->
- **No production load-testing.** Reported inference latency is
  single-request Python-side timing in a benchmark script, not a served
  API under concurrent load.

## Reproducing

```powershell
python evaluation/compare_models.py
```

See [models/README.md](models/README.md) for the full shared evaluation
protocol and [results/metrics.csv](results/metrics.csv) for raw per-run
numbers, including run metadata (date, commit, dataset scale).
