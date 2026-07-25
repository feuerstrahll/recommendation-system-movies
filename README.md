# Movie Recommendation System

Course project exploring recommendation approaches for movies using MovieLens
ratings and movie metadata.

## Documentation scope

This documentation follows the sources actually used for the recommendation
work. It does not treat the entire `feature/dev` branch as the project baseline.

| Area | Source used | Role |
|---|---|---|
| Recommendation models | [`main/models`](https://github.com/feuerstrahll/recommendation-system-movies/tree/main/models) | Primary source for the model experiments and scripts |
| Cold-start questionnaire | [`feature/dev/.../cold_start_questionnaire.py`](https://github.com/feuerstrahll/recommendation-system-movies/blob/feature/dev/recommendation-system-lab5/models/cold_start_questionnaire.py) | Supplemental onboarding component for users without rating history |

The `feature/dev` branch also contains other experimental files. Their presence
does not mean that they were used as the primary implementation, integrated
with the `main` models, or validated for deployment.

## Dataset summary

The presentation records the following project data:

| Movies | Ratings | MovieLens users | Features per movie |
|---:|---:|---:|---:|
| 45K+ | 270K+ | 671 | 20+ |

The workflow combines MovieLens interactions with metadata from The Movies
Dataset (TMDB), including movie, keyword, cast, crew, and identifier-link data.
The documented identifier contract uses the TMDB id as `movie_id` and retains
the original MovieLens id for mapping where available.

## Recommendation components

The primary model work in `main/models` includes:

- `content_lightFM.py` — a content-oriented LightFM experiment using movie
  titles, genres, and release year;
- `lightfm_model.py` — a collaborative LightFM experiment using user–movie
  interactions;
- `model_creation.ipynb` — exploratory work with SVD, nearest-neighbour methods,
  sentence embeddings, and FAISS;
- `lightfm_roc_curve.png` — an evaluation artifact from a previous run.

The cold-start questionnaire from `feature/dev` collects genre preferences,
favorite movies, and a release-period preference. It then creates initial
content-based recommendations for a new user.

## Status

| Component | Repository evidence | Documentation status |
|---|---|---|
| Content-oriented LightFM | Script in `main/models` | Source available; rerun locally before reporting results |
| Collaborative LightFM | Script in `main/models` | Source available; rerun locally before reporting results |
| SVD and FAISS exploration | Notebook in `main/models` | Experimental notebook, not presented as a production pipeline |
| Cold-start questionnaire | Module in `feature/dev` | Supplemental component; integration with the `main` models is not asserted |

This project documentation does not claim production readiness, guaranteed
performance, complete branch integration, or a deployed backend. Model quality
and runtime depend on the data snapshot, preprocessing, split, hardware,
dependency versions, and random seed.

## Recorded project metrics

The following values are preserved from the supplied presentation and describe
the project's recorded runs.

**LightFM**

- ROC AUC: **0.7277** on the documented 500-user sample;
- training time: approximately **30 seconds**;
- inference time: approximately **1 second**.

**LightGCN** — documented run with 610 evaluated users, CPU, and 20 epochs:

| Metric | K = 10 | K = 20 |
|---|---:|---:|
| Precision@K | 0.0324 | 0.0198 |
| Recall@K | 0.0652 | 0.0823 |
| NDCG@K | 0.0421 | 0.0527 |

The presentation also records these indicative stage timings:

| Stage | Recorded time |
|---|---:|
| Questionnaire | 2–5 min, interactive |
| Content-based | <0.1 s |
| Hybrid | ~0.5 s |
| Collaborative | ~1 s |

These are relevant project results, but they are not universal guarantees or
service-level objectives. Reproduce them under a documented environment before
using them for an external comparison.

## Project comparison

The presentation's qualitative assessment is retained below. “Proposed role”
describes the project's design decision, not a claim that a production service
was deployed.

| Criterion | Content-Based | LightFM (Collaborative) | Hybrid | LightGCN |
|---|---|---|---|---|
| Quality for a mature user | Medium | High | High | High |
| Cold start | Excellent | Does not address it alone | Partial | Does not address it alone |
| Training speed | Seconds | ~30 s | Minutes | 1–3 min on CPU |
| Inference speed | <0.1 s | ~1 s | ~0.5 s | 1–3 s |
| Scalability assessment | FAISS-based | Good | Medium | GPU considered |
| Real-time update assessment | Yes | Retraining | Partial | Retraining |
| Explainability | High | Medium | Medium | Low |
| Proposed role | Cold start | Primary | Warm start | Experiment |

## Repository layout relevant to this documentation

```text
recommendation-system-movies/
├── models/                                  # main branch: primary model work
│   ├── content_lightFM.py
│   ├── lightfm_model.py
│   ├── model_creation.ipynb
│   ├── lightfm_roc_curve.png
│   └── README.md
└── recommendation-system-lab5/
    └── models/
        └── cold_start_questionnaire.py      # feature/dev: supplemental source
```

The two locations are shown together to document provenance. They are not
represented as one already-integrated directory.

## Working with the code

1. Clone the repository and start from `main`.
2. Run model experiments from `main/models` with the data paths expected by the
   selected script or notebook.
3. If the cold-start questionnaire is needed, review the version in
   `feature/dev` and integrate it deliberately rather than copying the whole
   branch.
4. Record the dataset version, preprocessing steps, train/test split, random
   seed, dependency versions, and hardware when reporting results.

Before running the code, inspect each file's imports and data paths. The
repository contains experimental code, so local path or schema adjustments may
be required.

## Evaluation guidance

Treat committed charts, notebook output, and console logs as historical
artifacts from relevant project runs. For a reproducible comparison:

- use the same cleaned data and identifier mapping for every model;
- use a documented train/test split;
- exclude training interactions from evaluation candidates;
- report ranking metrics such as Precision@K, Recall@K, and NDCG@K under the
  same protocol;
- distinguish observed measurements from estimates or design targets.

## Known limitations

- The documentation does not establish a completed runtime connection between
  the `main` model scripts and the `feature/dev` questionnaire.
- No automated test suite, load test, security review, or deployment
  verification is claimed here.
- Raw LightFM scores are ranking values, not calibrated probabilities or star
  ratings.
- Branch contents can diverge; review the relevant files before merging or
  copying code.

## Presentation

[`movie-recommendation-system.html`](movie-recommendation-system.html) is a
standalone summary of the same scope and source attribution.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for a branch-aware workflow.

## Data sources

- [MovieLens](https://grouplens.org/datasets/movielens/)
- [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Review the terms and licensing of the code, datasets, and third-party
dependencies before redistribution or deployment. This documentation package
does not introduce a software license.
