from __future__ import annotations

from collections.abc import Iterable


def precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    recommended_k = list(recommended)[:k]
    if not recommended_k:
        return 0.0

    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(recommended_k)


def recall_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant:
        return 0.0

    recommended_k = list(recommended)[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    import math
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant:
        return 0.0

    recommended_k = list(recommended)[:k]
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(recommended_k)
        if item in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    """
    Average Precision at K: precision computed at each rank where a hit
    occurs, averaged over the number of relevant items (capped at k). Same
    per-user, within-candidate-set framing as precision/recall/NDCG above
    (unlike ROC-AUC, which compares scores across all items/users at once).
    Mean of this across users gives MAP@K.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant:
        return 0.0

    recommended_k = list(recommended)[:k]
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(recommended_k, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / i

    return precision_sum / min(len(relevant), k)

