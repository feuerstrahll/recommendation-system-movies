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

