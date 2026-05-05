import pandas as pd


def combine_scores(
    content_scores: pd.Series,
    collaborative_scores: pd.Series,
    content_weight: float = 0.4,
) -> pd.Series:
    if not 0 <= content_weight <= 1:
        raise ValueError("content_weight must be between 0 and 1.")

    collaborative_weight = 1 - content_weight
    all_items = content_scores.index.union(collaborative_scores.index)
    content = content_scores.reindex(all_items, fill_value=0)
    collaborative = collaborative_scores.reindex(all_items, fill_value=0)
    hybrid = content_weight * content + collaborative_weight * collaborative
    return hybrid.sort_values(ascending=False)

