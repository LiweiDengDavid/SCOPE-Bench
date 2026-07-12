"""Shared top-k evaluation kernels used by centralized and sequential evaluators."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def compute_item_pop_count(eval_data) -> "np.ndarray | None":
    """Training-interaction counts per item, indexed by internal item id.

    Prefers ``eval_data.train_dataset`` (the pure train split stashed by
    ``EvalDataLoader``) over ``additional_dataset``: the latter is the
    history-mask base and may be train+valid for the test loader under
    ``test_history_mask='train_valid'``, which would give Novelty/item-bucket
    metrics a different popularity base at valid vs test time.

    Returns ``None`` when no training split is attached. The shared loop and
    ``0 <= item_id < n_items`` guard are the single source of truth for both the
    centralized evaluator (novelty) and the trainer's score-bucket diagnostics;
    each caller adapts only the dtype/sentinel at its own boundary.
    """
    train_ds = getattr(eval_data, "train_dataset", None)
    if train_ds is None:
        train_ds = getattr(eval_data, "additional_dataset", None)
    if train_ds is None:
        return None
    df = train_ds.df if hasattr(train_ds, "df") else train_ds
    iid_field = eval_data.dataset.iid_field
    n_items = eval_data.dataset.item_num
    counts_arr = np.zeros(n_items, dtype=np.float64)
    for item_id, count in df[iid_field].value_counts().items():
        if 0 <= item_id < n_items:
            counts_arr[item_id] = count
    return counts_arr


def recall_(pos_index, pos_len):
    safe_len = pos_len.reshape(-1, 1)
    rec_ret = np.where(
        safe_len > 0,
        np.cumsum(pos_index, axis=1) / np.where(safe_len > 0, safe_len, 1),
        0.0,
    )
    return rec_ret.mean(axis=0)


def ndcg_(pos_index, pos_len):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float64)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        if idx == 0:
            idcg[row, :] = 0.0
        else:
            idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float64)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = dcg * pos_index
    dcg = np.cumsum(dcg, axis=1)

    result = np.where(idcg > 0, dcg / idcg, 0.0)
    return result.mean(axis=0)


def precision_(pos_index, pos_len):
    del pos_len
    return pos_index.cumsum(axis=1).mean(axis=0) / np.arange(1, pos_index.shape[1] + 1)


def map_(pos_index, pos_len):
    pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    sum_pre = np.cumsum(pre * pos_index, axis=1)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
    result = np.zeros_like(pos_index, dtype=np.float64)
    for row, lens in enumerate(actual_len):
        if lens == 0:
            result[row] = 0.0
            continue
        ranges = np.arange(1, pos_index.shape[1] + 1)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result.mean(axis=0)


def mrr_(pos_index, pos_len):
    del pos_len
    idxs = pos_index.argmax(axis=1)
    result = np.zeros_like(pos_index, dtype=np.float64)
    for row, idx in enumerate(idxs):
        if pos_index[row, idx] > 0:
            result[row, idx:] = 1 / (idx + 1)
        else:
            result[row, :] = 0
    return result.mean(axis=0)


def hit_(pos_index, pos_len):
    del pos_len
    return (np.cumsum(pos_index, axis=1) > 0).mean(axis=0)


def diversity_(topk_index):
    """Recommendation diversity via complement of Gini coefficient.

    Measures how evenly items are distributed across users' recommendation
    lists. 1.0 = perfectly uniform, 0.0 = all users get the same single item.

    Args:
        topk_index: (n_users, max_k) array of recommended item IDs.

    Returns:
        (max_k,) array of diversity scores at each cutoff k=1..max_k.
    """
    max_k = topk_index.shape[1]
    scores = np.empty(max_k, dtype=np.float64)

    for k in range(1, max_k + 1):
        items = topk_index[:, :k].ravel()
        _, counts = np.unique(items, return_counts=True)
        n = len(counts)
        if n <= 1:
            scores[k - 1] = 0.0
            continue
        sorted_counts = np.sort(counts).astype(np.float64)
        total = sorted_counts.sum()
        indices = np.arange(1, n + 1)
        gini = (2.0 * np.dot(indices, sorted_counts)) / (n * total) - (n + 1.0) / n
        scores[k - 1] = 1.0 - gini

    return scores


def novelty_(topk_index, item_pop_freq):
    """Self-information novelty: mean(-log2(popularity)) over recommended items.

    Args:
        topk_index: (n_users, max_k) array of recommended item IDs.
        item_pop_freq: (n_items,) array where item_pop_freq[i] = fraction of
            training interactions involving item i.

    Returns:
        (max_k,) array of novelty scores at each cutoff k=1..max_k.
    """
    pop = item_pop_freq[topk_index]
    pop = np.clip(pop, 1e-10, None)
    neg_log_pop = -np.log2(pop)
    cumsum = np.cumsum(neg_log_pop, axis=1)
    max_k = topk_index.shape[1]
    ks = np.arange(1, max_k + 1)
    per_user = cumsum / ks
    return per_user.mean(axis=0)


def coverage_(topk_index, n_items):
    """Catalog coverage: fraction of total items appearing in recommendations.

    Args:
        topk_index: (n_users, max_k) array of recommended item IDs.
        n_items: Total number of items in the dataset.

    Returns:
        (max_k,) array of coverage scores at each cutoff k=1..max_k.
    """
    max_k = topk_index.shape[1]
    scores = np.empty(max_k, dtype=np.float64)
    seen = set()
    for k in range(max_k):
        seen.update(topk_index[:, k].tolist())
        scores[k] = len(seen) / n_items
    return scores


metrics_dict = {
    "ndcg": ndcg_,
    "recall": recall_,
    "precision": precision_,
    "map": map_,
    "mrr": mrr_,
    "hit": hit_,
}


topk_metrics = {
    metric.lower(): metric
    for metric in [
        "Recall",
        "Precision",
        "NDCG",
        "MAP",
        "MRR",
        "Hit",
        "Diversity",
        "Novelty",
        "Coverage",
    ]
}


def validate_topk_args(metrics: Sequence[str], topk: Sequence[int]) -> None:
    if not isinstance(metrics, list):
        raise TypeError(f"The metrics must be a list, but get {type(metrics)}.")
    for metric in metrics:
        if metric.lower() not in topk_metrics:
            raise ValueError(f"There is no metric named '{metric}'.")

    if not isinstance(topk, list):
        raise TypeError(f"The topk must be a list, but get {type(topk)}.")
    for k in topk:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"The topk must be a positive integer, but get {k}.")


def build_bool_rec_matrix(
    topk_index: np.ndarray,
    positive_items: Iterable[Iterable[int]],
) -> np.ndarray:
    bool_rec_matrix = []
    for positives, recommended in zip(positive_items, topk_index):
        positive_set = set(positives)
        bool_rec_matrix.append([item in positive_set for item in recommended])
    return np.asarray(bool_rec_matrix, dtype=bool)


def compute_metric_arrays(
    metrics: Sequence[str],
    bool_rec_matrix: np.ndarray,
    pos_len_array: np.ndarray,
    topk_index: np.ndarray,
    n_items: int | None = None,
    item_pop_freq: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    metric_arrays: Dict[str, np.ndarray] = {}
    for metric in metrics:
        key = metric.lower()
        if key in metrics_dict:
            metric_arrays[metric] = metrics_dict[key](bool_rec_matrix, pos_len_array)
        elif key == "diversity":
            metric_arrays[metric] = diversity_(topk_index)
        elif key == "novelty":
            if item_pop_freq is None:
                raise ValueError(
                    "Novelty metric requires item popularity frequencies. "
                    "Ensure the evaluator provides item_pop_freq."
                )
            metric_arrays[metric] = novelty_(topk_index, item_pop_freq)
        elif key == "coverage":
            if n_items is None:
                raise ValueError(
                    "Coverage metric requires total item count. "
                    "Ensure the evaluator provides n_items."
                )
            metric_arrays[metric] = coverage_(topk_index, n_items)
    return metric_arrays


def build_topk_result_dict(
    metric_arrays: Dict[str, np.ndarray],
    topk: Sequence[int],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for metric, metric_result in metric_arrays.items():
        for k in topk:
            result[f"{metric}@{k}"] = round(float(metric_result[k - 1]), 6)

    return result
