# coding: utf-8
 
"""Unified Top-K evaluator built on the shared evaluation kernel."""

import os
import logging

import numpy as np
import pandas as pd
import torch

from .topk_kernel import (
    build_bool_rec_matrix,
    build_topk_result_dict,
    compute_item_pop_count,
    compute_metric_arrays,
    validate_topk_args,
)
from .lcds import build_lcds_result_dict, configured_cds_gain_table
from ..utils.training import get_local_time, artifact_token


class TopKEvaluator(object):
    """Top-K evaluator for ranking tasks."""

    def __init__(self, config):
        self.config = config
        self.metrics = config["metrics"]
        self.topk = config["topk"]
        self.save_recom_result = config["save_recommended_topk"]
        self.cds_gain_table = configured_cds_gain_table(config)
        if self.cds_gain_table is not None:
            logging.getLogger("nexusrec").info(
                "LCDS enabled: items=%s numeric=%s null=%s missing=%s",
                self.cds_gain_table.stats["item_count"],
                self.cds_gain_table.stats["numeric_score_count"],
                self.cds_gain_table.stats["null_score_count"],
                self.cds_gain_table.stats["missing_score_count"],
            )
        self._check_args()

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        del is_test
        if not batch_matrix_list:
            logger = logging.getLogger("nexusrec")
            logger.warning("Empty batch matrix list in evaluation")
            empty_result = {}
            for metric in self.metrics:
                for k in self.topk:
                    empty_result[f"{metric}@{k}"] = 0.0
            if self.cds_gain_table is not None:
                for metric in ("A-LCDS", "E-LCDS"):
                    for k in self.topk:
                        empty_result[f"{metric}@{k}"] = 0.0
            return empty_result

        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        if self.save_recom_result:
            model_name = self.config["model"]
            dataset_name = self.config["dataset"]
            max_k = max(self.topk)
            artifact_tag = self._artifact_tag(idx, max_k)
            file_path = os.path.join(
                self.config["checkpoint_dir"],
                f"[{model_name}]-[{dataset_name}]-[{artifact_tag}]-"
                f"[{get_local_time()}].topk.tsv",
            )
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, "id", eval_data.get_eval_users())
            x_df.columns = ["id"] + [f"top_{i}" for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep="\t", index=False)

        assert len(pos_len_list) == len(topk_index)
        bool_rec_matrix = build_bool_rec_matrix(topk_index, pos_items)
        pos_len_array = np.array(pos_len_list)
        n_items = eval_data.dataset.item_num
        if max(self.topk) > n_items:
            raise ValueError(
                f"max(topk)={max(self.topk)} exceeds the number of rankable items "
                f"(n_items={n_items}); requested cutoffs cannot be scored. "
                f"Lower topk or use a larger catalog."
            )
        item_pop_count = self._compute_item_pop_count(eval_data)
        item_pop_freq = self._normalize_item_pop_count(item_pop_count)
        metric_arrays = compute_metric_arrays(
            self.metrics,
            bool_rec_matrix,
            pos_len_array,
            topk_index,
            n_items=n_items,
            item_pop_freq=item_pop_freq,
        )
        result = build_topk_result_dict(metric_arrays, self.topk)
        if self.cds_gain_table is not None:
            result.update(
                build_lcds_result_dict(
                    topk_index,
                    self.cds_gain_table.gains,
                    self.topk,
                )
            )
        if self.config["item_bucket_metrics"]:
            if item_pop_count is None:
                raise ValueError(
                    "item_bucket_metrics requires training interactions through "
                    "eval_data.additional_dataset."
                )
            result.update(
                self._compute_item_bucket_metrics(
                    pos_items,
                    bool_rec_matrix,
                    pos_len_array,
                    topk_index,
                    item_pop_count,
                    n_items,
                    item_pop_freq,
                )
            )
        return result

    def _compute_item_pop_count(self, eval_data):
        """Compute item popularity counts from training interactions."""
        return compute_item_pop_count(eval_data)

    def _normalize_item_pop_count(self, item_pop_count):
        """Normalize item counts for novelty metrics."""
        if item_pop_count is None:
            return None
        total = item_pop_count.sum()
        if total <= 0:
            return np.zeros_like(item_pop_count, dtype=np.float64)
        return item_pop_count / total

    def _compute_item_bucket_metrics(
        self,
        pos_items,
        bool_rec_matrix,
        pos_len_array,
        topk_index,
        item_pop_count,
        n_items,
        item_pop_freq,
    ):
        """Compute top-k metrics on head/mid/tail positive-item buckets."""
        observed_counts = item_pop_count[item_pop_count > 0]
        if observed_counts.size == 0:
            raise ValueError("item_bucket_metrics requires non-empty training item counts.")

        tail_cut = np.quantile(
            observed_counts, float(self.config["item_bucket_tail_quantile"])
        )
        head_cut = np.quantile(
            observed_counts, float(self.config["item_bucket_head_quantile"])
        )

        # Boundaries must be strictly ordered: tail uses ``<= tail_cut`` and head
        # uses ``>= head_cut``. A collapsed quantile would put boundary items in
        # both buckets, so overlapping bucket definitions are rejected.
        if tail_cut >= head_cut:
            raise ValueError(
                f"item bucket quantiles collapsed (tail_cut={tail_cut} >= head_cut={head_cut}); "
                "head/tail buckets would double-count boundary items. Widen the gap between "
                "item_bucket_tail_quantile and item_bucket_head_quantile, or disable item buckets."
            )

        positive_count = np.array(
            [
                self._mean_positive_item_count(items, item_pop_count)
                for items in pos_items
            ],
            dtype=np.float64,
        )
        masks = {
            "tail": positive_count <= tail_cut,
            "mid": (positive_count > tail_cut) & (positive_count < head_cut),
            "head": positive_count >= head_cut,
        }

        result = {}
        for bucket_name, mask in masks.items():
            if not mask.any():
                continue
            metric_arrays = compute_metric_arrays(
                self.metrics,
                bool_rec_matrix[mask],
                pos_len_array[mask],
                topk_index[mask],
                n_items=n_items,
                item_pop_freq=item_pop_freq,
            )
            bucket_result = build_topk_result_dict(metric_arrays, self.topk)
            for key, value in bucket_result.items():
                result[f"{bucket_name}_{key}"] = value
            result[f"{bucket_name}_eval_users"] = int(mask.sum())
        return result

    def _mean_positive_item_count(self, items, item_pop_count):
        """Mean training count of positive items for one evaluation row."""
        counts = [item_pop_count[item_id] for item_id in items]
        return float(np.mean(counts))

    def _artifact_tag(self, idx, max_k):
        """Build a run-identifying tag for saved top-k artifacts."""
        raw_tag = (
            f"{self.config['type']}.{self.config['comment']}."
            f"seed{self.config['seed']}.idx{idx}.top{max_k}"
        )
        return artifact_token(raw_tag)

    def _check_args(self):
        validate_topk_args(self.metrics, self.topk)


__all__ = ["TopKEvaluator"]
