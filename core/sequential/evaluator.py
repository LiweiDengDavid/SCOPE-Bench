# coding: utf-8
"""Sequential evaluator built on the shared top-k evaluation kernel."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch

from ..evaluation.topk_kernel import (
    build_topk_result_dict,
    compute_metric_arrays,
)
from ..evaluation.lcds import build_lcds_result_dict, configured_cds_gain_table


logger = logging.getLogger("nexusrec")


class SequentialEvaluator:
    """Evaluator for sequential recommendation models."""

    def __init__(self, config):
        self.config = config
        self.k_list = config["topk"]
        self.metrics = config["metrics"]
        self.filter_seen = config["filter_seen"]
        self.cds_gain_table = configured_cds_gain_table(config)
        if self.cds_gain_table is not None:
            logger.info(
                "LCDS enabled: items=%s numeric=%s null=%s missing=%s",
                self.cds_gain_table.stats["item_count"],
                self.cds_gain_table.stats["numeric_score_count"],
                self.cds_gain_table.stats["null_score_count"],
                self.cds_gain_table.stats["missing_score_count"],
            )
        # Training-set item popularity for the Novelty metric. SequentialTrainer
        # fills it from the train split once data is available; eval-only callers
        # must provide it before requesting Novelty.
        self.item_pop_freq = None
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.user_ids = []

    def update(self, scores: torch.Tensor, batch: Dict[str, torch.Tensor]):
        """Update with a batch of predictions.

        Supports both single-target (standard) and multi-target (temporal_ratio /
        hybrid split) evaluation.  When ``targets_list`` and ``num_targets`` are
        present in *batch*, each target is evaluated independently against the
        same score vector (batch mode).  The primary ``targets`` key is used
        for the standard single-target path.
        """
        batch_size = scores.size(0)
        num_items = scores.size(1)

        scores_np = scores.detach().cpu().numpy()
        targets_np = batch["targets"].detach().cpu().numpy()
        user_ids_np = batch["user_ids"].detach().cpu().numpy()

        has_multi = "targets_list" in batch and "num_targets" in batch
        if has_multi:
            targets_list_np = batch["targets_list"].detach().cpu().numpy()
            num_targets_np = batch["num_targets"].detach().cpu().numpy()

        for i in range(batch_size):
            user_id = user_ids_np[i]
            user_scores = scores_np[i].copy()

            # Suppress padding token (item_id=0)
            if num_items > 0:
                user_scores[0] = float("-inf")

            if self.filter_seen and "item_seqs" in batch:
                # Filter the FULL user history, not just the (max_seq_len-)truncated
                # model window: for long-history users, items older than the window
                # would otherwise reappear in top-k and be counted by metrics.
                if "full_item_seqs" in batch:
                    hist = batch["full_item_seqs"][i].detach().cpu().numpy()
                    hist_len = batch["full_seq_lens"][i].item()
                else:
                    hist = batch["item_seqs"][i].detach().cpu().numpy()
                    hist_len = batch["seq_lens"][i].item()
                interacted = set(hist[:hist_len].tolist())
                # A re-purchased item can appear in the history AND as a current
                # target; masking it would make that target structurally
                # unreachable. Keep the current target(s) rankable — multi-target
                # rows still suppress EARLIER targets explicitly below.
                if has_multi and num_targets_np[i] > 1:
                    interacted -= {
                        int(t) for t in targets_list_np[i, : int(num_targets_np[i])]
                    }
                else:
                    interacted.discard(int(targets_np[i]))
                for item_id in interacted:
                    if 0 < item_id < num_items:
                        user_scores[item_id] = float("-inf")

            if has_multi and num_targets_np[i] > 1:
                # Multi-target: evaluate each target independently.
                # Suppress earlier targets from later evaluations to avoid
                # inflating metrics in temporal evaluation.
                nt = int(num_targets_np[i])
                scores_for_user = user_scores.copy()
                for t_idx in range(nt):
                    t = int(targets_list_np[i, t_idx])
                    if t > 0:
                        self.predictions.append(scores_for_user.copy())
                        self.targets.append(t)
                        self.user_ids.append(user_id)
                        # Suppress this target for subsequent evaluations
                        if 0 < t < num_items:
                            scores_for_user[t] = float("-inf")
            else:
                # Standard single-target evaluation.
                target = targets_np[i]
                self.predictions.append(user_scores)
                self.targets.append(target)
                self.user_ids.append(user_id)


    def _build_metric_inputs(self):
        if not self.predictions:
            return None, None, None, None, None

        score_matrix = np.asarray(self.predictions)  # [n_users, n_items+1]; col 0 = PAD
        target_array = np.asarray(self.targets)
        topk_max = max(self.k_list)
        # Rank over VALID items only by dropping the PAD column (index 0). Relying
        # on PAD's -inf score is not enough: the deterministic low-id tie-break
        # (shared with the centralized/federated paths via torch.argsort
        # descending+stable) would otherwise surface PAD as a filler whenever a
        # user has fewer finite-scored items than k. Excluding the column also
        # removes the phantom-slot inflation of the coverage denominator.
        # argsort(-scores, stable) gives descending order with ties broken by lower
        # (local) id; +1 maps local indices back to global item ids (1..n_items).
        valid_scores = score_matrix[:, 1:]
        topk_indices = np.argsort(-valid_scores, axis=1, kind="stable")[:, :topk_max] + 1
        hit_matrix = topk_indices == target_array.reshape(-1, 1)
        target_lengths = np.ones(len(target_array), dtype=np.int64)
        return score_matrix, target_array, topk_indices, hit_matrix, target_lengths

    def get_metrics(self) -> Dict[str, float]:
        inputs = self._build_metric_inputs()
        if inputs[0] is None:
            logger.warning("No predictions available for evaluation")
            return {}

        score_matrix, _target_array, topk_indices, hit_matrix, target_lengths = inputs
        # Valid item catalog excludes the PAD column (index 0), so the coverage
        # denominator is the real item count, not the padded width.
        n_items = score_matrix.shape[1] - 1
        if max(self.k_list) > n_items:
            raise ValueError(
                f"max(topk)={max(self.k_list)} exceeds the number of rankable items "
                f"(n_items={n_items}); requested cutoffs cannot be scored. "
                f"Lower topk or use a larger catalog."
            )
        metric_arrays = compute_metric_arrays(
            self.metrics,
            hit_matrix,
            target_lengths,
            topk_indices,
            n_items=n_items,
            item_pop_freq=self.item_pop_freq,
        )
        results = build_topk_result_dict(metric_arrays, self.k_list)
        if self.cds_gain_table is not None:
            results.update(
                build_lcds_result_dict(
                    topk_indices,
                    self.cds_gain_table.gains,
                    self.k_list,
                    item_id_offset=1,
                )
            )

        if "MRR" in self.metrics:
            results["MRR"] = results[f"MRR@{max(self.k_list)}"]

        return results

    def export_payload(self, include_scores: bool):
        """Return masked top-k payload using sequential shifted item ids.

        ``topk_indices`` uses the model/evaluator item id space where 0 is PAD and
        real items are 1..item_num. The exporter receives item_id_offset=1 and
        maps these ids back to the shared zero-based NexusRec internal item index.
        """
        inputs = self._build_metric_inputs()
        score_matrix, _target_array, topk_indices, _hit_matrix, _target_lengths = inputs
        if score_matrix is None:
            raise ValueError("No sequential predictions available for recommendation export.")
        topk_scores = None
        if include_scores:
            topk_scores = np.take_along_axis(score_matrix, topk_indices, axis=1)
        return {
            "eval_users": np.asarray(self.user_ids, dtype=np.int64),
            "topk_indices": topk_indices,
            "topk_scores": topk_scores,
            "item_count": int(score_matrix.shape[1] - 1),
        }
