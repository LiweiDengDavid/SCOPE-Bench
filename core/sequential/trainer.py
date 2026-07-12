# coding: utf-8
"""
Sequential Trainer for NexusRec
================================

Inherits TrainerBase to share optimizer setup, early-stopping logic,
and best-result tracking.  Overrides only what differs for sequential models:

- ``_create_evaluator``: returns SequentialEvaluator instead of TopKEvaluator
- ``evaluate``:          drives SequentialEvaluator (stateful accumulation,
                         filter-seen masking, single-target ranking)
- ``fit``:               sequential training loop; returns the same
                         (best_valid_score, best_valid_result, best_test_result)
                         triple as TrainerBase.fit() so that core.py needs no
                         paradigm branch.

Final test evaluation is always handled by core.py via evaluate_final_test().
"""

import math
import time
from typing import Dict

import torch
import torch.optim as optim

from ..base.trainer import TrainerBase
from ..evaluation.ranking import assert_finite_scores
from ..evaluation.export import (
    include_scores as export_includes_scores,
    is_enabled as export_is_enabled,
    write_recommendations,
)
from ..runtime.logger import TrainLogger


class SequentialTrainer(TrainerBase):
    """Trainer for sequential recommendation models."""

    def __init__(self, config, model):
        # Fail fast on artifact/metric flags the sequential paradigm does not
        # implement (precedent: the base trainer's score-components guard,
        # core/base/trainer.py); silently accepting them would let the run
        # "succeed" without ever producing the requested artifact.
        for flag in (
            "save_recommendation_scores",
            "save_recommendation_score_components",
            "item_bucket_metrics",
        ):
            if config[flag]:
                raise ValueError(
                    f"SequentialTrainer does not implement {flag}; "
                    "unset it for sequential models."
                )

        # Delegate shared setup to TrainerBase:
        # optimizer, scheduler, evaluator (via _create_evaluator override),
        # early-stopping params, best-result tracking, config extraction.
        super().__init__(config, model)

        self.logger.info(
            "Initialized SequentialTrainer | valid_metric=%s | stopping_step=%s",
            self.valid_metric,
            self.stopping_step,
        )

    # ------------------------------------------------------------------
    # Override: SequentialEvaluator instead of TopKEvaluator
    # ------------------------------------------------------------------

    def _create_evaluator(self, config):
        from .evaluator import SequentialEvaluator

        return SequentialEvaluator(config)

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0, write_export=False):
        """Evaluate via SequentialEvaluator (filter-seen, single-target ranking)."""
        del is_test  # SequentialEvaluator is driven directly; idx is artifact lineage.
        self.model.eval()
        self.evaluator.reset()
        for batch in eval_data:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            scores = self.model.full_sort_predict(batch)
            # Shared finiteness gate; the sequential evaluator owns its NumPy
            # ranking path and tie-break convention.
            assert_finite_scores(scores, self.model)
            self.evaluator.update(scores, batch)
        result = self.evaluator.get_metrics()
        if write_export and export_is_enabled(self.config):
            payload = self.evaluator.export_payload(
                export_includes_scores(self.config)
            )
            paths = write_recommendations(
                self.config,
                payload["eval_users"],
                payload["topk_indices"],
                payload["topk_scores"],
                idx=idx,
                metrics=result,
                user_count=self.model.n_users,
                item_count=payload["item_count"],
                item_id_offset=1,
            )
            self.last_recommendation_export_paths = paths
            self.logger.info("Recommendation export saved: %s", paths)
        return result

    # ------------------------------------------------------------------
    # Override: sequential training loop with standard return type
    # ------------------------------------------------------------------

    def fit(self, train_data, valid_data=None, test_data=None):
        """Train sequential model.

        Returns the same ``(best_valid_score, best_valid_result, best_test_result)``
        triple as ``TrainerBase.fit()``, enabling a single code path in core.py.
        """
        self.training_start_time = time.time()
        # Wire training-set item popularity into the evaluator so the Novelty
        # metric works on the sequential path (parity with centralized/federated,
        # which derive it from train interactions). Length item_num+1, indexed by
        # global id; the evaluator drops the PAD column at index 0. Persists across
        # the final-test evaluation (same evaluator instance).
        self.evaluator.item_pop_freq = train_data.dataset.compute_item_pop_freq()
        for epoch in range(self.start_epoch, self.max_epochs):
            self.model.pre_epoch_processing()
            self.train_epoch(train_data, epoch)

            # Respect eval_enabled / skip_eval_during_training (inherited from
            # TrainerBase) instead of a raw modulo, matching centralized/federated.
            if valid_data is not None and self._should_eval(epoch + 1, self.max_epochs):
                self._eval_and_update(valid_data, test_data, epoch)

                if self._patience_exhausted():
                    self.logger.info("Early stopping at epoch %s", epoch + 1)
                    break

            if self.lr_scheduler is not None and not isinstance(
                self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step()

            self.model.post_epoch_processing()
            # Resume checkpoint at the epoch boundary (captures model/optimizer/
            # scheduler/RNG/best-tracking + the DataLoader generator state).
            self._maybe_save_resume(epoch + 1, train_data)

        self._log_training_summary(self._get_best_epoch())
        return self.best_valid_score, self.best_valid_result, self.best_test_result

    def train_epoch(self, train_dataloader, epoch_idx: int = 0) -> Dict[str, float]:
        """Train for one epoch with dict-batches from SequentialDataLoader."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        nan_count = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model.calculate_loss(batch)
            loss_val = loss.item()

            if math.isnan(loss_val) or math.isinf(loss_val):
                nan_count += 1
                if nan_count >= self.config["nan_abort_threshold"]:
                    raise ValueError(
                        f"Training diverged: {nan_count} consecutive NaN/Inf losses"
                    )
                self.logger.warning("NaN/Inf loss at batch %s, skipping", batch_idx)
                continue

            nan_count = 0
            loss.backward()
            if self.clip_grad_norm and self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )
            self.optimizer.step()
            total_loss += loss_val
            num_batches += 1

            log_interval = self.config["batch_log_interval"]
            if log_interval > 0 and batch_idx % log_interval == 0:
                self.logger.debug(
                    "Batch %s/%s | Loss: %.4f",
                    batch_idx,
                    len(train_dataloader),
                    loss_val,
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - start_time

        # Per-epoch INFO log, matching centralized core.utils.training.train_epoch.
        TrainLogger.log_epoch_progress(
            epoch=epoch_idx + 1,
            total_epochs=self.max_epochs,
            stage="Train",
            loss=avg_loss,
            epoch_time=epoch_time,
            num_batches=num_batches,
        )

        return {
            "loss": avg_loss,
            "time": epoch_time,
            "num_batches": num_batches,
        }

    # Checkpointing is fully delegated to TrainerBase: best_model.pth is written
    # once post-training on the validation-best state, and training resume uses
    # save_training_state/load_training_state for full-state checkpoints.
