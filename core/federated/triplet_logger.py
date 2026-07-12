"""Optional federated training-triplet CSV diagnostics.

Kept outside ``FederatedTrainer`` so the trainer stays focused on the round
loop. Active only when ``config['save_training_triplets']`` is set; a single
call site in the client training loop.
"""

from __future__ import annotations

import csv
import os

import torch
import torch.nn.functional as F


class TripletLogger:
    """Append per-batch (user, pos, neg) training triplets — optionally with
    pairwise score components — to a CSV artifact, capped at a max row count."""

    def __init__(self, config, artifact_token):
        self.config = config
        # callable: raw_tag -> filesystem-safe token (core.utils.training.artifact_token)
        self._artifact_token = artifact_token
        self.rows_written = 0

    def record(self, round_idx, client_user, local_epoch, batch_idx, batch, client_model):
        if not torch.is_tensor(batch):
            return
        if batch.dim() < 2 or batch.shape[0] < 3:
            return

        max_rows = int(self.config["training_triplet_log_max_rows"])
        remaining = max_rows - self.rows_written
        if remaining <= 0:
            return

        users = batch[0].detach().cpu().tolist()
        positives = batch[1].detach().cpu().tolist()
        negatives = batch[2].detach().cpu().tolist()
        take = min(remaining, len(users))
        score_rows = []
        if self.config["save_training_triplet_scores"]:
            score_rows = self._score_rows(client_model, batch, take)

        file_path = self._artifact_path()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        mode = "w" if self.rows_written == 0 else "a"
        write_header = self.rows_written == 0
        fieldnames = [
            "round",
            "client_user",
            "local_epoch",
            "batch_idx",
            "user",
            "positive_item",
            "negative_item",
        ]
        if self.config["save_training_triplet_scores"]:
            fieldnames.extend(
                [
                    "final_pos_score",
                    "final_neg_score",
                    "anchor_pos_score",
                    "anchor_neg_score",
                    "positive_delta",
                    "negative_delta",
                    "final_score_gap",
                    "anchor_score_gap",
                    "residual_flip",
                    "bpr_hardness",
                ]
            )
        with open(file_path, mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for offset in range(take):
                row = {
                    "round": int(round_idx) + 1,
                    "client_user": int(client_user),
                    "local_epoch": int(local_epoch) + 1,
                    "batch_idx": int(batch_idx),
                    "user": int(users[offset]),
                    "positive_item": int(positives[offset]),
                    "negative_item": int(negatives[offset]),
                }
                if self.config["save_training_triplet_scores"]:
                    row.update(score_rows[offset])
                writer.writerow(row)
        self.rows_written += take

    def _score_rows(self, client_model, batch, take):
        users = batch[0, :take]
        positives = batch[1, :take]
        negatives = batch[2, :take]
        paired_users = torch.cat([users, users])
        paired_items = torch.cat([positives, negatives])

        with torch.no_grad():
            component_predictor = getattr(
                client_model, "pairwise_score_components", None
            )
            # Triplet scoring is meaningful only for anchor/semantic models that
            # expose the three components explicitly. Single-score models cannot
            # produce (final, anchor, semantic); require the explicit hook and fail
            # loudly instead of calling an incompatible forward().
            if not callable(component_predictor):
                raise ValueError(
                    f"save_training_triplet_scores requires the model "
                    f"('{type(client_model).__name__}') to implement "
                    "pairwise_score_components(users, items) -> "
                    "(final_score, anchor_score, semantic_delta)."
                )
            outputs = component_predictor(paired_users, paired_items)

        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            raise ValueError(
                "pairwise_score_components() must return "
                "(final_score, anchor_score, semantic_delta)."
            )

        final_scores, anchor_scores, semantic_delta = outputs[:3]
        final_pos = final_scores[:take].detach().cpu()
        final_neg = final_scores[take:].detach().cpu()
        anchor_pos = anchor_scores[:take].detach().cpu()
        anchor_neg = anchor_scores[take:].detach().cpu()
        positive_delta = semantic_delta[:take].detach().cpu()
        negative_delta = semantic_delta[take:].detach().cpu()

        final_gap = final_pos - final_neg
        anchor_gap = anchor_pos - anchor_neg
        residual_flip = negative_delta - positive_delta
        bpr_hardness = F.softplus(final_neg - final_pos)

        rows = []
        for offset in range(take):
            rows.append(
                {
                    "final_pos_score": float(final_pos[offset]),
                    "final_neg_score": float(final_neg[offset]),
                    "anchor_pos_score": float(anchor_pos[offset]),
                    "anchor_neg_score": float(anchor_neg[offset]),
                    "positive_delta": float(positive_delta[offset]),
                    "negative_delta": float(negative_delta[offset]),
                    "final_score_gap": float(final_gap[offset]),
                    "anchor_score_gap": float(anchor_gap[offset]),
                    "residual_flip": float(residual_flip[offset]),
                    "bpr_hardness": float(bpr_hardness[offset]),
                }
            )
        return rows

    def _artifact_path(self):
        raw_tag = (
            f"{self.config['type']}.{self.config['comment']}."
            f"seed{self.config['seed']}.training_triplets"
        )
        artifact_tag = self._artifact_token(raw_tag)
        file_name = (
            f"[{self.config['model']}]-[{self.config['dataset']}]-"
            f"[{artifact_tag}].csv"
        )
        return os.path.join(self.config["checkpoint_dir"], file_name)
