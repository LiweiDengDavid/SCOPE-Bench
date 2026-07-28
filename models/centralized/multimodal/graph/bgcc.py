# coding: utf-8
r"""
BGCC -- Behavior-Guided Candidate Calibration
################################################
Reference:
    Paper: "Behavior-Guided Candidate Calibration for Multimodal Recommendation",
           AAAI'2026 (arXiv 2605.22073). Zesheng Li, Chengchang Pan, Honggang Qi.
    Official repo: https://github.com/LIZESHENG13/bridge (model class ``BRIDGE``).

BGCC converts *training-only* co-user item co-occurrence into signed candidate
evidence and uses it to calibrate the item scores of a multimodal backbone. A
spectral analysis (SVD split of the backbone item embeddings) separates a
low-frequency shared-structure view from higher-frequency discriminative views;
moderate cross-view agreement helps ranking while strong agreement suppresses
recommendation-specific variation. The backbone keeps the representation space
stable; behavior evidence acts only where ranking is decided.

This is a STANDALONE model wrapping a configurable multimodal backbone (mirrors
DA-MRS/BeFA). ``config["backbone"]`` selects the encoder BGCC calibrates. The
official repo wraps its own ``DualFrequencyEncoder`` multimodal graph encoder;
the standalone wrapper here uses a compact, self-contained multimodal graph
backbone (``"mgcn"``) that produces user/item embeddings from id + projected
visual/text features (mirroring MGCN's construction). The behavior-guided
candidate calibration itself is ported faithfully from ``src/models/bridge.py``:

  * ``_build_behavior_score_matrix``  -> co-user item-item similarity (cosine/
    jaccard/logcooc), top-k sparsified, aggregated over the user's training
    history and per-user normalised (z / minmax) into signed evidence.
  * ``_frequency_views``              -> SVD low/high spectral split of item and
    user embeddings, giving an optional high-frequency residual signal.
  * ``_pair_gate`` / ``_combine_score_components`` -> a learned per-(user,item)
    gate meters ``behavior_weight`` (and ``residual_weight``); the calibration is
    applied only inside the base top-K candidate set in ``full_sort_predict``.

Setting ``behavior_weight`` (and ``residual_weight``) to 0 removes the
calibration and recovers the raw backbone scores.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


class _MGCNBackbone(nn.Module):
    """Compact self-contained multimodal graph backbone.

    Produces user/item embeddings from id embeddings + U-I graph propagation,
    fused with projected visual/text item features (mirrors MGCN's behavior
    purifier gate). Kept internal so BGCC calibrates a stable representation
    space without depending on the heavyweight reference encoder.
    """

    def __init__(self, n_users, n_items, embedding_dim, n_layers, norm_adj,
                 image_feats_dim, text_feats_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.norm_adj = norm_adj

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_trs = (
            nn.Linear(image_feats_dim, embedding_dim) if image_feats_dim else None
        )
        self.text_trs = (
            nn.Linear(text_feats_dim, embedding_dim) if text_feats_dim else None
        )
        self.gate_v = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())

    def forward(self, image_embedding=None, text_embedding=None):
        item_emb = self.item_id_embedding.weight
        # Behavior-guided purifier gate: modulate id embedding by projected
        # modality features (mirror MGCN's gate_v / gate_t).
        if self.image_trs is not None and image_embedding is not None:
            image_feats = self.image_trs(image_embedding.weight)
            item_emb = item_emb + item_emb * self.gate_v(image_feats)
        if self.text_trs is not None and text_embedding is not None:
            text_feats = self.text_trs(text_embedding.weight)
            item_emb = item_emb + item_emb * self.gate_t(text_feats)

        ego = torch.cat((self.user_embedding.weight, item_emb), dim=0)
        all_embeddings = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embeddings += [ego]
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        u_g, i_g = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g, i_g


class BGCC(RecommenderBase):
    def __init__(self, config, dataloader):
        super(BGCC, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["num_layers"]
        self.reg_weight = float(config["reg_weight"])

        # --- behavior-guided candidate calibration knobs (ref bridge.py) ---
        self.behavior_weight = float(config["behavior_weight"])
        self.behavior_topk = int(config["behavior_topk"])
        # Size of the base top-K shortlist C_u that calibration is restricted
        # to (ref bridge.py behavior_eval_topk / the paper's indicator 1[i in C_u]).
        self.behavior_eval_topk = int(config["behavior_eval_topk"])
        self.behavior_sim_method = str(config["behavior_sim_method"]).lower()
        self.behavior_aggregation = str(config["behavior_aggregation"]).lower()
        self.behavior_score_norm = str(config["behavior_score_norm"]).lower()

        # --- spectral high-frequency residual (ref _frequency_views) ---
        self.num_freq_bands = int(config["num_freq_bands"])
        self.low_band_count = int(config["low_band_count"])
        self.residual_weight = float(config["residual_weight"])

        # --- learned calibration gate (ref _pair_gate) ---
        self.gate_reg_weight = float(config["gate_reg_weight"])
        self.gate_target = float(config["gate_target"])

        # --- backbone selection ---
        backbone = config["backbone"]
        if backbone != "mgcn":
            raise ValueError(
                f"BGCC supports backbone='mgcn' only, got '{backbone}'"
            )

        # U-I adjacency (symmetric normalised) from the interaction matrix.
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)
        norm_adj = self._get_norm_adj_mat().to(self.device)

        # --- modality feature transforms live on the backbone (mirror MGCN) ---
        v_dim = self.v_feat.shape[1] if self.v_feat is not None else 0
        t_dim = self.t_feat.shape[1] if self.t_feat is not None else 0
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        self.backbone = _MGCNBackbone(
            self.n_users, self.n_items, self.embedding_dim, self.n_layers,
            norm_adj, v_dim, t_dim,
        )

        # --- behavior score matrix (ref _build_behavior_score_matrix) ---
        behavior_scores = self._build_behavior_score_matrix(dataloader)
        self.behavior_score_matrix = torch.from_numpy(behavior_scores).to(self.device)

        # --- learned gate parameters (ref bridge.py behavior_*_bias / scales) ---
        self.behavior_user_bias = nn.Embedding(self.n_users, 1)
        self.behavior_item_bias = nn.Embedding(self.n_items, 1)
        nn.init.zeros_(self.behavior_user_bias.weight)
        nn.init.zeros_(self.behavior_item_bias.weight)
        self.gate_base_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.gate_behavior_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    # ------------------------------------------------------------------ graphs
    def _get_norm_adj_mat(self):
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        ).tolil()
        R = self.interaction_matrix.tolil()
        A[: self.n_users, self.n_users :] = R
        A[self.n_users :, : self.n_users] = R.T
        A = A.todok()
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = sp.coo_matrix(D * A * D)
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(
            i, data, torch.Size((self.n_users + self.n_items,) * 2)
        )

    # -------------------------------------------------- behavior evidence (BGCC)
    def _build_behavior_score_matrix(self, dataloader):
        """Co-user item-item behavior evidence (ref bridge.py
        ``_build_behavior_score_matrix``).

        Item-item similarity from co-user co-occurrence (C = R^T R), top-k
        sparsified, propagated over each user's training history and per-user
        normalised into a signed evidence matrix ``[n_users, n_items]``.
        """
        train_mat = dataloader.inter_matrix(form="csr").astype(np.float32)
        n_users, n_items = train_mat.shape

        user_degree = np.asarray(train_mat.sum(axis=1)).reshape(-1).astype(np.float32)
        item_pop = np.asarray(train_mat.sum(axis=0)).reshape(-1).astype(np.float32)

        # Item-item co-occurrence C = R^T R with self-loops removed. Kept SPARSE
        # (CSR) throughout: every normalisation below maps a zero co-occurrence
        # to a zero similarity, so the similarity's non-zero pattern equals
        # cooc's. We never materialise the [n_items, n_items] dense grid, which
        # is what OOMs on large-item datasets (Clothing ~39k items).
        cooc = (train_mat.T @ train_mat).astype(np.float32)
        cooc.setdiag(0.0)
        cooc.eliminate_zeros()

        coo = cooc.tocoo()
        rows, cols = coo.row, coo.col
        vals = coo.data.astype(np.float32, copy=False)

        if self.behavior_sim_method == "cosine":
            denom = np.sqrt(np.maximum(item_pop, 1e-6))
            vals = vals / denom[rows] / denom[cols]
        elif self.behavior_sim_method == "jaccard":
            # union_ij = pop_i + pop_j - cooc_ij. On a non-zero cooc entry
            # pop_i, pop_j >= cooc_ij >= 1, so union > 0 -- matching the dense
            # ``where=union>0`` guard, which only ever zeroed already-zero cells.
            union = item_pop[rows] + item_pop[cols] - vals
            vals = vals / union
        elif self.behavior_sim_method == "logcooc":
            vals = np.log1p(vals)
        elif self.behavior_sim_method not in ("cooc", "raw"):
            raise ValueError(f"Unknown behavior_sim_method: {self.behavior_sim_method}")

        # The diagonal was removed on ``cooc`` and no normalisation reintroduces
        # it (rows != cols on every stored entry), so sim's diagonal is 0 --
        # equivalent to the dense ``np.fill_diagonal(sim, 0.0)``.
        sim = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))
        if 0 < self.behavior_topk < n_items:
            sim = self._prune_topk_per_row(sim, self.behavior_topk)

        # Per-user behavior scores as a SPARSE product R @ sim^T; densified only
        # here for the per-user aggregation / normalisation that follows.
        scores = train_mat.dot(sim.transpose())
        if not isinstance(scores, np.ndarray):
            scores = scores.toarray()
        scores = scores.astype(np.float32, copy=False)

        if self.behavior_aggregation == "mean":
            denom = np.maximum(user_degree, 1.0).reshape(-1, 1)
            scores = scores / denom
        elif self.behavior_aggregation == "sum_sqrt":
            denom = np.sqrt(np.maximum(user_degree, 1.0)).reshape(-1, 1)
            scores = scores / denom
        elif self.behavior_aggregation != "sum":
            raise ValueError(f"Unknown behavior_aggregation: {self.behavior_aggregation}")

        if self.behavior_score_norm == "z":
            mean = scores.mean(axis=1, keepdims=True)
            std = scores.std(axis=1, keepdims=True)
            scores = (scores - mean) / np.maximum(std, 1e-6)
        elif self.behavior_score_norm == "minmax":
            lo = scores.min(axis=1, keepdims=True)
            hi = scores.max(axis=1, keepdims=True)
            scores = (scores - lo) / np.maximum(hi - lo, 1e-6)
        elif self.behavior_score_norm != "none":
            raise ValueError(f"Unknown behavior_score_norm: {self.behavior_score_norm}")

        return scores.astype(np.float32, copy=False)

    @staticmethod
    def _prune_topk_per_row(sim, topk):
        """Keep only the ``topk`` largest entries of each CSR row, dropping the
        rest from the sparse structure -- the sparse analogue of the dense
        ``argpartition`` top-k prune. Rows with <= topk non-zeros are kept whole.

        Because the similarity is >= 0 with positives only on the co-occurrence
        support, taking the topk largest of a row's non-zeros is identical to the
        dense per-row argpartition: the dense version only ever selects a zero
        when a row has fewer than topk non-zeros, and those zeros contribute
        nothing to the pruned matrix.
        """
        sim = sim.tocsr()
        indptr, indices, data = sim.indptr, sim.indices, sim.data
        keep_rows, keep_cols, keep_vals = [], [], []
        for r in range(sim.shape[0]):
            start, end = indptr[r], indptr[r + 1]
            if end == start:
                continue
            row_cols = indices[start:end]
            row_vals = data[start:end]
            n_nz = row_vals.shape[0]
            if n_nz > topk:
                sel = np.argpartition(row_vals, n_nz - topk)[n_nz - topk:]
                row_cols = row_cols[sel]
                row_vals = row_vals[sel]
            keep_rows.append(np.full(row_cols.shape[0], r, dtype=np.int32))
            keep_cols.append(row_cols)
            keep_vals.append(row_vals)
        if keep_rows:
            out_rows = np.concatenate(keep_rows)
            out_cols = np.concatenate(keep_cols)
            out_vals = np.concatenate(keep_vals)
        else:
            out_rows = out_cols = np.empty(0, dtype=np.int32)
            out_vals = np.empty(0, dtype=np.float32)
        return sp.csr_matrix((out_vals, (out_rows, out_cols)), shape=sim.shape)

    def _candidate_behavior_scores(self, users, candidate_idx):
        """Gather signed behavior evidence for a candidate item set
        (ref bridge.py ``_candidate_behavior_scores`` dense branch)."""
        return self.behavior_score_matrix[users].gather(1, candidate_idx)

    def _pair_behavior_scores(self, users, items):
        """Signed behavior evidence for (user, item) pairs (ref bridge.py)."""
        return self.behavior_score_matrix[users, items]

    # ---------------------------------------------------- spectral split (BGCC)
    def _frequency_views(self, user_emb, item_emb):
        """SVD low/high spectral split (ref bridge.py ``_frequency_views`` +
        the encoder's SVD band decomposition). The first ``low_band_count``
        singular-value bands form the low-frequency shared view; the remainder
        are the high-frequency discriminative view."""
        low_user, high_user = self._svd_low_high(user_emb)
        low_item, high_item = self._svd_low_high(item_emb)
        return low_user, low_item, high_user, high_item

    def _svd_low_high(self, rep):
        M = max(self.num_freq_bands, 1)
        feat_dim = rep.shape[1]
        try:
            U, S, Vh = torch.linalg.svd(rep, full_matrices=False)
        except torch._C._LinAlgError:
            # cusolver's GPU SVD can fail to converge on ill-conditioned /
            # repeated-singular-value embedding matrices (observed during real
            # Baby training with residual_weight>0). Retry on the robust CPU
            # LAPACK path for THIS known error only; if that also fails, the
            # error propagates (no blanket suppression).
            U_c, S_c, Vh_c = torch.linalg.svd(rep.cpu(), full_matrices=False)
            U, S, Vh = U_c.to(rep.device), S_c.to(rep.device), Vh_c.to(rep.device)

        base = feat_dim // M
        remainder = feat_dim % M
        split_sizes = [base + (1 if i < remainder else 0) for i in range(M)]
        low_count = min(max(self.low_band_count, 1), M)
        low_dim = int(sum(split_sizes[:low_count]))
        low_dim = min(max(low_dim, 1), feat_dim)

        low = (U[:, :low_dim] * S[:low_dim]) @ Vh[:low_dim, :]
        high = rep - low
        return low, high

    # ------------------------------------------------------------- calibration
    def _pair_gate(self, users, items, behavior_scores):
        """Learned per-(user,item) gate metering calibration strength
        (ref bridge.py ``_pair_gate``)."""
        user_bias = self.behavior_user_bias(users)
        item_bias = self.behavior_item_bias(items)
        if items.dim() == 1:
            user_bias = user_bias.view(-1)
            item_bias = item_bias.view(-1)
        else:
            user_bias = user_bias.view(-1, 1)
            item_bias = item_bias.squeeze(-1)
        logits = self.gate_base_scale + user_bias + item_bias
        logits = logits + self.gate_behavior_scale * torch.tanh(behavior_scores.detach())
        return torch.sigmoid(logits)

    def _combine(self, users, items, base_scores, behavior_scores, residual_scores):
        """Combine base score with gated behavior + residual calibration
        (ref bridge.py ``_combine_score_components``)."""
        gate = self._pair_gate(users, items, behavior_scores)
        behavior_correction = self.behavior_weight * gate * behavior_scores
        residual_correction = self.residual_weight * gate * residual_scores
        final = base_scores + behavior_correction + residual_correction
        return final, gate

    # ----------------------------------------------------------------- forward
    def forward(self):
        image_embedding = getattr(self, "image_embedding", None)
        text_embedding = getattr(self, "text_embedding", None)
        u_g, i_g = self.backbone(image_embedding, text_embedding)
        return u_g, i_g

    def calculate_loss(self, interaction):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        # Consume the dataloader's clean per-user history-avoiding negatives when
        # supplied (use_neg_sampling=true emits a 3-tuple), matching the other
        # baselines; only fall back to uniform sampling for the 2-tuple contract.
        if len(interaction) >= 3:
            neg_items = interaction[2].to(self.device)
        else:
            neg_items = torch.randint(
                0, self.n_items, pos_items.shape, device=self.device
            )

        u_emb, i_emb = self.forward()
        low_user, low_item, high_user, high_item = self._frequency_views(u_emb, i_emb)

        u = u_emb[users]
        pos_base = torch.sum(u * i_emb[pos_items], dim=1)
        neg_base = torch.sum(u * i_emb[neg_items], dim=1)

        # High-frequency residual signal = high-view score minus low-view score
        # (ref bridge.py: residual = high - low).
        lu = low_user[users]
        hu = high_user[users]
        pos_low = torch.sum(lu * low_item[pos_items], dim=1)
        neg_low = torch.sum(lu * low_item[neg_items], dim=1)
        pos_high = torch.sum(hu * high_item[pos_items], dim=1)
        neg_high = torch.sum(hu * high_item[neg_items], dim=1)
        pos_residual = pos_high - pos_low
        neg_residual = neg_high - neg_low

        pos_behavior = self._pair_behavior_scores(users, pos_items)
        neg_behavior = self._pair_behavior_scores(users, neg_items)

        pos_final, pos_gate = self._combine(users, pos_items, pos_base, pos_behavior, pos_residual)
        neg_final, neg_gate = self._combine(users, neg_items, neg_base, neg_behavior, neg_residual)

        # Calibrated BPR (ref bridge.py ``_bpr_from_scores``).
        bpr = -F.logsigmoid(pos_final - neg_final).mean()

        # Gate regularisation pulls the mean gate toward gate_target.
        gate_mean = torch.cat([pos_gate, neg_gate], dim=0).mean()
        gate_reg = (gate_mean - self.gate_target) ** 2

        # L2 regularisation over the involved embeddings.
        reg = (
            u.pow(2).sum()
            + i_emb[pos_items].pow(2).sum()
            + i_emb[neg_items].pow(2).sum()
        ) / (2 * users.shape[0])

        return bpr + self.gate_reg_weight * gate_reg + self.reg_weight * reg

    def full_sort_predict(self, interaction):
        """Full-sort scoring with top-K candidate calibration (ref bridge.py
        ``full_sort_predict`` topk scope). The behavior evidence adjusts only
        the base top-K shortlist; the rest keep the raw backbone score."""
        users = interaction[0].to(self.device)
        u_emb, i_emb = self.forward()
        u = u_emb[users]
        base_scores = torch.matmul(u, i_emb.transpose(0, 1))

        # No calibration -> raw backbone scores (behavior_weight 0 recovers base).
        if self.behavior_weight == 0.0 and self.residual_weight == 0.0:
            return base_scores

        low_user, low_item, high_user, high_item = self._frequency_views(u_emb, i_emb)

        # Candidate shortlist C_u = base top-K items (ref bridge.py behavior_eval_topk).
        # Calibration is applied ONLY to these indices (the paper's 1[i in C_u]);
        # every other item keeps its raw base score via the untouched clone below.
        candidate_k = min(max(self.behavior_eval_topk, 1), self.n_items)
        _, candidate_idx = torch.topk(base_scores, k=candidate_k, dim=1)
        candidate_behavior = self._candidate_behavior_scores(users, candidate_idx)

        high_scores = torch.matmul(high_user[users], high_item.transpose(0, 1)).gather(1, candidate_idx)
        low_scores = torch.matmul(low_user[users], low_item.transpose(0, 1)).gather(1, candidate_idx)
        candidate_residual = high_scores - low_scores

        gate = self._pair_gate(users, candidate_idx, candidate_behavior)
        correction = (
            self.behavior_weight * gate * candidate_behavior
            + self.residual_weight * gate * candidate_residual
        )

        final_scores = base_scores.clone()
        final_scores.scatter_add_(1, candidate_idx, correction)
        return final_scores
