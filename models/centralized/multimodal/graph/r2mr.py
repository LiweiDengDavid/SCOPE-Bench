# coding: utf-8
r"""
R2MR
################################################
Reference:
    Review and Rewrite: Modality-consensus multimodal recommendation.
    KDD'2025 (official model class ``RMR``).
    Official code: models/rmr.py (+ common/trainer.py rewrite orchestration,
    common/abstract_recommender.py PCA+tanh feature preprocessing).

R2MR treats the pre-extracted modality features as *noisy reviews* to be audited
and, where confident, rewritten:

  (a) **Trainable modality features.** ``v_feat``/``t_feat`` start from the
      pre-extracted embeddings (``nn.Parameter``, faithful to the official
      ``nn.Embedding.from_pretrained(freeze=False).weight``). The official
      ``abstract_recommender`` first PCA-reduces ``v_feat`` to the text dimension
      and squashes BOTH modalities through ``tanh``; that preprocessing is
      reproduced here in ``__init__`` (cached to the dataset dir). Consequently
      ``dim_latent`` equals the text feature dimension.

  (b) **Reviewer (``review_modal``).** For each modality, a per-item consensus
      quality score in (0,1) is computed from the similarity between a projected
      user-id feature and the modality feature, masked by the user-item
      interactions and averaged over each item's interacting users. High score =
      the item's modality feature agrees with what its users' id-embeddings
      predict.

  (c) **In-forward soft rewrite.** Each modality feature is blended toward
      ``temp_user = tanh(iu_graph @ dropout(user_id))`` by its review score:
      ``feat <- feat*(score+1e-3) + (1-score)*temp_user`` — trusted features are
      largely kept, distrusted ones are pulled toward the id-graph signal.

  (d) **Propagation + scoring.** ``_gcn_pp`` runs two rounds of symmetric-
      normalized UI/IU propagation with a residual sum (user side seeded by a
      learnable ``{t,v}_preference``); the user/item reps are ``cat(t, v)`` and
      scored by dot product.

  (e) **BPR loss** (the official LIVE path — the repo's ``calculate_loss`` is a
      dead DualGNN copy-paste that crashes, and is NOT ported):
      ``-mean(logsigmoid(pos-neg)) + reg_weight * (0.5*Σ‖emb‖²)/batch``. The
      official live path HARDCODES the coefficient — ``emb_loss = 1e-4 *
      regularizer`` (models/rmr.py:330); the official ``reg_weight`` config key
      is dead. We keep ``reg_weight`` live + HPO-searched (round convention);
      the default 1e-4 matches the official hardcode.

  (f) **Rewriter (VQGAN).** Every ``rewrite_period`` epochs (up to
      ``max_rewrites`` times), items whose *both* review scores clear
      ``thresholds`` train a small bidirectional VQGAN (Encoder + vector-quantized
      Codebook + Decoder + a GAN Discriminator, prompt token 0=text->image,
      1=image->text). The cross-generated features are re-scored by the Reviewer
      and injected only where the new score clears ``thresholds``.

Deviations from the official code (documented):
  * **Chunked sparse Reviewer.** The official ``review_modal`` materializes a
    dense ``[n_items, n_users]`` interaction mask and runs an fp16 ``.half()``
    score matmul (O(n_users·n_items) memory). Here the interaction mask is kept
    sparse (CSR) and the per-item consensus is accumulated by chunking over items
    in full precision — numerically the same masked mean, without the dense/fp16
    blow-up.
  * **VQGAN rewriter folded into the model.** The official orchestration lives
    in the trainer's ``fit()`` loop (common/trainer.py:335-339): with 0-based
    ``epoch_idx`` (``start_epoch = 0``), the rewrite runs AFTER
    ``_train_epoch(epoch_idx)`` when ``epoch_idx % rewrite_period == 0 and
    epoch_idx != 0`` — epochs 6, 12, 18, ... complete, THEN the features are
    rewritten, so the model first trains on rewritten features at 0-based
    epochs 7, 13, 19, ... Our trainer calls ``pre_epoch_processing()`` BEFORE
    each epoch with NO arguments, so the model counts epochs itself and fires
    the rewrite before 0-based epoch ``e`` iff ``e-1 > 0 and (e-1) %
    rewrite_period == 0`` — the exact same epochs see rewritten features. The
    VQGAN sub-modules are held in a plain (non-``nn.Module``-registered)
    container so their parameters never leak into the framework-built main
    optimizer; the VQGAN trains with its own AdamW/Adam optimizers.
  * **Stable trainable set; rewritten features live in frozen BUFFERS.** The
    official ``update_modal_feat`` (models/rmr.py:261-268) ``del``s the feature
    Parameters and registers FRESH ``nn.Embedding.from_pretrained(...,
    freeze=False).weight`` Parameters; the official optimizer is built ONCE
    (common/trainer.py:87) and never rebuilt, so those fresh Parameters are
    never stepped — post-rewrite features receive gradients but stay FROZEN,
    while every other parameter keeps its Adam moments. Here the ORIGINAL
    ``t_feat``/``v_feat`` Parameters stay registered forever and the rewritten
    values live in persistent buffers that ``forward`` reads once
    ``rewrite_active`` flips: the trainable-parameter id-set never changes, so
    the framework's ``_refresh_optimizer`` never rebuilds the main optimizer
    (Adam moments persist, matching official) and the rewritten features are
    frozen (matching official).
  * **Constant LR.** The official trainer uses ``LambdaLR(1 - epoch/70)``,
    which linearly DECAYS the learning rate to 0 by epoch 70 and drives it
    negative beyond (a bug). Ours trains at a constant lr (the framework's
    standard scheduler) — no decay.
  * Dead ``torch_geometric`` / ``torch_scatter`` imports, the unused audio branch,
    ``self_attention``, ``Modal_Reviewer``, and the ``GCN``/``Base_gcn`` message-
    passing classes are stripped; ``.cuda()`` becomes ``self.device``.

All experiment-affecting numeric literals live in ``configs/models/R2MR.yaml``.
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from core.base import RecommenderBase


class R2MR(RecommenderBase):
    def __init__(self, config, dataloader):
        super(R2MR, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        # --- searchable / structural knobs (all from YAML) ---
        self.reg_weight = float(config["reg_weight"])
        self.thresholds = float(config["thresholds"])
        self.rewrite_period = int(config["rewrite_period"])
        self.max_rewrites = int(config["max_rewrites"])

        # --- VQGAN two-stage rewriter config (all from YAML) ---
        self._vq_cfg = {
            "code_book_num_vector": int(config["vq_code_book_num_vector"]),
            "code_dim": int(config["vq_code_dim"]),
            "beta": float(config["vq_beta"]),
            "gan_weight": float(config["vq_gan_weight"]),
            "enc_out_dim": int(config["vq_enc_out_dim"]),
            "prompt_dim": int(config["vq_prompt_dim"]),
            "enc_channels": list(config["vq_enc_channels"]),
            "dec_channels": list(config["vq_dec_channels"]),
            "dis_channels": list(config["vq_dis_channels"]),
            "vq_lr": float(config["vq_lr"]),
            "dis_lr": float(config["vq_dis_lr"]),
            "beta1": float(config["vq_beta1"]),
            "beta2": float(config["vq_beta2"]),
            "ts_epochs_first": int(config["ts_epochs_first"]),
            "ts_epochs_rest": int(config["ts_epochs_rest"]),
            "ts_batch_size": int(config["ts_batch_size"]),
        }

        assert self.t_feat is not None and self.v_feat is not None, (
            "R2MR requires both text and visual features"
        )

        # --- feature preprocessing: PCA(v -> text-dim) + tanh(both) ---
        # Faithful to the official abstract_recommender. dim_latent == text dim.
        self.dim_latent = int(self.t_feat.shape[1])
        t_feat_np, v_feat_np = self._preprocess_features(config)

        # Trainable modality features (faithful from_pretrained(freeze=False)).
        self.t_feat = nn.Parameter(torch.from_numpy(t_feat_np).to(self.device))
        self.v_feat = nn.Parameter(torch.from_numpy(v_feat_np).to(self.device))
        # Rewritten-feature storage. The Parameters above stay registered
        # FOREVER so the trainable-parameter id-set never changes across VQGAN
        # rewrites (the framework's _refresh_optimizer would otherwise rebuild
        # the main optimizer and reset every Adam moment; the official trainer
        # builds its optimizer once and never rebuilds). After the first
        # rewrite, forward reads these frozen persistent buffers instead —
        # official post-rewrite freeze semantics (see module docstring and
        # _inject_rewritten_features).
        self.register_buffer("t_feat_rewritten", torch.zeros_like(self.t_feat))
        self.register_buffer("v_feat_rewritten", torch.zeros_like(self.v_feat))
        self.register_buffer("rewrite_active", torch.zeros((), dtype=torch.bool))

        self.dropout = nn.Dropout(p=self.dropout_rate)

        # --- sparse sym-normalized UI / IU propagation graphs ---
        train_interactions = dataloader.inter_matrix(form="coo").astype(np.float32)
        self.ui_graph = self._matrix_to_sparse_tensor(
            self._csr_norm(train_interactions)
        )
        self.iu_graph = self._matrix_to_sparse_tensor(
            self._csr_norm(train_interactions.T.tocoo())
        )

        # Interaction mask kept SPARSE (CSR) for the chunked Reviewer; the
        # official dense [n_items, n_users] mask is never materialized.
        # item_user_csr[i] = users who interacted with item i.
        self._item_user_csr = train_interactions.T.tocsr().astype(np.float32)
        item_degree = np.asarray(self._item_user_csr.sum(axis=1)).reshape(-1)
        # 1.0 for items with no interactions (matches the official +adj_bool
        # denominator guard), else 0.0.
        self.register_buffer(
            "_item_no_inter",
            torch.from_numpy((item_degree == 0.0).astype(np.float32)),
        )
        # Sparse item-user indicator tensor [n_items, n_users] for masked sums.
        self._item_user_sparse = self._coo_to_sparse_tensor(
            self._item_user_csr.tocoo()
        )

        # --- learnable id embedding + per-modality user preference seeds ---
        self.user_id_embedding = nn.Parameter(
            nn.init.uniform_(torch.zeros(self.n_users, self.dim_latent), a=-1.0, b=1.0)
        )
        self.t_preference = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(self.n_users, self.dim_latent), gain=1.0)
        )
        self.v_preference = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(self.n_users, self.dim_latent), gain=1.0)
        )

        # --- Reviewer MLPs ---
        self.MLP_review = nn.Linear(self.dim_latent, self.dim_latent)
        self.MLP_review_t = nn.Linear(self.dim_latent, self.dim_latent)
        self.MLP_review_v = nn.Linear(self.dim_latent, self.dim_latent)
        nn.init.uniform_(self.MLP_review.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.MLP_review_t.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.MLP_review_v.weight, a=-1.0, b=1.0)

        # Reviewer outputs (populated by forward; consumed by the rewriter).
        self.t_score = None
        self.v_score = None

        # --- rewriter epoch bookkeeping (folded pre_epoch_processing loop) ---
        # cur_epoch counts pre-epoch hook calls == the 0-based index of the
        # epoch ABOUT to train (the schedule derivation lives in
        # pre_epoch_processing).
        self.cur_epoch = 0
        self.rewrite_count = 0
        # Lazily-built VQGAN two-stage trainer (kept OUT of the registered module
        # tree so its params never enter the main optimizer).
        self._vqgan = None

    # ------------------------------------------------------------------ setup
    def _preprocess_features(self, config):
        """PCA(v_feat -> text-dim) then tanh(both). Cached to the dataset dir.

        Returns float32 numpy arrays (t_feat, v_feat) both [n_items, dim_latent].
        """
        t_np = self.t_feat.detach().cpu().numpy().astype(np.float32)
        v_np = self.v_feat.detach().cpu().numpy().astype(np.float32)

        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        cache_v = os.path.join(
            dataset_path, "r2mr_pca_v_{}.npy".format(self.dim_latent)
        )

        # DEVIATION (latent): the official abstract_recommender ALWAYS runs
        # PCA(dim_latent).fit_transform on v_feat -- a centering + rotation even
        # when the input dim already equals dim_latent; we skip it in that
        # equal-dim case (unreachable on baby/sports/clothing: 4096 -> text-dim).
        if v_np.shape[1] == self.dim_latent:
            v_reduced = v_np
        elif os.path.isfile(cache_v):
            v_reduced = np.load(cache_v).astype(np.float32)
        else:
            pca = PCA(n_components=self.dim_latent)
            v_reduced = pca.fit_transform(v_np).astype(np.float32)
            if os.path.isdir(dataset_path):
                np.save(cache_v, v_reduced)

        t_out = np.tanh(t_np).astype(np.float32)
        v_out = np.tanh(v_reduced).astype(np.float32)
        return t_out, v_out

    def _csr_norm(self, coo_mat):
        """Symmetric d^-1/2 A d^-1/2 normalization (faithful csr_norm)."""
        csr_mat = coo_mat.tocsr()
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.0
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.0
        colsum_diag = sp.diags(colsum)

        return rowsum_diag * csr_mat * colsum_diag

    def _matrix_to_sparse_tensor(self, mat):
        coo = mat.tocoo() if not sp.isspmatrix_coo(mat) else mat
        return self._coo_to_sparse_tensor(coo)

    def _coo_to_sparse_tensor(self, coo):
        coo = coo.astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((coo.row, coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(coo.data.astype(np.float32))
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(
            indices, values, shape, dtype=torch.float32
        ).coalesce().to(self.device)

    # ------------------------------------------------------------------ core
    def _mm(self, x, y):
        return torch.sparse.mm(x, y)

    def review_modal(self, modal_feat, index=None, check_pattern="none"):
        """Per-item/modality consensus quality score in (0,1).

        Chunked SPARSE reformulation of the official dense-mask + fp16 matmul:
        for each modality feature ``feat`` ([m, dim]) and the projected user-id
        feature ``user_id_feat`` ([n_users, dim]), the per-(item,user) agreement
        is ``sigmoid(user_id_feat @ feat.T)``; masked by interactions and averaged
        over each item's interacting users. We accumulate the masked row-sums via
        the sparse item-user indicator: no dense [m, n_users] tensor and no fp16
        score matmul (full-precision, sparse in the interaction count).

        ``check_pattern`` in {"text","image"} re-scores VQGAN-generated features
        for a subset of items given by ``index`` (used by the rewriter).
        """
        user_id_feat = self.MLP_review(self.dropout(self.user_id_embedding))

        if check_pattern == "text":
            feats = [self.MLP_review_t(modal_feat[0])]
            item_user = torch.index_select(self._item_user_sparse, 0, index).coalesce()
            no_inter = self._item_no_inter[index]
        elif check_pattern == "image":
            feats = [self.MLP_review_v(modal_feat[0])]
            item_user = torch.index_select(self._item_user_sparse, 0, index).coalesce()
            no_inter = self._item_no_inter[index]
        else:
            feats = modal_feat
            item_user = self._item_user_sparse
            no_inter = self._item_no_inter

        # Per-item interaction count (denominator); +no_inter guards empty rows.
        deg = torch.sparse.sum(item_user, dim=1).to_dense()
        denom = deg + no_inter

        review_list = []
        for feat in feats:
            # score[i, u] = sigmoid(<user_id_feat[u], feat[i]>); we need, per item
            # i, the sum over interacting users u of score[i, u]. Compute it as a
            # sparse-masked reduction chunked over items to bound memory.
            feat_score = self._masked_consensus(user_id_feat, feat, item_user, denom)
            review_list.append(feat_score.unsqueeze(1))
        return torch.cat(review_list, dim=1)

    def _masked_consensus(self, user_id_feat, feat, item_user, denom):
        """Sum_{u in N(i)} sigmoid(<user_id_feat[u], feat[i]>) / denom[i].

        Chunked over items; only the (item, interacting-user) entries are ever
        evaluated, so this stays sparse in the interaction count.
        """
        m = feat.shape[0]
        out = torch.zeros(m, device=feat.device, dtype=feat.dtype)
        item_user = item_user.coalesce()
        # ``item_user`` row ids are already 0..m-1: for the full pass they index
        # all items, and for a rewriter subset ``index_select`` re-bases them to
        # 0..m-1 (verified), so they line up with the 0-indexed ``feat`` rows.
        item_rows = item_user.indices()[0]
        user_cols = item_user.indices()[1]
        # Only the (item, interacting-user) entries are scored: per nonzero,
        # sigmoid(<feat[item], user_id_feat[user]>), scatter-summed per item.
        logits = (feat[item_rows] * user_id_feat[user_cols]).sum(dim=1)
        out = out.index_add(0, item_rows, torch.sigmoid(logits))
        return out / denom

    def _gcn_pp(self, feat, preference, uig, iug):
        """Two rounds of sym-normalized UI/IU propagation with residual sum.

        Faithful to the official ``_gcn_pp(norm=True)`` path.
        """
        item_res = item_embed = F.normalize(feat)
        user_res = user_embed = F.normalize(preference)
        for _ in range(2):
            user_agg = self._mm(uig, item_embed)
            item_agg = self._mm(iug, user_embed)
            item_embed = item_agg
            user_embed = user_agg
            item_res = item_res + item_embed
            user_res = user_res + user_embed
        return user_res, item_res

    def _active_modal_feats(self):
        """(t_feat, v_feat) actually consumed by forward: the trainable
        Parameters before the first VQGAN rewrite, the frozen rewritten buffers
        after (official post-rewrite freeze — see _inject_rewritten_features)."""
        if bool(self.rewrite_active):
            return self.t_feat_rewritten, self.v_feat_rewritten
        return self.t_feat, self.v_feat

    def forward(self):
        t_feat, v_feat = self._active_modal_feats()

        modal_scores = self.review_modal(
            [self.MLP_review_t(t_feat), self.MLP_review_v(v_feat)]
        )
        t_score = modal_scores[:, 0].unsqueeze(-1)
        v_score = modal_scores[:, 1].unsqueeze(-1)
        self.t_score = t_score.squeeze(-1)
        self.v_score = v_score.squeeze(-1)

        # In-forward soft rewrite toward the id-graph signal.
        temp_user = torch.tanh(self._mm(self.iu_graph, self.dropout(self.user_id_embedding)))
        t_feat = t_feat * (t_score + 1e-3) + (1 - t_score) * temp_user
        v_feat = v_feat * (v_score + 1e-3) + (1 - v_score) * temp_user

        v_user_embed, v_item_embed = self._gcn_pp(
            v_feat, self.v_preference, self.ui_graph, self.iu_graph
        )
        t_user_embed, t_item_embed = self._gcn_pp(
            t_feat, self.t_preference, self.ui_graph, self.iu_graph
        )

        item_rep = torch.cat([t_item_embed, v_item_embed], dim=-1)
        user_rep = torch.cat([t_user_embed, v_user_embed], dim=-1)
        return user_rep, item_rep

    # ------------------------------------------------------------------ loss
    def calculate_loss(self, interaction):
        """BPR loss (the official LIVE ``bpr_loss`` path).

        Consumes dataloader-supplied negatives (``interaction[2]``) when present,
        else samples a uniform negative per positive.
        """
        user_embed, item_embed = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        if len(interaction) >= 3:
            neg_items = interaction[2]
        else:
            neg_items = torch.randint(
                0, self.n_items, pos_items.shape, device=pos_items.device
            )

        u_e = user_embed[users]
        pos_e = item_embed[pos_items]
        neg_e = item_embed[neg_items]

        pos_scores = torch.sum(torch.mul(u_e, pos_e), dim=1)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), dim=1)

        regularizer = (
            1.0 / 2 * (u_e ** 2).sum()
            + 1.0 / 2 * (pos_e ** 2).sum()
            + 1.0 / 2 * (neg_e ** 2).sum()
        ) / self.batch_size

        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        emb_loss = self.reg_weight * regularizer
        return mf_loss + emb_loss

    def full_sort_predict(self, interaction):
        user_tensor, item_tensor = self.forward()
        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    # -------------------------------------------------------------- rewriter
    def pre_epoch_processing(self):
        """Fold the official post-epoch VQGAN rewrite loop into the pre-epoch hook.

        Official ``common/trainer.py:335-339`` (0-based ``epoch_idx``,
        ``start_epoch = 0``): the rewrite runs AFTER ``_train_epoch(epoch_idx)``
        when ``epoch_idx % rewrite_period == 0 and epoch_idx != 0`` (and fewer
        than ``max_rewrites`` cycles so far, the official ``ts_cnt`` guard), so
        the model first TRAINS on rewritten features at 0-based epochs
        ``rewrite_period+1, 2*rewrite_period+1, ...``.

        The framework calls this hook BEFORE each epoch, so with ``e`` the
        0-based index of the epoch about to train, the rewrite that officially
        followed the just-completed epoch ``e-1`` must fire here iff
        ``e-1 > 0 and (e-1) % rewrite_period == 0`` — before epochs 7, 13, 19,
        ... for the default period 6 (``e-1 > 0`` both skips the first hook
        call, where no epoch has completed, and reproduces the official
        ``epoch_idx != 0`` exclusion).
        """
        completed = self.cur_epoch - 1  # 0-based index of the last completed epoch
        self.cur_epoch += 1
        if (
            completed > 0
            and completed % self.rewrite_period == 0
            and self.rewrite_count < self.max_rewrites
            and self.t_score is not None
            and self.v_score is not None
        ):
            self._run_rewrite_cycle()

    def _run_rewrite_cycle(self):
        """Train the bidirectional VQGAN on both-high items and inject accepted
        cross-generated features into the frozen rewritten-feature buffers."""
        was_training = self.training
        t_score = self.t_score.detach().cpu().numpy()
        v_score = self.v_score.detach().cpu().numpy()
        # Officially the trainer snapshots the CURRENT model features
        # (common/trainer.py:343) — the Parameters before the first rewrite,
        # the previously rewritten tensors afterwards.
        t_feat_cur, v_feat_cur = self._active_modal_feats()
        t_np = t_feat_cur.detach().cpu().numpy()
        v_np = v_feat_cur.detach().cpu().numpy()

        # Item grouping by the review thresholds.
        both_high = np.where((t_score >= self.thresholds) & (v_score >= self.thresholds))[0]
        text_high_vis_low = np.where(
            (t_score >= self.thresholds) & (v_score < self.thresholds)
        )[0]
        vis_high_text_low = np.where(
            (t_score < self.thresholds) & (v_score >= self.thresholds)
        )[0]

        # Nothing to learn from -> still count the cycle (matches official ts_cnt).
        if both_high.shape[0] == 0:
            self.rewrite_count += 1
            return

        if self._vqgan is None:
            self._vqgan = _VQGANTrainer(self._vq_cfg, self.dim_latent, self.device)

        ts_epochs = (
            self._vq_cfg["ts_epochs_first"]
            if self.rewrite_count == 0
            else self._vq_cfg["ts_epochs_rest"]
        )

        # Bidirectional training set: (text->image) and (image->text).
        train_feature = np.concatenate([t_np[both_high], v_np[both_high]], axis=0)
        train_label = np.concatenate([v_np[both_high], t_np[both_high]], axis=0)
        train_pattern = np.concatenate(
            [np.zeros(both_high.shape[0], dtype=np.int64),
             np.ones(both_high.shape[0], dtype=np.int64)],
            axis=0,
        )
        self._vqgan.train_stage(
            train_feature, train_pattern, train_label, ts_epochs,
            self._vq_cfg["ts_batch_size"],
        )

        # Generate the weak modality: text-high/vis-low -> synth image (prompt 0);
        # vis-high/text-low -> synth text (prompt 1). Start from the current
        # features; only Reviewer-accepted indices are overwritten below.
        new_v_full = v_np.copy()
        new_t_full = t_np.copy()

        if text_high_vis_low.shape[0] > 0:
            image_pred = self._vqgan.infer(
                t_np[text_high_vis_low],
                np.zeros(text_high_vis_low.shape[0], dtype=np.int64),
            )
            acc_idx, acc_feat = self._accept(
                image_pred, text_high_vis_low, check_pattern="image"
            )
            new_v_full[acc_idx] = acc_feat
        if vis_high_text_low.shape[0] > 0:
            text_pred = self._vqgan.infer(
                v_np[vis_high_text_low],
                np.ones(vis_high_text_low.shape[0], dtype=np.int64),
            )
            acc_idx, acc_feat = self._accept(
                text_pred, vis_high_text_low, check_pattern="text"
            )
            new_t_full[acc_idx] = acc_feat

        self._inject_rewritten_features(new_t_full, new_v_full)
        self.rewrite_count += 1
        if was_training:
            self.train()

    def _accept(self, pred, index, check_pattern):
        """Re-score generated features via the Reviewer; keep those > thresholds."""
        self.eval()
        with torch.no_grad():
            pred_tensor = torch.from_numpy(pred.astype(np.float32)).to(self.device)
            index_tensor = torch.from_numpy(index.astype(np.int64)).to(self.device)
            score = self.review_modal(
                [pred_tensor], index=index_tensor, check_pattern=check_pattern
            ).squeeze(-1)
            score = score.detach().cpu().numpy()
        keep = score > self.thresholds
        return index[keep], pred[keep]

    def _inject_rewritten_features(self, t_np, v_np):
        """Adopt rewritten features WITHOUT touching the trainable-parameter set.

        Official ``update_modal_feat`` (models/rmr.py:261-268) ``del``s the old
        feature Parameters and registers fresh ``nn.Embedding.from_pretrained(...,
        freeze=False).weight`` Parameters; the official optimizer is built ONCE
        (common/trainer.py:87) and never rebuilt, so the fresh Parameters are
        never stepped — post-rewrite features are FROZEN while every other
        parameter keeps its Adam moments. We replicate that by copying the
        rewritten values into persistent buffers and flipping ``rewrite_active``:
        the original Parameters stay registered (stable trainable id-set, so the
        framework's ``_refresh_optimizer`` never rebuilds the main optimizer)
        and ``forward`` reads the frozen buffers from now on.
        """
        self.t_feat_rewritten.copy_(
            torch.from_numpy(t_np.astype(np.float32)).to(self.device)
        )
        self.v_feat_rewritten.copy_(
            torch.from_numpy(v_np.astype(np.float32)).to(self.device)
        )
        self.rewrite_active.fill_(True)


# --------------------------------------------------------------------------
# VQGAN two-stage rewriter (kept OUT of R2MR's registered module tree so its
# parameters never enter the framework-built main optimizer). Faithful to the
# official Encoder / Cookbook (VQ) / Decoder / Discriminator / VQGANTrainer.
# --------------------------------------------------------------------------
class _Linears(nn.Module):
    def __init__(self, inp_dim, out_dim, ln=True):
        super().__init__()
        self.ln = ln
        self.layer = nn.Linear(inp_dim, out_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        if self.ln:
            self.LN = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.layer(x)
        if self.ln:
            x = self.LN(x) + x
        return x


class _MLPStack(nn.Module):
    def __init__(self, inp_dim, out_dim, channel_list, final_ln=False):
        super().__init__()
        channels = [inp_dim] + list(channel_list)
        layers = [
            _Linears(channels[i], channels[i + 1], ln=True)
            for i in range(len(channels) - 1)
        ]
        layers.append(_Linears(channels[-1], out_dim, ln=final_ln))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _Cookbook(nn.Module):
    """Vector-quantization codebook with straight-through estimator."""

    def __init__(self, num_codebook_vectors, code_dim, beta):
        super().__init__()
        self.code_dim = code_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_codebook_vectors, code_dim)
        nn.init.xavier_normal_(self.embedding.weight, gain=1.0)

    def forward(self, z):
        z_r = z.view(-1, self.code_dim).contiguous()
        d = (
            torch.sum(z_r ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_r, self.embedding.weight.t())
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        code_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )
        z_q = z + (z_q - z).detach()
        return z_q, code_loss


class _Discriminator(nn.Module):
    def __init__(self, inp_dim, dis_channel_list):
        super().__init__()
        channels = [inp_dim] + list(dis_channel_list)
        layers = [
            _Linears(channels[i], channels[i + 1], ln=True)
            for i in range(len(channels) - 1)
        ]
        layers.append(nn.Linear(channels[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class _VQGAN4Rec(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.prompt_embed = nn.Embedding(2, cfg["prompt_dim"])
        self.encoder = _MLPStack(
            feat_dim + cfg["prompt_dim"], cfg["enc_out_dim"], cfg["enc_channels"],
            final_ln=False,
        )
        self.decoder = _MLPStack(
            cfg["enc_out_dim"], feat_dim, cfg["dec_channels"], final_ln=False
        )
        self.code_book = _Cookbook(
            cfg["code_book_num_vector"], cfg["code_dim"], cfg["beta"]
        )

    def forward(self, dense_feat, prompt_token):
        prompt_emb = self.prompt_embed(prompt_token)
        x = torch.cat([dense_feat, prompt_emb], dim=-1)
        encode_embed = self.encoder(x)
        code_embed, code_loss = self.code_book(encode_embed)
        decode_embed = self.decoder(code_embed)
        return decode_embed, code_loss


class _TSDataset(Dataset):
    def __init__(self, feature, pattern, label=None):
        self.feature = torch.from_numpy(feature.astype(np.float32))
        self.pattern = torch.from_numpy(pattern.astype(np.int64))
        self.label = None if label is None else torch.from_numpy(label.astype(np.float32))

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, idx):
        if self.label is None:
            return self.feature[idx], self.pattern[idx]
        return self.feature[idx], self.pattern[idx], self.label[idx]


class _VQGANTrainer:
    """Standalone two-stage (VQGAN + GAN) trainer with its OWN optimizers.

    Deliberately NOT an ``nn.Module`` child of R2MR: its parameters must never be
    picked up by the framework's main optimizer (faithful to the official design
    where the VQGAN is a separate object trained inside the trainer loop).
    """

    def __init__(self, cfg, feat_dim, device):
        self.cfg = cfg
        self.device = device
        self.vqgan = _VQGAN4Rec(cfg, feat_dim).to(device)
        self.discriminator = _Discriminator(feat_dim, cfg["dis_channels"]).to(device)
        self.opt_vq = torch.optim.AdamW(
            self.vqgan.parameters(), lr=cfg["vq_lr"], eps=1e-8,
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=cfg["dis_lr"], eps=1e-8,
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.criterion = nn.BCELoss()
        self.gan_weight = cfg["gan_weight"]

    def train_stage(self, feature, pattern, label, epochs, batch_size):
        self.vqgan.train()
        self.discriminator.train()
        loader = DataLoader(
            _TSDataset(feature, pattern, label),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )
        for _ in range(epochs):
            for feat, pat, lab in loader:
                feat = feat.to(self.device)
                pat = pat.to(self.device)
                lab = lab.to(self.device)
                decode_feat, code_loss = self.vqgan(feat, pat)
                rec_loss = torch.abs(decode_feat - lab).mean()

                real_label = torch.ones(lab.shape[0], 1, device=self.device)
                fake_label = torch.zeros(lab.shape[0], 1, device=self.device)
                disc_real = self.discriminator(lab)
                # Detach the generated features for the discriminator's fake
                # branch (standard GAN discriminator update). This matches the
                # official intent -- train the discriminator to tell real from
                # generated -- while avoiding the in-place-modification autograd
                # error the official two-backward/one-graph order (step opt_vq
                # between the VQ and GAN backward passes) triggers here: the
                # discriminator update no longer depends on generator params that
                # opt_vq.step() just mutated.
                disc_fake = self.discriminator(decode_feat.detach())
                gan_loss = self.gan_weight * (
                    self.criterion(disc_real, real_label)
                    + self.criterion(disc_fake, fake_label)
                )

                vq_loss = rec_loss + code_loss
                vq_loss.backward()
                self.opt_vq.step()
                self.opt_vq.zero_grad()

                gan_loss.backward()
                self.opt_disc.step()
                self.opt_disc.zero_grad()

    @torch.no_grad()
    def infer(self, feature, pattern):
        self.vqgan.eval()
        loader = DataLoader(
            _TSDataset(feature, pattern, None),
            batch_size=self.cfg["ts_batch_size"], shuffle=False, num_workers=0,
        )
        out = []
        for feat, pat in loader:
            feat = feat.to(self.device)
            pat = pat.to(self.device)
            decode_feat, _ = self.vqgan(feat, pat)
            out.append(decode_feat.detach().cpu().numpy())
        return np.concatenate(out, axis=0)
