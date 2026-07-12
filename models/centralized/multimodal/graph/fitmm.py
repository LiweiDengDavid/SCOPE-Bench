# coding: utf-8
r"""
FITMM -- Frequency-aware Information-bottleneck MultiModal recommendation
########################################################################
Reference:
    Paper: "FITMM: Frequency-aware Information-bottleneck for MultiModal
           Recommendation", ACM MM'2025.
    Official repo model class ``FITMM`` in ``src/models/fitmm.py``.

FITMM encodes id / visual / textual signals with three light bipartite GCN
branches. Each visual/textual branch projects its item features through a
two-Linear MLP with a ``leaky_relu`` in between (the id branch feeds the raw
id embedding, no MLP), stacks the per-branch user preference table on top,
L2-normalises the ego rows (``F.normalize``), and layer-sums ``num_layers``
symmetric-normalised propagations plus the normalised input. The three 64-d
views are concatenated into a 192-d representation; the item side is enhanced
by propagating over an item-item multimodal kNN graph (``mm_adj``), the user
side by the concatenated raw preference tables ``cat(id_pref, v_pref, t_pref)``
(the official ``user_modal_fused_input``; the optional user-graph propagation
over it is inert because ``user_emb.npy`` is required-absent). Base and
enhanced views are each decomposed into ``num_freq_bands`` spectral bands: the
192-d row is split into its three 64-d modality chunks and each chunk is
factorised with ``torch.linalg.svd``; the singular triplets are partitioned
into ``num_freq_bands`` contiguous groups, each group is reconstructed and
re-concatenated (SVD of the embedding matrix -- not a graph eigendecomposition
or an FFT), and per band the enhanced-view reconstruction is ADDED to the
base-view one.

Fusion (``TaskAwareFrequencyFusion``; SEPARATE ``fusion_layer_user`` /
``fusion_layer_item`` modules): a single-Linear gate ``freq_gate =
Linear(3*d -> M) + Sigmoid`` ("Pos"; Tanh for "Dual") reads the ORIGINAL
pre-fusion representation (``task_emb``) and emits per-node per-BAND scalars
``(N, M)``; ``band_gates = 1 + gate_scale * freq_gate(task_emb)`` with a
LEARNABLE scalar ``gate_scale`` (init 0.5). The fused output is
``sum_m sigmoid(freq_weights)_m * gate_m * band_m`` with learnable per-band
logits ``freq_weights`` (init ones).

This is a FAITHFUL port of the model's LIVE path only. The main objective is
**binary cross-entropy with logits** on positive / negative dot scores (NOT
BPR). Two auxiliary terms accompany it:

  * ``ib_loss``  -- a DETERMINISTIC information-bottleneck surrogate on the
    per-band gate delta ``delta = band_gates - 1`` of shape ``(N, M)``,
    computed inside the forward pass over ALL users + ALL items (official
    computes it in ``forward``, not on the batch slice):
    ``ib_alpha * mean_N(sum_M delta^2) + ib_mu * mean_N(||delta|| *
    sum_M relu(delta_thr - ib_phi_plus))``, where ``delta_thr`` is the ReLU'd
    delta for "Pos" and ``|delta|`` for "Dual" (the squared term keeps the
    signed delta). No noise sampling, so evaluation stays deterministic;
  * ``cl_loss``  -- the frequency contrastive loss over the batch band slices
    (users + POSITIVE items, JOINTLY): for every ordered band pair ``m != n``
    an MSE pulls ``cos(band_m, band_n)`` toward 1; for the upper-half bands
    (``m >= M // 2``) an InfoNCE term ``-mean(log_softmax(band @ band.T /
    tau))`` over the FULL logits matrix (raw dot products, no L2
    normalisation) is added; the whole sum is divided once by ``M * (M - 1)``.

Faithfulness / porting notes:

  * The per-chunk ``torch.linalg.svd`` is guarded by the BGCC
    ``torch._C._LinAlgError`` -> CPU-LAPACK retry pattern (``_svd_guarded``).
    Clustered / repeated singular values (e.g. rank-deficient modality
    projections) can make cusolver's GPU SVD fail to converge and can blow up
    the SVD backward; the CPU LAPACK path is the robust fallback and the error
    is only caught for that one class (no blanket suppression).
  * ``mm_adj`` follows the official ``get_knn_adj_mat`` +
    ``compute_normalized_laplacian``: cosine-topk neighbour indices over a
    ONES-valued adjacency with row-sum symmetric normalisation (uniform
    ~1/(knn_k + 1e-7) edge values), blended by ``mm_image_weight`` -- NOT the
    shared sim-weighted ``build_knn_normalized_graph`` helper. It is built once
    and cached to the dataset directory keyed by BOTH ``knn_k`` and
    ``mm_image_weight`` (the official cache guard was ``and False``; caching is
    re-enabled here with the blend weight in the key so a YAML change can never
    silently reuse a stale blend).
  * Dead code from the reference is NOT ported: ``User_Graph_sample``,
    ``MKMMDLoss``, ``InfoNCE_Loss``, ``IB_Layer``, ``TaskAwareFrequencyFusionMulti``,
    ``result_embed``, ``weight_i``, ``user_id_embedding``, the unused
    ``freq_gate_raw`` / residual ``alpha`` inside the fusion module and the
    unused module-level ``freq_weights`` (ones/M) of the decomposition stack,
    ``weighted_sum``/``weighted_max`` fusion branches, init-time edge dropout,
    and the ``user_emb.npy`` user-graph path (it crashes upstream and is
    required-absent here).
  * All experiment values live in ``configs/models/FITMM.yaml`` (dim_latent,
    knn_k, mm_image_weight, contrastive temperature, IB direction / alpha /
    mu / phi_plus, IB weight, CL weight). The fusion init constants
    (``gate_scale`` init 0.5, ``freq_weights`` init ones) are ARCHITECTURE
    constants hardcoded by the official code, not experiment knobs.

Config note: the reference's ``reg_weight`` is the *contrastive* loss weight
(a misnomer); it is exposed here as ``cl_weight`` for clarity and documented as
such (design decision: rename in YAML, keep behaviour faithful).
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


class TaskAwareFrequencyFusion(nn.Module):
    """Official ``TaskAwareFrequencyFusion`` (live path only).

    Per-node per-BAND scalar gates from the ORIGINAL pre-fusion representation:
    ``band_gates = 1 + gate_scale * freq_gate(task_emb)`` with ``freq_gate`` a
    single ``Linear(embed_dim -> M)`` + Sigmoid ("Pos") / Tanh ("Dual") and a
    LEARNABLE scalar ``gate_scale`` (init 0.5). Fused output =
    ``sum_m sigmoid(freq_weights)_m * gate_m * band_m`` (``freq_weights`` init
    ones). The IB surrogate is evaluated on the same ``(N, M)`` gates."""

    def __init__(self, num_bands, embed_dim, ib_direction, ib_alpha, ib_mu, ib_phi_plus):
        super().__init__()
        self.num_bands = num_bands
        self.ib_direction = ib_direction
        self.ib_alpha = ib_alpha
        self.ib_mu = ib_mu
        self.ib_phi_plus = ib_phi_plus
        # Learnable per-band fusion logits (official init: ones).
        self.freq_weights = nn.Parameter(torch.ones(num_bands))
        if ib_direction == "Dual":
            self.freq_gate = nn.Sequential(nn.Linear(embed_dim, num_bands), nn.Tanh())
        elif ib_direction == "Pos":
            self.freq_gate = nn.Sequential(nn.Linear(embed_dim, num_bands), nn.Sigmoid())
        else:
            raise ValueError(
                f"Invalid ib_direction: {ib_direction!r} (expected 'Pos' or 'Dual')"
            )
        # Learnable residual gate scale (official init: 0.5).
        self.gate_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def ib_surrogate_loss_from_gate(self, gate_values):
        """Official ``ib_surrogate_loss_from_gate`` on ``(N, M)`` gates.

        ``delta = gates - 1``; "Pos" ReLUs the delta (and thresholds it as-is),
        "Dual" keeps the signed delta for the squared term and thresholds
        ``|delta|``. ``term1 = ib_alpha * mean_N(sum_M delta^2)``;
        ``term2 = ib_mu * mean_N(sqrt(sum_M delta^2 + 1e-12) *
        sum_M relu(delta_thr - ib_phi_plus))``."""
        delta = gate_values - 1.0  # (N, M)
        if self.ib_direction == "Pos":
            delta = F.relu(delta)
            delta_for_threshold = delta
        else:  # "Dual" -- validated in __init__
            delta_for_threshold = delta.abs()
        delta_norm_sq = torch.sum(delta * delta, dim=1)  # (N,)
        term1 = self.ib_alpha * delta_norm_sq.mean()
        delta_norm = torch.sqrt(delta_norm_sq + 1e-12)  # (N,); official eps
        exceed = F.relu(delta_for_threshold - self.ib_phi_plus)  # (N, M)
        term2 = self.ib_mu * (delta_norm * torch.sum(exceed, dim=1)).mean()
        return term1 + term2

    def forward(self, band_components, task_emb):
        band_tensor = torch.stack(band_components, dim=1)  # (N, M, D)
        band_gates = 1.0 + self.gate_scale * self.freq_gate(task_emb)  # (N, M)
        ib_loss = self.ib_surrogate_loss_from_gate(band_gates)
        band_weights = torch.sigmoid(self.freq_weights).view(1, self.num_bands, 1)
        gated_bands = band_gates.unsqueeze(-1) * band_tensor  # (N, M, D)
        fused_emb = torch.sum(band_weights * gated_bands, dim=1)  # (N, D)
        return fused_emb, ib_loss


class FITMM(RecommenderBase):
    def __init__(self, config, dataloader):
        super(FITMM, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        # --- dimensions ---
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["num_layers"]
        self.n_mm_layers = config["n_mm_layers"]
        self.knn_k = config["knn_k"]
        self.mm_image_weight = float(config["mm_image_weight"])
        self.aggr_mode = config["aggr_mode"]

        # --- spectral band decomposition + residual gate ---
        self.num_freq_bands = int(config["num_freq_bands"])
        self.ib_direction = str(config["ib_direction"])
        self.cl_temperature = float(config["cl_temperature"])

        # --- information-bottleneck gate-delta surrogate ---
        self.ib_alpha = float(config["ib_alpha"])
        self.ib_mu = float(config["ib_mu"])
        self.ib_phi_plus = float(config["ib_phi_plus"])
        self.ib_weight = float(config["ib_weight"])

        # --- frequency contrastive loss weight (reference `reg_weight` misnomer) ---
        self.cl_weight = float(config["cl_weight"])

        # The fused representation is cat(id, v, t) -> 3 modality chunks of embed_dim.
        self.n_modalities = 3
        self.rep_dim = self.n_modalities * self.embedding_dim

        # --- U-I bipartite adjacency (symmetric normalised) ---
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = self._get_norm_adj_mat().to(self.device)

        # --- id / user-preference embeddings (one user-preference block per branch) ---
        # Official: item id table xavier_uniform_; the three per-branch user
        # preference tables xavier_normal_(gain=1) (official GCN.preference).
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.user_id_preference = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_v_preference = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_t_preference = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        nn.init.xavier_normal_(self.user_id_preference.weight, gain=1)
        nn.init.xavier_normal_(self.user_v_preference.weight, gain=1)
        nn.init.xavier_normal_(self.user_t_preference.weight, gain=1)

        # --- learnable per-modality user weighting weight_u [n_users, 3, 1] ---
        # Official init: xavier_normal_ logits re-materialised as softmax over
        # the 3 modality slots (rows sum to 1).
        self.weight_u = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(self.n_users, self.n_modalities, 1))
        )
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        # --- modality feature projections (dv/dt -> 4*d -> d) ---
        # Official GCN: MLP_1(F.leaky_relu(MLP(features))).
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], 4 * self.feat_embed_dim),
                nn.LeakyReLU(),
                nn.Linear(4 * self.feat_embed_dim, self.embedding_dim),
            )
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], 4 * self.feat_embed_dim),
                nn.LeakyReLU(),
                nn.Linear(4 * self.feat_embed_dim, self.embedding_dim),
            )

        # --- item-item multimodal kNN graph (cached; ref guard was `and False`) ---
        self.mm_adj = self._build_mm_adj().to(self.device)

        # --- per-side residual task-adaptive fusion (official separate modules) ---
        self.fusion_layer_user = TaskAwareFrequencyFusion(
            self.num_freq_bands, self.rep_dim, self.ib_direction,
            self.ib_alpha, self.ib_mu, self.ib_phi_plus,
        )
        self.fusion_layer_item = TaskAwareFrequencyFusion(
            self.num_freq_bands, self.rep_dim, self.ib_direction,
            self.ib_alpha, self.ib_mu, self.ib_phi_plus,
        )

    # ------------------------------------------------------------------ graphs
    def _get_norm_adj_mat(self):
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        ).tolil()
        R = self.interaction_matrix.tolil()
        A[: self.n_users, self.n_users:] = R
        A[self.n_users:, : self.n_users] = R.T
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

    def _get_knn_adj_mat(self, mm_embeddings):
        """Official ``get_knn_adj_mat``: cosine-topk neighbour indices, then the
        ones-valued normalized Laplacian (uniform ~1/(knn_k + 1e-7) edge
        values)."""
        context_norm = mm_embeddings.div(
            torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True)
        )
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0], device=knn_ind.device)
        indices0 = torch.unsqueeze(indices0, 1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return self._compute_normalized_laplacian(indices, adj_size)

    def _compute_normalized_laplacian(self, indices, adj_size):
        """Official ``compute_normalized_laplacian`` over a ONES-valued kNN
        adjacency (row-sum symmetric normalisation). Local copy on purpose --
        the shared ``build_knn_normalized_graph`` helper is sim-weighted and
        NOT what FITMM uses (same pattern as the freedom.py port)."""
        adj = torch.sparse_coo_tensor(
            indices, torch.ones_like(indices[0]), adj_size, dtype=torch.float32
        )
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size, dtype=torch.float32)

    def _build_mm_adj(self):
        """Item-item multimodal kNN Laplacian (official FITMM ``mm_adj``).

        Blends the per-modality ONES-valued cosine-kNN Laplacians by
        ``mm_image_weight``. Cached to the dataset directory as
        ``fitmm_mm_adj_knn{knn_k}_w{mm_image_weight}.pt`` -- the cache key
        includes BOTH knobs so a YAML change never reuses a stale blend (the
        official cache guard was ``and False``; caching re-enabled here)."""
        dataset_path = os.path.abspath(self.config["data_path"] + self.config["dataset"])
        os.makedirs(dataset_path, exist_ok=True)
        cache_file = os.path.join(
            dataset_path,
            f"fitmm_mm_adj_knn{self.knn_k}_w{self.mm_image_weight}.pt",
        )
        if os.path.exists(cache_file):
            return torch.load(cache_file, map_location="cpu", weights_only=False)

        image_adj = text_adj = None
        if self.v_feat is not None:
            image_adj = self._get_knn_adj_mat(self.image_embedding.weight.detach())
        if self.t_feat is not None:
            text_adj = self._get_knn_adj_mat(self.text_embedding.weight.detach())

        if image_adj is not None and text_adj is not None:
            mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        elif image_adj is not None:
            mm_adj = image_adj
        else:
            mm_adj = text_adj
        # Deliberately NOT coalesced: the official code propagates the raw
        # uncoalesced blend (duplicate image/text edges accumulate inside
        # torch.sparse.mm), and coalescing first changes fp32 rounding.
        torch.save(mm_adj, cache_file)
        return mm_adj

    # ---------------------------------------------------------- GCN propagation
    def _gcn_branch(self, user_pref, item_feats):
        """Layer-sum of ``num_layers`` symmetric-normalised bipartite convolutions
        plus the L2-NORMALISED input embedding (official FITMM ``GCN.forward``:
        ``x = F.normalize(cat(preference, features)); sum([x] + conv outputs)``).
        ``user_pref``/``item_feats`` are the per-branch user/item inputs."""
        ego = torch.cat((user_pref, item_feats), dim=0)
        ego = F.normalize(ego)
        all_embeddings = ego
        h = ego
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.norm_adj, h)
            all_embeddings = all_embeddings + h
        u_emb, i_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_emb, i_emb

    # ------------------------------------------------------------- spectral SVD
    def _svd_guarded(self, mat):
        """``torch.linalg.svd`` with the BGCC targeted CPU-LAPACK fallback.

        cusolver's GPU SVD can raise ``torch._C._LinAlgError`` (failed to
        converge) on ill-conditioned / repeated-singular-value matrices (e.g. a
        rank-deficient modality projection). Retry on the robust CPU LAPACK path
        for THIS known error only; if that also fails the error propagates (no
        blanket suppression)."""
        try:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        except torch._C._LinAlgError:
            U_c, S_c, Vh_c = torch.linalg.svd(mat.cpu(), full_matrices=False)
            U, S, Vh = U_c.to(mat.device), S_c.to(mat.device), Vh_c.to(mat.device)
        return U, S, Vh

    def _decompose_chunk(self, chunk):
        """Decompose one modality chunk [N, d] into ``num_freq_bands`` band
        reconstructions by partitioning its singular triplets into contiguous
        groups (ref ``frequency_decompose_svd_separate``)."""
        M = max(self.num_freq_bands, 1)
        U, S, Vh = self._svd_guarded(chunk)
        r = S.shape[0]
        base = r // M
        remainder = r % M
        sizes = [base + (1 if i < remainder else 0) for i in range(M)]
        bands = []
        start = 0
        for size in sizes:
            end = start + size
            if size > 0:
                # Official association order (U_i @ diag(S_i) @ V_i) -- kept
                # verbatim so fp32 rounding matches the reference bit-for-bit.
                band = U[:, start:end] @ torch.diag(S[start:end]) @ Vh[start:end, :]
            else:
                band = torch.zeros_like(chunk)
            bands.append(band)
            start = end
        return bands

    def _frequency_decompose(self, rep):
        """Split [N, 3*d] into three modality chunks, decompose each into bands,
        and re-concatenate per band -> list of ``num_freq_bands`` tensors [N, 3*d]."""
        chunks = torch.split(rep, self.embedding_dim, dim=1)
        per_chunk_bands = [self._decompose_chunk(c) for c in chunks]
        M = max(self.num_freq_bands, 1)
        bands = [
            torch.cat([per_chunk_bands[c][m] for c in range(self.n_modalities)], dim=1)
            for m in range(M)
        ]
        return bands

    # ----------------------------------------------------------------- forward
    def _encode(self):
        """Encode users/items into fused 192-d representations. Returns the fused
        reps, the per-band enhanced representations needed by the CL loss, and
        the full-node ``ib_loss`` (official computes it in ``forward``)."""
        item_id_feats = self.item_id_embedding.weight
        u_id, i_id = self._gcn_branch(self.user_id_preference.weight, item_id_feats)

        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        else:
            image_feats = torch.zeros(self.n_items, self.embedding_dim, device=self.device)
        u_v, i_v = self._gcn_branch(self.user_v_preference.weight, image_feats)

        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        else:
            text_feats = torch.zeros(self.n_items, self.embedding_dim, device=self.device)
        u_t, i_t = self._gcn_branch(self.user_t_preference.weight, text_feats)

        # Item representation = cat(id, v, t) [n_items, 3*d].
        item_rep = torch.cat((i_id, i_v, i_t), dim=1)
        # User representation = per-modality slices scaled by learnable weight_u.
        user_stack = torch.stack((u_id, u_v, u_t), dim=1)  # [n_users, 3, d]
        user_rep = (user_stack * self.weight_u).reshape(self.n_users, self.rep_dim)

        # Item-side graph enhancement: propagate over mm_adj.
        item_graph = item_rep
        for _ in range(self.n_mm_layers):
            item_graph = torch.sparse.mm(self.mm_adj, item_graph)
        # User-side enhancement input = cat of the RAW preference tables
        # (official ``user_modal_fused_input = cat(id/v/t preference)``; the
        # user-graph propagation over it is inert -- user_emb.npy is
        # required-absent -- so the input passes through, NOT user_rep).
        user_graph = torch.cat(
            (
                self.user_id_preference.weight,
                self.user_v_preference.weight,
                self.user_t_preference.weight,
            ),
            dim=1,
        )

        # Spectral bands; per band the enhanced view is added to the base rep.
        user_rep_bands = self._frequency_decompose(user_rep)
        item_rep_bands = self._frequency_decompose(item_rep)
        user_graph_bands = self._frequency_decompose(user_graph)
        item_graph_bands = self._frequency_decompose(item_graph)

        M = max(self.num_freq_bands, 1)
        user_bands = [user_graph_bands[m] + user_rep_bands[m] for m in range(M)]
        item_bands = [item_graph_bands[m] + item_rep_bands[m] for m in range(M)]

        # Fusion gates read the ORIGINAL pre-fusion representations (task_emb);
        # the IB surrogate is evaluated here over ALL users + ALL items.
        user_fused, ib_user = self.fusion_layer_user(user_bands, user_rep)
        item_fused, ib_item = self.fusion_layer_item(item_bands, item_rep)

        return {
            "user_rep": user_fused,
            "item_rep": item_fused,
            "user_bands": user_bands,
            "item_bands": item_bands,
            "ib_loss": ib_user + ib_item,
        }

    def forward(self):
        state = self._encode()
        return state["user_rep"], state["item_rep"]

    # ------------------------------------------------------- frequency CL loss
    def _frequency_contrastive_loss(self, user_bands, item_bands):
        """Official ``frequency_contrastive_loss`` on the batch band slices,
        users + items JOINTLY. For every ordered band pair ``m != n`` an MSE
        pulls ``cos(band_m, band_n)`` toward 1; for the upper-half bands
        (``m >= M // 2``) an InfoNCE term ``-mean(log_softmax(band @ band.T /
        tau))`` over the FULL logits matrix (RAW dot products, no L2
        normalisation). The whole sum is divided once by ``M * (M - 1)``.

        FAITHFUL QUIRK: at ``num_freq_bands: 1`` the divisor is 0 (division by
        zero), exactly as in the official code; unreachable via the HPO space
        ({2, 3, 4}), reachable only by a manual YAML edit."""
        M = self.num_freq_bands
        loss = torch.zeros((), device=self.device)
        for m in range(M):
            user_high = user_bands[m]
            item_high = item_bands[m]
            for n in range(M):
                if m != n:
                    user_sim = F.cosine_similarity(user_high, user_bands[n], dim=-1)
                    item_sim = F.cosine_similarity(item_high, item_bands[n], dim=-1)
                    loss = loss + F.mse_loss(user_sim, torch.ones_like(user_sim))
                    loss = loss + F.mse_loss(item_sim, torch.ones_like(item_sim))
            if m >= M // 2:
                user_contrast = F.log_softmax(
                    torch.mm(user_high, user_high.T) / self.cl_temperature, dim=-1
                )
                item_contrast = F.log_softmax(
                    torch.mm(item_high, item_high.T) / self.cl_temperature, dim=-1
                )
                loss = loss - torch.mean(user_contrast) - torch.mean(item_contrast)
        return loss / (M * (M - 1))

    # ---------------------------------------------------------------- training
    def calculate_loss(self, interaction):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        # Consume the dataloader's clean per-user history-avoiding negatives when
        # supplied (use_neg_sampling=true emits a 3-tuple); fall back to uniform
        # sampling only for the 2-tuple contract.
        if len(interaction) >= 3:
            neg_items = interaction[2].to(self.device)
        else:
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=self.device)

        state = self._encode()
        user_rep = state["user_rep"]
        item_rep = state["item_rep"]

        u = user_rep[users]
        pos_scores = torch.sum(u * item_rep[pos_items], dim=1)
        neg_scores = torch.sum(u * item_rep[neg_items], dim=1)

        # Main objective: BCE-with-logits (faithful; NOT BPR). Positives -> 1,
        # negatives -> 0.
        scores = torch.cat((pos_scores, neg_scores), dim=0)
        labels = torch.cat(
            (torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0
        )
        main_loss = F.binary_cross_entropy_with_logits(scores, labels)

        # IB gate-delta surrogate over ALL nodes, straight from the forward pass
        # (official computes ib_loss in forward, never on the batch slice).
        ib_loss = state["ib_loss"]

        # Frequency contrastive loss on the batch band slices: users + POSITIVE
        # items, jointly (official calculate_loss slicing).
        user_band_batch = [band[users] for band in state["user_bands"]]
        item_band_batch = [band[pos_items] for band in state["item_bands"]]
        cl_loss = self._frequency_contrastive_loss(user_band_batch, item_band_batch)

        return main_loss + self.ib_weight * ib_loss + self.cl_weight * cl_loss

    def full_sort_predict(self, interaction):
        """Full-sort scoring: full forward + dot product. Deterministic (no live
        dropout / random / SVD RNG)."""
        users = interaction[0].to(self.device)
        user_rep, item_rep = self.forward()
        u = user_rep[users]
        return torch.matmul(u, item_rep.transpose(0, 1))
