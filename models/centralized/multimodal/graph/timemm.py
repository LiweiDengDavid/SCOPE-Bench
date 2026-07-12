# coding: utf-8
r"""TimeMM -- Temporal multi-scale MultiModal recommendation
############################################################
Reference:
    Paper: "TimeMM: Time-aware Multi-scale MultiModal Recommendation", SIGIR'26.
    Official repo model class ``TIMEMM`` in ``src/models/timemm.py`` (a fork of
    the FITMM repo) plus the three helper builders
    ``time_edge_weight_builder_multiscale.py`` / ``time_user_gate.py`` /
    ``time_state_builder.py`` and the (disabled) aux losses
    ``multiscale_order_smoothness_loss.py`` / ``loss_multiscale_complement.py``.

TimeMM conditions a three-branch multimodal bipartite GCN on interaction time.
Everything time-derived is FROZEN at ``__init__`` from the TRAIN interactions
(faithful to the released code: no per-query reference time), in two forms:

  * **K recency-kernel edge-weight sets.** For each of the ``K = len(tau_age_list)``
    time scales, every training edge ``(u, i)`` is weighted by
    ``w_k = clip(exp(-log1p(age_sec) / tau_k), clip_min, clip_max)`` where
    ``age_sec = user_anchor(u) - t_ui`` and ``user_anchor(u)`` is the user's max
    train timestamp (``transform='inv'`` gives ``1/(1 + age/tau_k)`` instead).
    The bipartite convolution is then a WEIGHTED GCN normalised by weighted
    degrees ``w_ij / sqrt(deg_w[i] * deg_w[j])``.
  * **Per-node temporal features.** ``user_state`` [U,7] / ``item_state`` [I,7]
    (log1p count/span/gap/age stats; item age uses the GLOBAL max train time as
    anchor) feed a ``TimeScaleGate`` that mixes the K per-scale embeddings per
    node (softmax over scales, temperature 0.7); ``user_time_feat`` [U,4]
    (z-scored log1p mean/median age, span + recent-fraction) feeds a
    ``TimeAwareUserModalWeight`` gate that mixes the 3 modality user reps.

Each ``TemporalSpectralFilter`` branch projects its features (dv/dt -> 4*d -> d
for v/t; raw id), forms ``node_init = F.normalize(cat(user_preference, feats))``,
runs ``num_layers`` weighted-GCN convolutions per scale (layer-sum incl. the
input), stacks the K per-scale embeddings and gate-mixes them. The item view is
``cat(id, v, t)[items]`` plus a residual pass over the item-item multimodal kNN
graph ``mm_adj`` (official all-ones kNN Laplacian: uniform ``1/(knn_k + 1e-7)``
values on the cosine-topk support). The user view routes the 3 modality reps by
the user time gate, then ADDS the raw preference concat ``cat(id_pref, v_pref,
t_pref)`` as a residual (official ``propagate_user_item_graphs`` with no user
graph: ``user_emb_fused = user_embedding + fused_user_input``). Scoring is a
dot product.

**Main objective: BINARY CROSS-ENTROPY with logits ONLY.** This is faithful to
the SHIPPED RELEASE, where the auxiliary terms are computed-then-discarded (the
``calculate_loss`` return line adds only ``mf_loss``, with ``+ aux_loss * ... +
reg_term`` commented out). The auxiliary terms ARE implemented here and exposed
behind YAML weight gates that DEFAULT TO ZERO, so the out-of-the-box loss is
exactly the shipped BCE-only behavior:

  * ``aux_weight``  -- the official ``aux_loss`` = COMPLEMENT loss ONLY
    (off-diagonal squared correlation of the centered per-scale decision
    margins, with an optional variance floor);
  * ``order_smooth_weight`` -- multi-scale order-smoothness (hinge on per-scale
    user-pos L2 energies, ``E_k >= E_{k+1} + margin``). Official computes this
    term then DISCARDS it -- it is excluded even from the commented-out total
    (``mf_loss #+ aux_loss*aux_weight + reg_term``) -- hence the separate gate,
    deliberately NOT HPO-searched;
  * ``reg_weight1`` -- gate-entropy regularisation (mean entropy of the user
    modality gate);
  * ``reg_weight2`` -- an L2 term on the fused preferences + id embeddings.

Porting notes:

  * ``torch_geometric`` is stripped: the K weighted-normalised bipartite
    adjacencies are precomputed ONCE as sparse tensors (weighted degrees +
    ``torch.sparse.mm``); each branch is ``K x num_layers`` sparse mms -- cheap.
  * The framework Dataset drops the timestamp column (it loads only
    ``[userID, itemID, split_label]``), so TimeMM re-reads ``inter.csv``
    (``userID, itemID, timestamp, split_label``; train rows have
    ``split_label == 0``) from the dataset dir. When ``inter.csv`` is absent
    (tests / feature-only setups) it falls back to per-interaction timestamps
    supplied on the dataloader as ``dataloader.timestamps`` (aligned to the
    ``inter_matrix(form='coo')`` edge order); if neither is available all ages
    are 0 and the recency weights degrade to all-ones (plain multi-branch GCN).
  * Constant / missing timestamps degrade gracefully: ``age -> 0`` gives
    ``exp(0) = 1`` clipped to ``clip_max`` -- an unweighted GCN.
  * ``mm_adj`` is built once and cached to the dataset directory (the official
    guard disabled caching with ``and False``).
  * The time-stat builders run on the DEDUPLICATED per-(u,i) edge rows (latest
    train timestamp per pair); official aggregates over ALL train rows. Bit-
    identical on deduplicated data -- our ``.inter`` datasets are dedup -- and
    differs only under duplicate (u,i) events.
  * Official's ``aggr_mode`` knob is NOT ported (its non-"add" branch discards
    the recency edge weights and the shipped search space is the singleton
    ['add']); unknown ``transform`` values FAIL FAST instead of official's
    silent linear fallback; official's ``use_gap`` (default False, dead) is not
    ported.
  * Dead FITMM residue is NOT ported: ``result_embed``, the init-time edge
    dropout (``edge_index_dropv/t``, ``dropv/t_node_idx``), the ``user_emb.npy``
    user-graph crash path (``topk_sample`` / ``user_adj``), ``time_item_gate``,
    and the never-called ``DecisionSpaceAlignLoss``.
  * All experiment literals live in ``configs/models/TimeMM.yaml`` (dim_latent,
    knn_k, mm_image_weight, tau list, clip bounds, gate hidden widths, gate
    temperature, recent window, order-smoothness margin, aux/variance-floor
    weights). No Baby config ships upstream (only games); defaults follow the
    FITMM-family conventions.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase
from core.config import ConfigValidationError


class TimeScaleGate(nn.Module):
    """Per-node mixture weights over the K time scales (ref ``TimeScaleGate``).

    Shared-head design: a user stem and an item stem each map their temporal
    state through ``[Linear -> ReLU]`` to a common hidden width, then a SHARED
    head produces K logits; user and item logits are concatenated over nodes and
    softmax-normalised across scales at ``temperature``. Output ``gate`` is
    ``[N, K]`` with ``sum_k gate[n, k] = 1`` (``N = U + I``). Gate dropout is 0 at
    eval so scoring is deterministic.
    """

    def __init__(self, K, user_in, item_in, hidden, temperature):
        super().__init__()
        self.K = int(K)
        self.temperature = float(max(temperature, 1e-6))
        self.user_stem = nn.Sequential(nn.Linear(user_in, hidden), nn.ReLU())
        self.item_stem = nn.Sequential(nn.Linear(item_in, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, self.K)

    def forward(self, user_state, item_state):
        u_logits = self.head(self.user_stem(user_state))  # [U, K]
        i_logits = self.head(self.item_stem(item_state))  # [I, K]
        logits = torch.cat([u_logits, i_logits], dim=0)   # [N, K]
        return torch.softmax(logits / self.temperature, dim=1)


class TimeAwareUserModalWeight(nn.Module):
    """Per-user modality mixture weights from the user time features.

    MLP ``user_time_feat[U, F] -> hidden -> 3`` then softmax over the 3 modalities
    (ref ``TimeAwareUserModalWeight``). Returns ``[U, 3, 1]``.
    """

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, user_time_feat):
        w = torch.softmax(self.mlp(user_time_feat), dim=1)  # [U, 3]
        return w.unsqueeze(-1)                               # [U, 3, 1]


class TemporalSpectralFilter(nn.Module):
    """One id/visual/textual branch: multi-scale weighted-GCN + time-scale gate.

    ``node_init = F.normalize(cat(user_preference, projected_feats))``. For each of
    the K precomputed weighted-normalised adjacencies, run ``num_layers`` sparse
    convolutions and sum the layer outputs (incl. the input) -> one per-scale
    embedding ``[N, D]``; stack to ``[K, N, D]``. The ``TimeScaleGate`` produces
    ``[N, K]`` scale weights that mix the stack into the fused ``[N, D]`` view.

    ``dim_latent`` set -> features are projected ``dim_feat -> 4*dim_latent ->
    dim_latent`` (v/t branches); ``dim_latent`` None -> features are used raw and
    the id preference block is sized to the id embedding width (the id branch).
    """

    def __init__(self, num_user, num_item, num_scales, feat_dim, dim_latent,
                 num_layers, scale_gate_hidden, scale_gate_temperature,
                 u_state_dim, i_state_dim):
        super().__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.K = num_scales
        self.num_layers = num_layers
        self.dim_latent = dim_latent

        if dim_latent:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(num_user, dim_latent))
            )
            self.feature_mlp = nn.Linear(feat_dim, 4 * dim_latent)
            self.feature_mlp_out = nn.Linear(4 * dim_latent, dim_latent)
        else:
            # id branch: preference block matches the (raw) id feature width.
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(num_user, feat_dim))
            )

        self.time_scale_gate = TimeScaleGate(
            K=self.K,
            user_in=u_state_dim,
            item_in=i_state_dim,
            hidden=scale_gate_hidden,
            temperature=scale_gate_temperature,
        )

    def forward(self, features, scale_adjs, user_state, item_state):
        """``scale_adjs`` is a list of K sparse [N, N] weighted-normalised
        adjacencies. Returns ``(filter_emb [N, D], per_scale [K, N, D],
        preference [U, D])``."""
        if self.dim_latent:
            projected = self.feature_mlp_out(F.leaky_relu(self.feature_mlp(features)))
        else:
            projected = features

        node_init = torch.cat((self.preference, projected), dim=0)
        node_init = F.normalize(node_init)

        per_scale = []
        for adj in scale_adjs:
            layer_sum = node_init
            hidden = node_init
            for _ in range(self.num_layers):
                hidden = torch.sparse.mm(adj, hidden)
                layer_sum = layer_sum + hidden
            per_scale.append(layer_sum)
        per_scale = torch.stack(per_scale, dim=0)  # [K, N, D]

        gate = self.time_scale_gate(user_state, item_state)  # [N, K]
        gate_kn1 = gate.t().unsqueeze(-1)                     # [K, N, 1]
        filter_emb = (per_scale * gate_kn1).sum(dim=0)        # [N, D]
        return filter_emb, per_scale, self.preference


class TimeMM(RecommenderBase):
    def __init__(self, config, dataloader):
        super(TimeMM, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        # --- dimensions ---
        self.dim_latent = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.num_layers = config["num_layers"]
        self.n_mm_layers = config["n_mm_layers"]
        self.knn_k = config["knn_k"]
        self.mm_image_weight = float(config["mm_image_weight"])
        # NOTE: official's ``aggr_mode`` knob is NOT ported. Its only live branch
        # (GraphConvolutionNetwork.message, ref :798-810) degenerates for any
        # value != "add" into unweighted raw-message aggregation that DISCARDS
        # the recency edge weights, and the shipped TIMEMM.yaml pins the search
        # space to the singleton ['add'] -- so the non-add path is never
        # exercised. This port hardwires the "add" weighted-sym normalisation.

        # --- multi-scale recency kernels ---
        self.tau_age_list = list(config["tau_age_list"])
        self.num_scale = len(self.tau_age_list)
        self.use_age = bool(config["use_age"])
        self.transform = config["transform"]
        if self.transform not in ("exp", "inv"):
            # Official ``_factor`` silently falls back to a LINEAR kernel (t/tau,
            # its raise is commented out); our previous port silently fell back
            # to 'exp'. Neither silent fallback is acceptable: fail fast.
            raise ConfigValidationError(
                f"Unknown TimeMM recency transform '{self.transform}'; the "
                "official kernels are 'exp' and 'inv'."
            )
        self.clip_min = float(config["edge_weight_clip_min"])
        self.clip_max = float(config["edge_weight_clip_max"])
        self.recent_window_sec = int(config["recent_window_sec"])

        # --- gate widths ---
        self.scale_gate_hidden = int(config["scale_gate_hidden"])
        self.scale_gate_temperature = float(config["scale_gate_temperature"])
        self.user_modal_gate_hidden = int(config["user_modal_gate_hidden"])

        # --- auxiliary-loss weights (0 = shipped BCE-only release) ---
        self.aux_weight = float(config["aux_weight"])
        self.order_smooth_weight = float(config["order_smooth_weight"])
        self.reg_weight1 = float(config["reg_weight1"])
        self.reg_weight2 = float(config["reg_weight2"])
        self.order_smooth_margin = float(config["order_smooth_margin"])
        self.use_corr = bool(config["use_corr"])
        self.use_var_floor = bool(config["use_var_floor"])
        self.var_floor = float(config["var_floor"])
        self.var_weight = float(config["var_weight"])

        self.n_modalities = 3

        # --- U-I interactions (COO) ---
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)

        # --- id / preference embeddings ---
        self.user_id_embedding = nn.Embedding(self.n_users, self.dim_latent)
        self.item_id_embedding = nn.Embedding(self.n_items, self.dim_latent)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # --- modality feature embeddings (trainable, from pretrained feats) ---
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        # --- item-item multimodal kNN graph (cached) ---
        self.mm_adj = self._build_mm_adj().to(self.device)

        # --- per-interaction timestamps -> frozen time tensors ---
        # Aligned to the COO edge order of interaction_matrix.
        edge_users = self.interaction_matrix.row.astype(np.int64)
        edge_items = self.interaction_matrix.col.astype(np.int64)
        timestamps = self._load_edge_timestamps(dataloader, edge_users, edge_items)

        # K recency-kernel edge-weight sets [K, 2*nnz] + z-scored temporal states.
        self.register_buffer(
            "edge_weight",
            self._build_multiscale_edge_weights(edge_users, timestamps),
        )
        user_state, item_state = self._build_user_item_states(
            edge_users, edge_items, timestamps
        )
        self.register_buffer("user_state", user_state)
        self.register_buffer("item_state", item_state)
        self.register_buffer(
            "user_time_feat",
            self._build_user_time_features(edge_users, timestamps),
        )

        # K weighted-normalised sparse bipartite adjacencies (one per scale).
        self.scale_adjs = [
            adj.to(self.device)
            for adj in self._build_scale_adjacencies(edge_users, edge_items)
        ]

        # --- user-side modality routing gate ---
        self.time_user_gate = TimeAwareUserModalWeight(
            in_dim=self.user_time_feat.size(1), hidden=self.user_modal_gate_hidden
        )

        # --- three temporal-spectral branches (id / visual / textual) ---
        # Faithful to the reference's fixed 3-modality routing: ALL three branches
        # (and the 3-way user modality gate) always exist. A missing modality
        # (v_feat / t_feat is None) is zero-filled in _encode with a feat_embed_dim
        # feature block, exactly as the FITMM port handles it -- this keeps the id/
        # v/t slots fixed at 3 so the routing gate always broadcasts, and prevents
        # the reference's own crash on a single-modality dataset (it references
        # self.v_rep / self.t_rep unconditionally). Documented convention deviation.
        self._v_feat_dim = self.v_feat.shape[1] if self.v_feat is not None else self.feat_embed_dim
        self._t_feat_dim = self.t_feat.shape[1] if self.t_feat is not None else self.feat_embed_dim
        self.id_branch = TemporalSpectralFilter(
            num_user=self.n_users, num_item=self.n_items, num_scales=self.num_scale,
            feat_dim=self.dim_latent, dim_latent=None, num_layers=self.num_layers,
            scale_gate_hidden=self.scale_gate_hidden,
            scale_gate_temperature=self.scale_gate_temperature,
            u_state_dim=self.user_state.size(1), i_state_dim=self.item_state.size(1),
        )
        self.visual_branch = TemporalSpectralFilter(
            num_user=self.n_users, num_item=self.n_items, num_scales=self.num_scale,
            feat_dim=self._v_feat_dim, dim_latent=self.dim_latent,
            num_layers=self.num_layers, scale_gate_hidden=self.scale_gate_hidden,
            scale_gate_temperature=self.scale_gate_temperature,
            u_state_dim=self.user_state.size(1), i_state_dim=self.item_state.size(1),
        )
        self.textual_branch = TemporalSpectralFilter(
            num_user=self.n_users, num_item=self.n_items, num_scales=self.num_scale,
            feat_dim=self._t_feat_dim, dim_latent=self.dim_latent,
            num_layers=self.num_layers, scale_gate_hidden=self.scale_gate_hidden,
            scale_gate_temperature=self.scale_gate_temperature,
            u_state_dim=self.user_state.size(1), i_state_dim=self.item_state.size(1),
        )

    # ------------------------------------------------------------ timestamp shim
    def _load_edge_timestamps(self, dataloader, edge_users, edge_items):
        """Return a per-edge Unix-timestamp array aligned to the COO edge order.

        Production path: re-read ``inter.csv`` from the dataset dir (the framework
        Dataset drops the timestamp column, loading only
        ``[userID, itemID, split_label]``), keep TRAIN rows (``split_label == 0``),
        and map each COO edge to its LATEST train timestamp for that (user, item)
        pair. Test / feature-only path: fall back to ``dataloader.timestamps``
        (aligned to the COO edge order). If neither is available, ages are 0 and
        the recency weights degrade to all-ones (documented graceful degradation).
        """
        dataset_path = os.path.abspath(self.config["data_path"] + self.config["dataset"])
        inter_file = os.path.join(dataset_path, self.config["interaction_file"])
        uid_field = self.config["USER_ID_FIELD"]
        iid_field = self.config["ITEM_ID_FIELD"]
        time_col = self.config["TIME_FIELD"]
        split_col = self.config["inter_splitting_label"]

        if os.path.isfile(inter_file):
            df = pd.read_csv(
                inter_file,
                usecols=[uid_field, iid_field, time_col, split_col],
                sep=self.config["field_separator"],
            )
            df = df[df[split_col] == 0]
            # LATEST train timestamp per (user, item), matching the edge key.
            latest = (
                df.groupby([uid_field, iid_field])[time_col].max().reset_index()
            )
            key_to_ts = {
                (int(u), int(i)): int(t)
                for u, i, t in zip(
                    latest[uid_field].to_numpy(),
                    latest[iid_field].to_numpy(),
                    latest[time_col].to_numpy(),
                )
            }
            ts = np.array(
                [key_to_ts.get((int(u), int(i)), 0)
                 for u, i in zip(edge_users, edge_items)],
                dtype=np.int64,
            )
            return ts

        # Fallback shim: per-interaction timestamps injected on the dataloader.
        if hasattr(dataloader, "timestamps"):
            ts = np.asarray(dataloader.timestamps, dtype=np.int64)
            if ts.shape[0] != edge_users.shape[0]:
                raise ValueError(
                    f"dataloader.timestamps has {ts.shape[0]} entries but the "
                    f"interaction graph has {edge_users.shape[0]} edges; the "
                    "timestamp array must align to inter_matrix(form='coo') order."
                )
            return ts

        # No time signal available: all ages 0 -> unweighted GCN.
        return np.zeros(edge_users.shape[0], dtype=np.int64)

    # --------------------------------------------------- frozen time tensors
    def _recency_factor(self, age_sec):
        """Recency kernel for one scale set (numpy). ``exp`` -> ``exp(-log1p(age)/
        tau)`` (robust for seconds); ``inv`` -> ``1/(1 + age/tau)``; any other
        value raises (official's ``_factor`` silently falls back to a linear
        ``t/tau`` kernel -- replaced by fail-fast per repo rules). Official's
        ``use_gap`` flag (default False, dead: it only toggles a placeholder gap
        tensor that never feeds the weights) is not ported."""
        age = np.maximum(age_sec.astype(np.float32), 0.0)
        out = np.empty((self.num_scale, age.shape[0]), dtype=np.float32)
        for k, tau in enumerate(self.tau_age_list):
            tau = float(max(tau, 1e-8))
            if self.transform == "exp":
                out[k] = np.exp(-np.log1p(age) / tau)
            elif self.transform == "inv":
                out[k] = 1.0 / (1.0 + age / tau)
            else:
                raise ConfigValidationError(
                    f"Unknown TimeMM recency transform '{self.transform}'; the "
                    "official kernels are 'exp' and 'inv'."
                )
        return out

    def _edge_ages(self, edge_users, timestamps):
        """Per-edge age in seconds = user_anchor(u) - t_edge, clipped at 0
        (user_anchor = user's max train timestamp)."""
        anchor = np.zeros(self.n_users, dtype=np.int64)
        np.maximum.at(anchor, edge_users, timestamps.astype(np.int64))
        age = anchor[edge_users] - timestamps.astype(np.int64)
        return np.maximum(age, 0)

    def _build_multiscale_edge_weights(self, edge_users, timestamps):
        """K recency-kernel edge-weight sets, symmetrised to [K, 2*nnz]."""
        nnz = edge_users.shape[0]
        if not self.use_age:
            w = np.ones((self.num_scale, nnz), dtype=np.float32)
        else:
            age = self._edge_ages(edge_users, timestamps)
            w = self._recency_factor(age)
        w = np.clip(w, self.clip_min, self.clip_max).astype(np.float32)
        w_t = torch.tensor(w, dtype=torch.float32)
        return torch.cat([w_t, w_t], dim=1)  # symmetric: [K, 2*nnz]

    def _build_scale_adjacencies(self, edge_users, edge_items):
        """Precompute K weighted-normalised sparse bipartite adjacencies.

        For scale k the adjacency is symmetric over ``N = n_users + n_items`` nodes
        (item indices offset by ``n_users``) with edge value
        ``w_ij / sqrt(deg_w[i] * deg_w[j])`` -- the exact weighted-GCN norm of the
        reference (weighted degrees via scatter_add), materialised once so the
        branch convolution is a plain ``torch.sparse.mm``."""
        n = self.n_users + self.n_items
        rows = edge_users
        cols = edge_items + self.n_users
        # Symmetric edge lists (u->i and i->u), matching edge_weight's [K, 2*nnz].
        sym_rows = torch.tensor(np.concatenate([rows, cols]), dtype=torch.long)
        sym_cols = torch.tensor(np.concatenate([cols, rows]), dtype=torch.long)

        adjs = []
        for k in range(self.num_scale):
            w = self.edge_weight[k]  # [2*nnz]
            deg = torch.zeros(n, dtype=w.dtype)
            deg.scatter_add_(0, sym_rows, w)
            deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
            norm_w = deg_inv_sqrt[sym_rows] * w * deg_inv_sqrt[sym_cols]
            indices = torch.stack([sym_rows, sym_cols], dim=0)
            adj = torch.sparse_coo_tensor(indices, norm_w, (n, n)).coalesce()
            adjs.append(adj)
        return adjs

    def _build_user_time_features(self, edge_users, timestamps):
        """Per-user time features [U, 4] = z-scored (log1p mean age, log1p median
        age, log1p span, recent-fraction), anchored at each user's max train time
        (ref ``build_user_time_features_from_train_df``).

        DEVIATION (documented): computed over the DEDUPLICATED per-(u,i) edge
        rows (latest train timestamp per pair), whereas official aggregates over
        ALL train rows -- bit-identical on deduplicated data, which our
        ``.inter`` datasets are; differs only under duplicate (u,i) events."""
        df = pd.DataFrame({"u": edge_users, "t": timestamps.astype(np.int64)})
        gmax = df.groupby("u")["t"].transform("max")
        df["age"] = np.maximum((gmax - df["t"]).to_numpy(np.int64), 0).astype(np.float64)
        df["is_recent"] = (df["t"] >= (gmax - self.recent_window_sec)).astype(np.float64)

        g = df.groupby("u")
        mean_age = g["age"].mean()
        p50_age = g["age"].median()
        span = (g["t"].max() - g["t"].min()).clip(lower=0).astype(np.float64)
        recent_frac = g["is_recent"].mean()

        out = np.zeros((self.n_users, 4), dtype=np.float32)
        idx = mean_age.index.to_numpy(np.int64)
        out[idx, 0] = np.log1p(mean_age.to_numpy(np.float64)).astype(np.float32)
        out[idx, 1] = np.log1p(p50_age.to_numpy(np.float64)).astype(np.float32)
        out[idx, 2] = np.log1p(span.to_numpy(np.float64)).astype(np.float32)
        out[idx, 3] = recent_frac.to_numpy(np.float64).astype(np.float32)

        mean = out.mean(axis=0, keepdims=True)
        std = out.std(axis=0, keepdims=True) + 1e-6
        return torch.from_numpy((out - mean) / std)

    def _build_user_item_states(self, edge_users, edge_items, timestamps):
        """Per-user [U, 7] and per-item [I, 7] temporal states (log1p features).

        User features: count, span, mean gap, age-last (=0 by construction), mean
        age (user anchor), std gap, std age. Item features: count, #distinct users,
        span, age-last (global anchor), mean age (global anchor), std gap, std age.
        Item ages use the GLOBAL max train timestamp as anchor (ref
        ``build_user_item_states_from_train_df`` with ``include_std=True``).

        DEVIATION (documented): computed over the DEDUPLICATED per-(u,i) edge
        rows (latest train timestamp per pair), whereas official aggregates over
        ALL train rows -- bit-identical on deduplicated data, which our
        ``.inter`` datasets are; differs only under duplicate (u,i) events.
        """
        ts = timestamps.astype(np.int64)

        # ---- user side ----
        du = pd.DataFrame({"u": edge_users, "t": ts})
        du = du.sort_values(["u", "t"], kind="mergesort")
        du["gap"] = du.groupby("u")["t"].diff().fillna(0).clip(lower=0).astype(np.int64)
        umax = du.groupby("u")["t"].transform("max")
        du["age"] = np.maximum((umax - du["t"]).to_numpy(np.int64), 0)
        gu = du.groupby("u")
        u_feat = np.stack([
            gu.size().to_numpy(np.float64),                                    # count
            (gu["t"].max() - gu["t"].min()).to_numpy(np.float64),             # span
            gu["gap"].mean().to_numpy(np.float64),                            # mean gap
            np.zeros(gu.ngroups, dtype=np.float64),                           # age last = 0
            gu["age"].mean().to_numpy(np.float64),                            # mean age
            gu["gap"].std(ddof=0).fillna(0.0).to_numpy(np.float64),           # std gap
            gu["age"].std(ddof=0).fillna(0.0).to_numpy(np.float64),           # std age
        ], axis=1)
        user_state = np.zeros((self.n_users, 7), dtype=np.float32)
        user_state[gu.size().index.to_numpy(np.int64)] = np.log1p(
            np.maximum(u_feat, 0.0)
        ).astype(np.float32)

        # ---- item side (global anchor) ----
        global_anchor = int(ts.max()) if ts.shape[0] > 0 else 0
        di = pd.DataFrame({"u": edge_users, "i": edge_items, "t": ts})
        di = di.sort_values(["i", "t"], kind="mergesort")
        di["gap"] = di.groupby("i")["t"].diff().fillna(0).clip(lower=0).astype(np.int64)
        di["age"] = np.maximum(global_anchor - di["t"].to_numpy(np.int64), 0)
        gi = di.groupby("i")
        i_feat = np.stack([
            gi.size().to_numpy(np.float64),                                   # count
            gi["u"].nunique().to_numpy(np.float64),                           # #users
            (gi["t"].max() - gi["t"].min()).to_numpy(np.float64),            # span
            (global_anchor - gi["t"].max()).clip(lower=0).to_numpy(np.float64),  # age last
            gi["age"].mean().to_numpy(np.float64),                           # mean age
            gi["gap"].std(ddof=0).fillna(0.0).to_numpy(np.float64),          # std gap
            gi["age"].std(ddof=0).fillna(0.0).to_numpy(np.float64),          # std age
        ], axis=1)
        item_state = np.zeros((self.n_items, 7), dtype=np.float32)
        item_state[gi.size().index.to_numpy(np.int64)] = np.log1p(
            np.maximum(i_feat, 0.0)
        ).astype(np.float32)

        return torch.from_numpy(user_state), torch.from_numpy(item_state)

    # ------------------------------------------------------------------ graphs
    def _knn_normalized_laplacian(self, features):
        """Official item-item kNN Laplacian (ref ``get_knn_adj_mat`` +
        ``compute_normalized_laplacian``, :273-294), implemented locally: the
        kNN SUPPORT comes from the cosine-sim topk, but the adjacency VALUES are
        ALL-ONES (``torch.ones_like(indices[0])``, NOT the sim weights), row-sum
        symmetric-normalised. Every row holds exactly ``knn_k`` ones, so all
        values are the uniform ``1/(knn_k + 1e-7)`` and strictly non-negative."""
        context_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        indices0 = torch.arange(knn_ind.shape[0], device=features.device)
        indices0 = indices0.unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        adj = torch.sparse_coo_tensor(
            indices,
            torch.ones(indices.size(1), dtype=features.dtype, device=features.device),
            adj_size,
        )
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def _build_mm_adj(self):
        """Item-item multimodal kNN Laplacian (ref mm_adj; cache re-enabled).

        Blends the per-modality OFFICIAL all-ones kNN Laplacians (uniform
        ``1/(knn_k + 1e-7)`` values on the cosine-topk support, see
        ``_knn_normalized_laplacian``) by ``mm_image_weight`` and caches to the
        dataset directory as ``timemm_mm_adj_ones_{knn_k}.pt`` (the official
        guard disabled caching with ``and False``; the ``_ones`` cache name
        invalidates caches from earlier builds that stored sim-weighted
        values)."""
        dataset_path = os.path.abspath(self.config["data_path"] + self.config["dataset"])
        os.makedirs(dataset_path, exist_ok=True)
        cache_file = os.path.join(dataset_path, f"timemm_mm_adj_ones_{self.knn_k}.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file, map_location="cpu", weights_only=False)

        image_adj = text_adj = None
        if self.v_feat is not None:
            image_adj = self._knn_normalized_laplacian(self.image_embedding.weight.detach())
        if self.t_feat is not None:
            text_adj = self._knn_normalized_laplacian(self.text_embedding.weight.detach())

        if image_adj is not None and text_adj is not None:
            mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        elif image_adj is not None:
            mm_adj = image_adj
        else:
            mm_adj = text_adj
        mm_adj = mm_adj.coalesce()
        torch.save(mm_adj, cache_file)
        return mm_adj

    # ----------------------------------------------------------------- forward
    def _encode(self):
        """Encode users/items into fused representations + per-scale slices.

        User view (ref ``forward`` :307-367 + ``propagate_user_item_graphs``
        :425-453): the time-gated routed concat of the 3 modality user reps
        PLUS the raw preference concat ``fused_user_input = cat(id_pref,
        v_pref, t_pref)`` added as a residual -- with no user graph, official's
        ``propagated_user_emb`` IS ``fused_user_input``, and ``user_emb_fused =
        user_embedding + propagated_user_emb``. Item view: ``cat(id, v, t)
        [items]`` plus its mm_adj propagation residual.

        Returns a dict with the user/item views for scoring, the stacked
        per-scale (id/v/t concatenated) node representations for the aux losses,
        the user modality gate weights, and the fused preference block (also the
        L2-reg target, official ``all_preference``)."""
        # ID branch (raw id embeddings as features).
        id_fused, id_scale, id_pref = self.id_branch(
            self.item_id_embedding.weight, self.scale_adjs,
            self.user_state, self.item_state,
        )

        # Visual branch (zero-filled when v_feat is absent -> fixed 3-modality slot).
        if self.v_feat is not None:
            v_features = self.image_embedding.weight
        else:
            v_features = torch.zeros(self.n_items, self._v_feat_dim, device=self.device)
        v_fused, v_scale, v_pref = self.visual_branch(
            v_features, self.scale_adjs, self.user_state, self.item_state,
        )

        # Textual branch (zero-filled when t_feat is absent).
        if self.t_feat is not None:
            t_features = self.text_embedding.weight
        else:
            t_features = torch.zeros(self.n_items, self._t_feat_dim, device=self.device)
        t_fused, t_scale, t_pref = self.textual_branch(
            t_features, self.scale_adjs, self.user_state, self.item_state,
        )

        scale_reps = [id_scale, v_scale, t_scale]
        modality_pref = [id_pref, v_pref, t_pref]
        item_views = [id_fused[self.n_users:], v_fused[self.n_users:], t_fused[self.n_users:]]
        user_views = [id_fused[: self.n_users], v_fused[: self.n_users], t_fused[: self.n_users]]

        # Item view = concat(id, v, t)[items].
        item_rep = torch.cat(item_views, dim=1)

        # User view = time-gated mix of the modality user reps, concatenated.
        gate_weights = self.time_user_gate(self.user_time_feat)  # [U, M, 1]
        user_stack = torch.stack(user_views, dim=2)              # [U, D, M]
        routed = gate_weights.transpose(1, 2) * user_stack       # [U, D, M]
        user_rep = torch.cat([routed[:, :, m] for m in range(routed.size(2))], dim=1)

        # Residual: propagate the item view over the item-item mm graph.
        propagated = item_rep
        for _ in range(self.n_mm_layers):
            propagated = torch.sparse.mm(self.mm_adj, propagated)
        item_rep = item_rep + propagated

        # Preference residual (ref propagate_user_item_graphs :438-451): the raw
        # preference concat is the user-side "propagated" input (no user graph)
        # and is ADDED to the routed user rep: user_emb_fused = routed + prefs.
        fused_pref = torch.cat(modality_pref, dim=1)  # [U, sum(dims)]
        user_rep = user_rep + fused_pref

        # Per-scale node reps concatenated across modalities: [K, N, sum(dims)].
        scale_rep = torch.cat(scale_reps, dim=-1)
        return {
            "user_rep": user_rep,
            "item_rep": item_rep,
            "scale_rep": scale_rep,
            "gate_weights": gate_weights,
            "fused_pref": fused_pref,
        }

    def forward(self):
        state = self._encode()
        return state["user_rep"], state["item_rep"]

    # ------------------------------------------------------------- BCE main loss
    def _bce_from_state(self, state, users, pos_items, neg_items):
        user_rep = state["user_rep"]
        item_rep = state["item_rep"]
        u = user_rep[users]
        pos_scores = torch.sum(u * item_rep[pos_items], dim=1)
        neg_scores = torch.sum(u * item_rep[neg_items], dim=1)
        scores = torch.cat((pos_scores, neg_scores), dim=0)
        labels = torch.cat(
            (torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0
        )
        return F.binary_cross_entropy_with_logits(scores, labels)

    def bce_main_loss_only(self, interaction):
        """The shipped-release objective: BCE-with-logits main term ONLY.

        Exposed so tests can pin that the DEFAULT ``calculate_loss`` (all aux
        weights 0) equals exactly this pure BCE loss."""
        users, pos_items, neg_items = self._parse_interaction(interaction)
        state = self._encode()
        return self._bce_from_state(state, users, pos_items, neg_items)

    # ------------------------------------------------------------- aux losses
    def _order_smoothness_loss(self, u_scales, pos_scales):
        """Multi-scale order-smoothness hinge (ref ``MultiScaleOrderSmoothnessLoss``,
        short->long, normalize_by_dim, reduction='sum'): enforce
        ``E_k >= E_{k+1} + margin`` on per-scale user-pos L2 energies.

        In the official release this term is computed-then-DISCARDED: it is
        excluded even from the commented-out total (``mf_loss #+ aux_loss *
        aux_weight + reg_term``, where ``aux_loss`` is the complement loss
        only). It is therefore gated by its own ``order_smooth_weight``
        (default 0, deliberately NOT HPO-searched), never by ``aux_weight``."""
        energies = []
        for uk, ik in zip(u_scales, pos_scales):
            diff = uk - ik
            e = (diff * diff).sum(dim=1) / float(uk.size(1))
            energies.append(e.mean())
        E = torch.stack(energies, dim=0)
        violations = F.relu(E[1:] + self.order_smooth_margin - E[:-1])
        return violations.sum()

    def _complement_loss(self, u_scales, pos_scales, neg_scales):
        """Multi-scale complement loss (ref ``MultiScaleComplementLoss``):
        off-diagonal squared (correlation | covariance) of the centered per-scale
        decision margins, plus an optional variance floor."""
        U = torch.stack(u_scales, dim=0)      # [K, B, D]
        P = torch.stack(pos_scales, dim=0)
        N = torch.stack(neg_scales, dim=0)
        K, B, _ = U.shape
        if K <= 1:
            return U.new_zeros(())
        M = (U * P).sum(dim=-1) - (U * N).sum(dim=-1)  # [K, B]
        M = M - M.mean(dim=1, keepdim=True)
        cov = (M @ M.transpose(0, 1)) / max(B, 1)      # [K, K]
        if self.use_corr:
            std = torch.sqrt(torch.diag(cov).clamp_min(1e-8))
            denom = (std[:, None] * std[None, :]).clamp_min(1e-8)
            C = cov / denom
        else:
            C = cov
        off = C - torch.diag(torch.diag(C))
        loss = (off ** 2).mean()
        if self.use_var_floor:
            std_m = torch.sqrt((M * M).mean(dim=1) + 1e-8)
            loss = loss + self.var_weight * F.relu(self.var_floor - std_m).mean()
        return loss

    def _gate_entropy_reg(self, gate_weights):
        """Mean entropy of the user modality gate (ref ``regularization`` gate
        term). ``gate_weights`` is ``[U, M, 1]``."""
        gate_prob = gate_weights.squeeze(-1).clamp_min(1e-12)  # [U, M]
        entropy = -(gate_prob * gate_prob.log()).sum(dim=1).mean()
        return entropy

    def _l2_reg(self, fused_pref):
        """L2 term on fused preferences + id embeddings (ref ``regularization``
        L2 term)."""
        return (
            fused_pref.pow(2).mean()
            + self.user_id_embedding.weight.pow(2).mean()
            + self.item_id_embedding.weight.pow(2).mean()
        )

    # ---------------------------------------------------------------- training
    def _parse_interaction(self, interaction):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        # Consume the dataloader's clean per-user history-avoiding negatives when
        # supplied (3-tuple); fall back to uniform sampling for the 2-tuple contract.
        if len(interaction) >= 3:
            neg_items = interaction[2].to(self.device)
        else:
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=self.device)
        return users, pos_items, neg_items

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = self._parse_interaction(interaction)
        state = self._encode()

        # Main objective: BCE-with-logits (faithful; NOT BPR). This ALONE is the
        # shipped-release loss; the aux terms below are gated to 0 by default.
        main_loss = self._bce_from_state(state, users, pos_items, neg_items)

        # Auxiliary terms are computed only when their YAML weight gate is nonzero
        # (the shipped release has them commented out -> all weights default 0).
        # aux_weight gates the COMPLEMENT loss ONLY -- official's aux_loss
        # (:410, comp_loss) excludes the order-smoothness term even from the
        # commented-out total; smoothness has its own order_smooth_weight gate.
        aux_total = main_loss.new_zeros(())

        if self.aux_weight != 0.0 or self.order_smooth_weight != 0.0:
            scale_rep = state["scale_rep"]  # [K, N, D]
            u_scales = [scale_rep[k, :self.n_users][users] for k in range(self.num_scale)]
            pos_scales = [scale_rep[k, self.n_users:][pos_items] for k in range(self.num_scale)]

        if self.aux_weight != 0.0:
            neg_scales = [scale_rep[k, self.n_users:][neg_items] for k in range(self.num_scale)]
            aux_total = aux_total + self.aux_weight * self._complement_loss(
                u_scales, pos_scales, neg_scales
            )

        if self.order_smooth_weight != 0.0:
            aux_total = aux_total + self.order_smooth_weight * self._order_smoothness_loss(
                u_scales, pos_scales
            )

        if self.reg_weight1 != 0.0:
            aux_total = aux_total + self.reg_weight1 * self._gate_entropy_reg(
                state["gate_weights"]
            )
        if self.reg_weight2 != 0.0:
            aux_total = aux_total + self.reg_weight2 * self._l2_reg(state["fused_pref"])

        return main_loss + aux_total

    def full_sort_predict(self, interaction):
        """Full-sort scoring: full forward + dot product. Deterministic (gate
        dropout is off, no sampling at eval)."""
        users = interaction[0].to(self.device)
        user_rep, item_rep = self.forward()
        u = user_rep[users]
        return torch.matmul(u, item_rep.transpose(0, 1))
