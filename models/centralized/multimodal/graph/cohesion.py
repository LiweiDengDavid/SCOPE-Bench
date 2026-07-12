# coding: utf-8
r"""
COHESION -- Co-optimized Heterogeneous Fusion for Multimodal Recommendation
###########################################################################
Reference:
    Paper: SIGIR'2025 COHESION (co-optimized dual-stage heterogeneous fusion).
    Official code: ``src/models/cohesion.py`` (EnochE-style MMRec framework).

COHESION fuses id and multimodal (visual / textual) signals in two stages:

  * **Early stage.** Three ``GCNLayer`` branches (``id_gcn`` / ``v_gcn`` /
    ``t_gcn``) run over a shared symmetric-normalized bipartite user-item graph.
    Each branch projects its raw feature (``MLP -> leaky_relu -> MLP_1``), blends
    it with the id embedding via an RMS mix
    ``sqrt(|(id^2 + feat^2)/2| + 1e-8)``, prepends a learnable per-user
    ``preference`` block, L2-normalizes, and runs ``num_layer`` layer-refined
    propagation rounds (each ``adj @ emb`` reweighted by the cosine similarity to
    the ego embedding), summing the layers. The three branch outputs are
    concatenated into a per-node ``[n_nodes, 3 * dim]`` representation.
  * **Late stage.** Items are further propagated ``n_mm_layers`` times over a
    per-modality cosine-kNN item-item graph (``mm_adj``); users are refined by a
    weighted sum over their top-``uu_topk`` user-user neighbours (shared-item
    co-occurrence). The two blocks are added back to the early representation.

Training scores are re-weighted by an **adaptive modality balancer**
(``1 - softmax(pos - neg)`` per 64-d modality block, detached); the objective is
a ``log2``-BPR plus a squared-norm regulariser on the user preferences and the
per-user modality weights (``weight_u``).

Faithful-port notes / documented deviations from the official code:

  * **User-user graph derived in-model.** The official model loads a
    ``user_graph_dict.npy`` produced by an offline O(n_users^2) Python double loop
    (``preprocessing/dualgnn-gen-u-u-matrix.py``: TRAIN-only shared-item counts,
    per-user top-``uu_graph_top_k``, 200 officially). The counts are reproduced
    identically, vectorised as a sparse ``B @ B^T`` (zero diagonal); the
    per-user neighbour selection then follows the official semantics exactly —
    ``torch.topk(row, min(nnz, top_k))`` on the dense count row (torch.topk
    does NOT tie-break by low index, and co-occurrence counts are tie-heavy, so
    a lexicographic sort would select different neighbour SETS). Residual tie
    order is torch-version-defined. Cached to ``cohesion_uu_topk{k}.pt`` in the
    dataset dir.
  * **mm_adj cache keyed by every searched input.** The official
    ``mm_adj_{knn_k}.pt`` cache omits ``mm_image_weight`` (fixed per dataset
    there, HPO-searched here); the port keys the cache by ``knn_k`` AND the
    blend weight (``mm_adj_cache_filename``) so trials never silently reuse
    another trial's blend.
  * **Deterministic user-graph padding.** ``topk_sample`` pads users with fewer
    than ``k`` neighbours by *cycling* their neighbour list (the official code
    used ``np.random.randint`` duplicates). This makes both training draws and
    evaluation reproducible.
  * **Interaction-free deterministic eval.** The official ``full_sort_predict``
    sliced a stale ``result_embed`` left over from the last training forward.
    We recompute the representation + late fusion inside ``full_sort_predict``
    (both stages are interaction-free) and, matching the official train/eval
    asymmetry, do **not** apply the adaptive modality weight at eval.
  * **Canonical dropout key.** Edge dropout reads ``dropout_rate`` (default 0 =
    identity), replacing the official ``dropout`` key.
  * Unused ``torch_geometric`` / ``torch_sparse`` imports and the dead
    ``BGCNLayer`` (an id-only clone of ``GCNLayer``) are dropped.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase
from core.utils import build_norm_adj_matrix

# Rows of the sparse co-occurrence matrix are densified in fixed-size chunks
# before running torch.topk, purely to bound peak memory on large user counts;
# the chunk size cannot affect the selected neighbours.
_UU_TOPK_CHUNK_ROWS = 2048


def mm_adj_cache_filename(knn_k, mm_image_weight):
    """Cache key for the blended mm graph: every searched input that determines it.

    ``mm_image_weight`` is HPO-searched (COHESION.yaml), so it must be part of
    the key — the official ``mm_adj_{knn_k}.pt`` name (weight fixed per dataset
    there) would make later trials silently reuse trial 1's blend. The weight
    is encoded via str(float), a lossless roundtrip repr (mirrors
    DRAGON/FREEDOM); the ``cohesion_`` prefix is a new pattern so stale
    weight-less caches and sibling models' caches can never collide.
    """
    return "cohesion_mm_adj_k{}_w{}.pt".format(knn_k, float(mm_image_weight))


class COHESION(RecommenderBase):
    def __init__(self, config, dataloader):
        super(COHESION, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.n_nodes = self.n_users + self.n_items
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.knn_k = config["knn_k"]
        self.mm_image_weight = config["mm_image_weight"]
        self.edge_dropout = float(config["dropout_rate"])
        self.uu_topk = config["uu_topk"]
        self.num_layer = config["num_layer"]
        self.reg_weight = float(config["reg_weight"])
        # Latent width of each modality block (id / v / t branches); the
        # concatenated representation is 3 * dim_latent. The official model
        # hardcodes ``dim_latent = 64`` independently of ``embedding_size``
        # (src/models/cohesion.py), so this is a plain YAML key, not searched.
        self.dim_latent = config["dim_latent"]
        # Per-user u-u neighbour-pool size stored in the user graph dict; the
        # official preprocessing hardcodes top-200 (dualgnn-gen-u-u-matrix.py).
        # Plain YAML key, not searched.
        self.uu_graph_top_k = config["uu_graph_top_k"]

        self.v_rep, self.t_rep, self.id_rep = None, None, None
        self.v_preference, self.t_preference, self.id_preference = None, None, None

        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        os.makedirs(dataset_path, exist_ok=True)

        # Train interactions (bipartite user-item edges).
        self.train_interactions = dataloader.inter_matrix(form="coo").astype(np.float32)

        # User-user graph, derived from train-only shared-item co-occurrence and
        # cached (replaces the official offline O(n^2) loop; identical output).
        self.user_graph_dict = self._build_user_graph_dict(dataset_path)

        # Item-item modality graph (mixed visual/text cosine-kNN Laplacian).
        self.mm_adj = self._build_mm_adj(dataset_path)

        # Feature embeddings (trainable copies of the raw features).
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        # Per-user modality weights, kept only for the regularisation term
        # (faithful to the official ``weight_u`` [n_users, 2, 1]).
        weight_u = torch.tensor(
            np.random.randn(self.n_users, 2, 1), dtype=torch.float32
        )
        nn.init.xavier_normal_(weight_u)
        self.weight_u = nn.Parameter(F.softmax(weight_u, dim=1))

        # Normalized bipartite adjacency + edge info for optional edge dropout.
        self.edge_indices, self.edge_values = self._get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)
        self.norm_adj = self._get_norm_adj_mat().to(self.device)
        self.masked_adj = self.norm_adj

        # Three GCN branches (id / visual / textual).
        self.id_feat = nn.Parameter(
            self._xavier_param((self.n_items, self.dim_latent))
        )
        self.id_gcn = GCNLayer(
            self.n_users, self.n_items, self.num_layer, self.dim_latent,
            self.device, self.id_feat,
        )
        if self.v_feat is not None:
            self.v_gcn = GCNLayer(
                self.n_users, self.n_items, self.num_layer, self.dim_latent,
                self.device, self.v_feat,
            )
        if self.t_feat is not None:
            self.t_gcn = GCNLayer(
                self.n_users, self.n_items, self.num_layer, self.dim_latent,
                self.device, self.t_feat,
            )

        self.user_graph = _UserGraphSample()

        # Deterministic user-graph sample used at eval (and the initial training
        # sample); padding cycles neighbours so it is reproducible.
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.uu_topk)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def _xavier_param(self, shape):
        tensor = torch.tensor(np.random.randn(*shape), dtype=torch.float32)
        nn.init.xavier_normal_(tensor, gain=1)
        return tensor.to(self.device)

    def _build_user_graph_dict(self, dataset_path):
        """Derive the per-user top-k shared-item co-occurrence neighbours.

        Counts follow ``preprocessing/dualgnn-gen-u-u-matrix.py`` exactly:
        train-only user-user shared-item counts, vectorised as sparse
        ``B @ B^T`` (B = binary user-item matrix) with a zeroed diagonal —
        verified identical to the official O(n^2) double loop. Selection also
        follows the official semantics: per-user
        ``torch.topk(row, min(nnz, top_k))`` on the dense float32 count row
        (the official script's exact call). ``torch.topk`` does NOT tie-break
        by low index, and co-occurrence counts are tie-heavy, so a
        lexicographic ``(-count, id)`` sort would select different neighbour
        SETS; residual tie order among equal counts is torch-version-defined
        (same torch build => identical output). Rows are densified in chunks
        to bound memory. Cached to ``cohesion_uu_topk{top_k}.pt`` — a new name
        so stale caches built with the old lexicographic rule are never reused.
        """
        top_k = self.uu_graph_top_k
        cache_file = os.path.join(dataset_path, f"cohesion_uu_topk{top_k}.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file, weights_only=False)

        # Binary user-item matrix; co-occurrence counts = B @ B^T (CSR so row
        # blocks slice cheaply).
        binary = (self.train_interactions.tocsr() > 0).astype(np.float32)
        cooc = (binary @ binary.transpose()).tocsr()

        user_graph_dict = {}
        for start in range(0, self.n_users, _UU_TOPK_CHUNK_ROWS):
            end = min(start + _UU_TOPK_CHUNK_ROWS, self.n_users)
            chunk = torch.from_numpy(cooc[start:end].toarray())
            for offset in range(end - start):
                u = start + offset
                row = chunk[offset]
                row[u] = 0.0  # official double loop never fills the diagonal
                n_neighbours = int(torch.count_nonzero(row).item())
                picked_values, picked_indices = torch.topk(
                    row, min(n_neighbours, top_k)
                )
                user_graph_dict[u] = [
                    picked_indices.tolist(), picked_values.tolist()
                ]

        torch.save(user_graph_dict, cache_file)
        return user_graph_dict

    def _build_mm_adj(self, dataset_path):
        # Keyed by knn_k AND the HPO-searched mm_image_weight (see
        # mm_adj_cache_filename) so trials never reuse another trial's blend.
        mm_adj_file = os.path.join(
            dataset_path, mm_adj_cache_filename(self.knn_k, self.mm_image_weight)
        )
        if os.path.exists(mm_adj_file):
            return torch.load(mm_adj_file, weights_only=False).to(self.device)

        image_adj, text_adj = None, None
        if self.v_feat is not None:
            _, image_adj = self._get_knn_adj_mat(self.v_feat)
        if self.t_feat is not None:
            _, text_adj = self._get_knn_adj_mat(self.t_feat)

        if image_adj is not None and text_adj is not None:
            mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        else:
            mm_adj = image_adj if image_adj is not None else text_adj

        torch.save(mm_adj.cpu(), mm_adj_file)
        return mm_adj.to(self.device)

    def _get_knn_adj_mat(self, features):
        features = features.to(self.device)
        context_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        knn_k = min(self.knn_k, sim.shape[1])
        _, knn_ind = torch.topk(sim, knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1)
        indices0 = indices0.expand(-1, knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self._compute_normalized_laplacian(indices, adj_size)

    def _compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(
            indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size
        )
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def _get_edge_info(self):
        rows = torch.from_numpy(self.train_interactions.row)
        cols = torch.from_numpy(self.train_interactions.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(
            indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size
        )
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        return rows_inv_sqrt * cols_inv_sqrt

    def _get_norm_adj_mat(self):
        # Symmetric D^{-1/2} A D^{-1/2} over the bipartite user-item graph
        # (users in the upper block, items offset by n_users), matching the
        # official ``get_norm_adj_mat`` (1e-7 degree epsilon).
        return build_norm_adj_matrix(
            self.train_interactions, self.n_users, self.n_items
        )

    # ------------------------------------------------------------------ #
    # Per-epoch hook
    # ------------------------------------------------------------------ #
    def pre_epoch_processing(self):
        # Re-sample the (deterministic) user-graph for the training late-fusion.
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.uu_topk)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)
        if self.edge_dropout <= 0.0:
            self.masked_adj = self.norm_adj
            return
        degree_len = int(self.edge_values.size(0) * (1.0 - self.edge_dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        keep_indices = self.edge_indices[:, degree_idx]
        keep_values = self._normalize_adj_m(
            keep_indices, torch.Size((self.n_users, self.n_items))
        )
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse_coo_tensor(
            all_indices, all_values, self.norm_adj.shape
        ).to(self.device)

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #
    def _build_representation(self, adj):
        id_rep, self.id_preference = self.id_gcn(self.id_feat, self.id_feat, adj)
        # Faithful to the official code: the id branch enters the representation
        # DETACHED (``id_rep.data``), so its GCN MLP/preference receive no
        # gradient from the ranking loss — only the v/t branches do.
        id_rep_data = id_rep.detach()

        reps = [id_rep_data]
        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.v_feat, self.id_feat, adj)
            reps.append(self.v_rep)
        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.t_feat, self.id_feat, adj)
            reps.append(self.t_rep)

        representation = torch.cat(reps, dim=1)
        return representation

    def _process_user_item_representation(self, representation, user_graph, user_weight):
        user_rep = representation[: self.n_users]
        item_rep = representation[self.n_users:]

        h_i = item_rep
        for _ in range(self.n_layers):
            h_i = torch.sparse.mm(self.mm_adj, h_i)
        h_u = self.user_graph(user_rep, user_graph, user_weight)

        user_rep = user_rep + h_u
        item_rep = item_rep + h_i
        return user_rep, item_rep

    def _adaptive_optimization(self, user_e, pos_e, neg_e):
        n_mod = user_e.shape[1] // self.dim_latent
        pos_score_ = torch.mul(user_e, pos_e).view(-1, n_mod, self.dim_latent).sum(dim=-1)
        neg_score_ = torch.mul(user_e, neg_e).view(-1, n_mod, self.dim_latent).sum(dim=-1)
        modality_indicator = 1 - (pos_score_ - neg_score_).softmax(-1).detach()
        adaptive_weight = torch.tile(
            modality_indicator.view(-1, n_mod, 1), [1, 1, self.dim_latent]
        )
        return adaptive_weight.view(-1, n_mod * self.dim_latent)

    def forward(self, interaction):
        user_nodes = interaction[0]
        # Clone to avoid mutating the caller's tensors when offsetting item ids.
        pos_item_nodes = interaction[1] + self.n_users
        neg_item_nodes = interaction[2] + self.n_users

        representation = self._build_representation(self.masked_adj)
        user_rep, item_rep = self._process_user_item_representation(
            representation, self.epoch_user_graph, self.user_weight_matrix
        )
        result_embed = torch.cat((user_rep, item_rep), dim=0)

        user_tensor = result_embed[user_nodes]
        pos_item_tensor = result_embed[pos_item_nodes]
        neg_item_tensor = result_embed[neg_item_nodes]

        adaptive_weight = self._adaptive_optimization(
            user_tensor, pos_item_tensor, neg_item_tensor
        )
        pos_scores = torch.sum(user_tensor * pos_item_tensor * adaptive_weight, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor * adaptive_weight, dim=1)
        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        user = interaction[0]
        if len(interaction) >= 3:
            pos = interaction[1]
            neg = interaction[2]
        else:
            pos = interaction[1]
            neg = torch.randint(0, self.n_items, pos.shape, device=pos.device)

        pos_scores, neg_scores = self.forward((user, pos, neg))
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        reg_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_v + reg_t)
        reg_loss = reg_loss + self.reg_weight * (self.weight_u ** 2).mean()
        return loss_value + reg_loss

    def full_sort_predict(self, interaction):
        # Recompute the representation + late fusion (interaction-free) so eval is
        # deterministic and self-contained. No adaptive modality weight at eval,
        # matching the official train/eval asymmetry. Use the unmasked adjacency
        # and the deterministic user-graph sample built at init.
        representation = self._build_representation(self.norm_adj)
        user_rep, item_rep = self._process_user_item_representation(
            representation, self.epoch_user_graph, self.user_weight_matrix
        )
        temp_user_tensor = user_rep[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_rep.t())

    def topk_sample(self, k):
        """Per-user top-k user-graph neighbours with softmax(count) weights.

        Users with fewer than ``k`` neighbours are padded deterministically by
        cycling their neighbour list (no ``np.random.randint``); users with zero
        neighbours get a zero-weight self-referential row (index 0).
        """
        user_graph_index = []
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        zero_row = [0] * k
        for i in range(len(self.user_graph_dict)):
            neighbours = self.user_graph_dict[i][0]
            counts = self.user_graph_dict[i][1]
            if len(neighbours) == 0:
                user_graph_index.append(list(zero_row))
                continue
            if len(neighbours) < k:
                sample = list(neighbours[:k])
                weight = list(counts[:k])
                j = 0
                while len(sample) < k:
                    # Deterministic cycle over the existing neighbours.
                    sample.append(neighbours[j % len(neighbours)])
                    weight.append(counts[j % len(counts)])
                    j += 1
                user_graph_index.append(sample)
                user_weight_matrix[i] = F.softmax(torch.tensor(weight), dim=0)
                continue
            sample = list(neighbours[:k])
            weight = list(counts[:k])
            user_weight_matrix[i] = F.softmax(torch.tensor(weight), dim=0)
            user_graph_index.append(sample)
        return user_graph_index, user_weight_matrix


class _UserGraphSample(nn.Module):
    """Weighted sum over each user's sampled user-user neighbours."""

    def forward(self, features, user_graph, user_matrix):
        # user_graph: [n_users, k] neighbour indices -> gather [n_users, k, dim].
        index = torch.as_tensor(user_graph, dtype=torch.long, device=features.device)
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        return u_pre.squeeze(1)


class GCNLayer(nn.Module):
    """Layer-refined GCN branch with early-stage id/feature RMS fusion."""

    def __init__(self, num_user, num_item, num_layer, dim_latent, device, features):
        super(GCNLayer, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.num_layer = num_layer
        self.device = device
        preference = torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32
        )
        nn.init.xavier_normal_(preference, gain=1)
        self.preference = nn.Parameter(preference.to(device))
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, features, id_embd, adj):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))
        # Early-stage RMS blend of id and projected feature embeddings.
        temp_features = torch.abs(
            ((torch.mul(id_embd, id_embd) + torch.mul(temp_features, temp_features)) / 2) + 1e-8
        ).sqrt()
        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)
        ego_embeddings = x
        all_embeddings = ego_embeddings
        embeddings_layers = [all_embeddings]

        for _ in range(self.num_layer):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum("a,ab->ab", _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)
        return ui_all_embeddings, self.preference
