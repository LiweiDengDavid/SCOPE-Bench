# coding: utf-8
r"""
REARM
################################################
Reference:
    https://dl.acm.org/doi/10.1145/3746027.3755779
    ACM MM'2025: Shouxing Ma, Yawen Zeng, Shiqing Wu, Guandong Xu.
    "Refining Contrastive Learning and Homography Relations for
     Multi-Modal Recommendation."

Faithful self-contained port of the official ``model.py`` (+ ``utils/helper.py``
graph builders). REARM refines recommendation-relevant modal-shared and
modal-unique information through a meta-network low-rank transform and an
orthogonal constraint respectively, and jointly incorporates co-occurrence and
similarity (interest) graphs of users and items.

Design decisions binding this port:

  * The framework ships raw ``v_feat``/``t_feat`` + the train interaction matrix,
    not REARM's separately-processed graph artifacts. Every graph is therefore
    rebuilt in ``__init__`` from raw features + interactions and cached to the
    dataset dir:
      - user/item co-occurrence = sparse ``B @ Bᵀ`` (train-only), diagonal
        zeroed (identical off-diagonal counts to the official dense O(n²)
        intersection loop), then per dense count row ``torch.topk(row,
        min(nnz, co_occur_top_n))`` -- the official ``creat_dict_graph``
        selection semantics, including torch.topk's (torch-version-defined)
        tie order -- then a DETERMINISTIC ``topk_sample`` (cycle-padded, no
        RNG) selecting the first ``num_user_co``/``num_item_co`` neighbours
        with soft-maxed weights. Rows with no co-occurring peer fall back to
        the most-popular ids via a STABLE descending popularity sort
        (official: stable ``sorted(..., reverse=True)`` over dict insertion
        order; ties break by ascending id here vs first-appearance order
        there -- identical on id-sorted train files).
      - user-interest / item similarity kNN = chunked cosine top-k with the
        official symmetric degree normalisation.
  * DEVIATION: the user-interest average divides by the interaction row degree
    CLAMPED at 1 (``torch.clamp(row_deg, min=1.0)``); the official
    data_loader divides by the raw degree, so 0-interaction users yield NaN
    interest rows. Dormant on 5-core-filtered data (no 0-degree train users);
    the clamp only guards synthetic/edge fixtures.
  * REARM's internal node ids are GLOBAL (items offset by ``+n_users``); our
    ``interaction`` carries LOCAL item ids, so ``calculate_loss`` applies the
    ``+n_users`` offset when indexing the [n_users+n_items, ...] representation.
  * L2 is realised via the optimizer's ``weight_decay`` (their AdamW pattern);
    set ``optimizer: adamw`` and ``weight_decay: <reg_weight>`` in the YAML. No
    in-graph reg term (``reg_loss = 0``), matching the official code.
  * ``MultiheadAttention(embed_dim=1)`` construction is preserved verbatim -- the
    64 feature dims are treated as a length-64 sequence of size-1 tokens.
  * Attention dropout is driven by ``s_drop``/``m_drop`` (their own YAML keys);
    ``model.eval()`` disables it, so ``full_sort_predict`` is deterministic.

The official ``n_layers`` (heterography depth) is exposed as ``num_layers`` here
because ``n_layers`` is a reserved/deprecated model-config key in this repo; the
semantics are unchanged.
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


class REARM(RecommenderBase):
    def __init__(self, config, dataloader):
        super(REARM, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.n_nodes = self.n_users + self.n_items
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = self.embedding_dim

        # Homography / heterography propagation depths.
        self.num_layers = config["num_layers"]       # heterography (user-item)
        self.n_ii_layers = config["n_ii_layers"]     # item homography
        self.n_uu_layers = config["n_uu_layers"]     # user homography

        # Meta-network low-rank dimension.
        self.k = config["rank"]

        # Graph-mixing weights (co-occurrence vs similarity / modality blend).
        self.uu_co_weight = config["uu_co_weight"]
        self.ii_co_weight = config["ii_co_weight"]
        self.u_mm_image_weight = config["u_mm_image_weight"]
        self.i_mm_image_weight = config["i_mm_image_weight"]

        # kNN neighbourhood sizes and co-occurrence sample counts.
        self.user_knn_k = config["user_knn_k"]
        self.item_knn_k = config["item_knn_k"]
        self.num_user_co = config["num_user_co"]
        self.num_item_co = config["num_item_co"]
        # Per-row neighbour cap of the cached co-occurrence dict; mirrors the
        # official helper.creat_dict_graph constant (200). Plain, non-searched.
        self.co_occur_top_n = config["co_occur_top_n"]

        # Loss coefficients / temperature.
        self.cl_tmp = config["cl_tmp"]
        self.cl_loss_weight = config["cl_loss_weight"]
        self.diff_loss_weight = config["diff_loss_weight"]

        # Attention dropout probabilities (their own knobs, distinct from
        # dropout_rate). eval() turns these off -> deterministic scoring.
        self.s_drop = config["s_drop"]
        self.m_drop = config["m_drop"]

        # Train-only user-item interaction matrix (LOCAL item ids).
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)

        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        os.makedirs(dataset_path, exist_ok=True)

        # ------------------------------------------------------------------
        # Heterography (user-item) normalized adjacency A_hat (n_nodes x n_nodes).
        # ------------------------------------------------------------------
        self.norm_adj = self._build_ui_norm_adj().to(self.device)

        # ------------------------------------------------------------------
        # Attention / embedding modules.
        # ------------------------------------------------------------------
        self.ly_norm = nn.LayerNorm(self.feat_embed_dim)
        self.self_i_attn1 = nn.MultiheadAttention(1, 1, dropout=self.s_drop, batch_first=True)
        self.self_i_attn2 = nn.MultiheadAttention(1, 1, dropout=self.s_drop, batch_first=True)
        self.mutual_i_attn1 = nn.MultiheadAttention(1, 1, dropout=self.m_drop, batch_first=True)
        self.mutual_i_attn2 = nn.MultiheadAttention(1, 1, dropout=self.m_drop, batch_first=True)

        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        self.prl = nn.PReLU()
        # BPR contrast operator: score rows are [pos, neg] -> pos - neg.
        self.register_buffer("cal_bpr", torch.tensor([[1.0], [-1.0]]))

        # ------------------------------------------------------------------
        # Homography co-occurrence graphs (sparse B@Bᵀ -> top-200 -> topk_sample).
        # ------------------------------------------------------------------
        user_dict_graph = self._build_co_occurrence_dict(
            "user", dataset_path, "rearm_user_co_dict", self.co_occur_top_n
        )
        item_dict_graph = self._build_co_occurrence_dict(
            "item", dataset_path, "rearm_item_co_dict", self.co_occur_top_n
        )
        # Fallback rows (deterministic) for users/items with no co-occurring peer:
        # the globally most-interacted ids, mirroring topK_users / topK_items.
        # STABLE descending sort = the official stable sorted(..., reverse=True)
        # (data_loader.dict_user_items); popularity ties break by ascending id
        # here vs dict first-appearance order there (identical on id-sorted
        # train files).
        user_pop = np.asarray(self.interaction_matrix.sum(axis=1)).reshape(-1)
        item_pop = np.asarray(self.interaction_matrix.sum(axis=0)).reshape(-1)
        topk_users = np.argsort(-user_pop, kind="stable").tolist()
        topk_users_counts = user_pop[topk_users].tolist()
        topk_items = np.argsort(-item_pop, kind="stable").tolist()
        topk_items_counts = item_pop[topk_items].tolist()

        self.user_co_graph = self._topk_sample(
            self.n_users, user_dict_graph, self.num_user_co,
            topk_users, topk_users_counts,
        ).to(self.device)
        self.item_co_graph = self._topk_sample(
            self.n_items, item_dict_graph, self.num_item_co,
            topk_items, topk_items_counts,
        ).to(self.device)

        # ------------------------------------------------------------------
        # Similarity (interest) kNN graphs + user interest priors, from raw feats.
        # ------------------------------------------------------------------
        sp_inter = self._scipy_coo_to_torch(self.interaction_matrix).to(self.device)
        row_deg = torch.sparse.sum(sp_inter, [1]).to_dense().unsqueeze(dim=1)
        # DEVIATION (documented in the module docstring): the official
        # data_loader divides the interest sums by the RAW row degree, so
        # 0-interaction users yield NaN interest rows; we clamp the degree at 1
        # to keep those rows finite. Dormant on 5-core data (no 0-degree train
        # users) -- guards synthetic/edge fixtures only.
        row_deg = torch.clamp(row_deg, min=1.0)

        i_v_adj = i_t_adj = u_v_adj = u_t_adj = None
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_i_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_u_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            u_v_interest = torch.sparse.mm(sp_inter, self.v_feat) / row_deg
            self.user_v_prefer = nn.Parameter(u_v_interest.detach().clone())
            u_v_adj = self._build_knn_adj(u_v_interest, self.user_knn_k, dataset_path,
                                          "rearm_u_v_knn")
            i_v_adj = self._build_knn_adj(self.v_feat, self.item_knn_k, dataset_path,
                                          "rearm_i_v_knn")
            self.i_mm_adj = i_v_adj
            self.u_mm_adj = u_v_adj

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_i_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_u_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            u_t_interest = torch.sparse.mm(sp_inter, self.t_feat) / row_deg
            self.user_t_prefer = nn.Parameter(u_t_interest.detach().clone())
            u_t_adj = self._build_knn_adj(u_t_interest, self.user_knn_k, dataset_path,
                                          "rearm_u_t_knn")
            i_t_adj = self._build_knn_adj(self.t_feat, self.item_knn_k, dataset_path,
                                          "rearm_i_t_knn")
            self.i_mm_adj = i_t_adj
            self.u_mm_adj = u_t_adj

        if self.v_feat is not None and self.t_feat is not None:
            self.i_mm_adj = self.i_mm_image_weight * i_v_adj + (1.0 - self.i_mm_image_weight) * i_t_adj
            self.u_mm_adj = self.u_mm_image_weight * u_v_adj + (1.0 - self.u_mm_image_weight) * u_t_adj

        # Strengthened homography graphs (co-occurrence blended with similarity).
        self.stre_ii_graph = (
            self.ii_co_weight * self.item_co_graph
            + (1.0 - self.ii_co_weight) * self.i_mm_adj
        ).coalesce()
        self.stre_uu_graph = (
            self.uu_co_weight * self.user_co_graph
            + (1.0 - self.uu_co_weight) * self.u_mm_adj
        ).coalesce()

        # ------------------------------------------------------------------
        # Meta-network (knowledge compression + low-rank transform generators).
        # ------------------------------------------------------------------
        self.mlp_u1 = _MetaMLP(self.feat_embed_dim, self.feat_embed_dim * self.k)
        self.mlp_u2 = _MetaMLP(self.feat_embed_dim, self.feat_embed_dim * self.k)
        self.mlp_i1 = _MetaMLP(self.feat_embed_dim, self.feat_embed_dim * self.k)
        self.mlp_i2 = _MetaMLP(self.feat_embed_dim, self.feat_embed_dim * self.k)
        self.meta_netu = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim, bias=True)
        self.meta_neti = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim, bias=True)

        self._reset_parameters()

    # ======================================================================
    # Parameter init (mirrors official _reset_parameters).
    # ======================================================================
    def _reset_parameters(self):
        nn.init.normal_(self.user_id_embedding.weight, std=0.1)
        nn.init.normal_(self.item_id_embedding.weight, std=0.1)
        if self.v_feat is not None:
            nn.init.xavier_normal_(self.image_i_trs.weight)
            nn.init.xavier_normal_(self.image_u_trs.weight)
        if self.t_feat is not None:
            nn.init.xavier_normal_(self.text_i_trs.weight)
            nn.init.xavier_normal_(self.text_u_trs.weight)

    # ======================================================================
    # Graph builders (scalable rewrites of utils/helper.py).
    # ======================================================================
    def _scipy_coo_to_torch(self, coo):
        """scipy COO -> torch.sparse_coo float tensor (uncoalesced)."""
        coo = coo.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))

    def _sym_norm_from_scipy(self, adj_coo, size):
        """Symmetric degree-normalized sparse tensor D^-1/2 (adj) D^-1/2.

        Degree uses the adjacency's own row sums (values in [0,1] -> similarity
        normalization; 0/1 -> degree normalization), matching the official
        ``torch_sparse_tensor_norm_adj`` where sim_adj == degree_adj here except
        the kNN path passes an all-ones degree matrix (handled by the caller).
        """
        adj_coo = adj_coo.tocoo().astype(np.float32)
        row_sum = np.asarray(adj_coo.sum(axis=1)).reshape(-1) + 1e-7
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)
        norm = d_mat.dot(adj_coo).dot(d_mat).tocoo()
        return self._scipy_coo_to_torch(norm).coalesce()

    def _build_ui_norm_adj(self):
        """Bipartite user-item normalized adjacency (n_nodes x n_nodes)."""
        R = self.interaction_matrix.tocoo()
        # Off-diagonal blocks: users->items (col + n_users), items->users.
        up_rows = R.row
        up_cols = R.col + self.n_users
        lo_rows = R.col + self.n_users
        lo_cols = R.row
        rows = np.concatenate([up_rows, lo_rows])
        cols = np.concatenate([up_cols, lo_cols])
        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        return self._sym_norm_from_scipy(adj, self.n_nodes)

    def _co_occurrence_topk_dict(self, side, top_n, chunk=1024):
        """Per-row top-``top_n`` co-occurrence neighbours as {row: [idx, val]}.

        ``B @ Bᵀ`` counts shared partners: for the user side B is the U x I
        interaction matrix (two users co-occur through shared items); for the
        item side B is the I x U matrix. The diagonal (self co-occurrence) is
        zeroed -- the off-diagonal counts are identical to the official dense
        O(n²) intersection loop (``creat_co_occur_matrix``). Computed row-block
        by row-block so the full n x n product is never materialised.

        Neighbour SELECTION follows the official ``creat_dict_graph`` exactly:
        ``torch.topk(row, min(nnz, top_n))`` over each dense float32 count row,
        keeping torch.topk's own tie order (which is torch-version-defined --
        best-possible artifact equivalence; ``topk_sample`` later consumes the
        first ``num_*_co`` entries in THIS order).
        """
        if side == "user":
            B = self.interaction_matrix.tocsr()
        else:
            B = self.interaction_matrix.transpose().tocsr()
        n = B.shape[0]
        Bt = B.transpose().tocsr()
        dict_graph = {}
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            # Dense [block, n] shared-partner counts (float32, exact ints).
            block = (B[start:end] @ Bt).toarray()
            # Zero the self-loop (official co_graph_matrix never fills the
            # diagonal), so nnz and topk both exclude it.
            block[np.arange(end - start), np.arange(start, end)] = 0.0
            block_t = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
            for local_row in range(end - start):
                row = block_t[local_row]
                keep = min(int(torch.count_nonzero(row)), top_n)
                topk_ui = torch.topk(row, keep)
                dict_graph[start + local_row] = [
                    topk_ui.indices.tolist(), topk_ui.values.tolist(),
                ]
        return dict_graph

    def _build_co_occurrence_dict(self, side, dataset_path, cache_name, top_n):
        # top_n is part of the cache identity (it caps the stored rows).
        cache_file = os.path.join(dataset_path, "{}_top{}.pt".format(cache_name, top_n))
        if os.path.exists(cache_file):
            return torch.load(cache_file, map_location="cpu", weights_only=False)
        dict_graph = self._co_occurrence_topk_dict(side, top_n)
        torch.save(dict_graph, cache_file)
        return dict_graph

    def _topk_sample(self, n_ui, dict_graph, k, topk_ui, topk_ui_counts):
        """Deterministic top-k co-occurrence sampler (cycle-padded, no RNG).

        Selects the top-``k`` neighbours of each row and soft-maxes their counts
        into edge weights. When a row has fewer than ``k`` neighbours the sample
        is padded by CYCLING its own neighbours (official code pads with
        ``np.random.randint`` draws; we cycle deterministically -- same support,
        no eval/build nondeterminism). Rows with no neighbour fall back to the
        globally most-interacted ids (``topk_ui``), as in the official code.
        """
        weight_matrix = torch.zeros(n_ui, k)
        ui_graph_index = []
        for i in range(n_ui):
            neighbours = dict_graph[i][0]
            weights = dict_graph[i][1]
            if len(neighbours) == 0:
                sample = list(topk_ui[:k])
                denom = sum(topk_ui_counts[:k])
                weight = (np.array(topk_ui_counts[:k]) / denom).tolist()
            elif len(neighbours) < k:
                sample = list(neighbours)
                weight = list(weights)
                # Deterministic cycle padding.
                pad = k - len(sample)
                sample += [neighbours[j % len(neighbours)] for j in range(pad)]
                weight += [weights[j % len(weights)] for j in range(pad)]
            else:
                sample = list(neighbours[:k])
                weight = list(weights[:k])
            ui_graph_index.append(sample)
            weight_matrix[i] = F.softmax(torch.tensor(weight, dtype=torch.float32), dim=0)

        rows = []
        cols = []
        for i in range(n_ui):
            rows.extend([i] * k)
            cols.extend(ui_graph_index[i])
        indices = torch.tensor([rows, cols], dtype=torch.int64)
        values = weight_matrix.flatten()
        return torch.sparse_coo_tensor(indices, values, (n_ui, n_ui)).coalesce()

    def _build_knn_adj(self, mm_embeddings, knn_k, dataset_path, cache_name):
        cache_file = os.path.join(dataset_path, cache_name + "_{}.pt".format(knn_k))
        if os.path.exists(cache_file):
            return torch.load(cache_file, map_location="cpu", weights_only=False).to(self.device)
        adj = self._knn_normalized_graph(mm_embeddings.detach(), knn_k)
        torch.save(adj.cpu(), cache_file)
        return adj.to(self.device)

    def _knn_normalized_graph(self, embeddings, knn_k, chunk=2048):
        """Cosine-similarity top-k graph with the official degree normalization.

        Chunked over rows so the full n x n similarity is never materialised.
        The similarity VALUES weight the adjacency, while degree normalization
        uses an all-ones (unit) degree per edge -- exactly the official
        ``get_knn_adj_mat`` (``degree_adj`` built from ones).
        """
        n = embeddings.shape[0]
        context_norm = F.normalize(embeddings, dim=1)
        knn_k = min(knn_k, n)
        rows = []
        cols = []
        sim_vals = []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            block_sim = torch.mm(context_norm[start:end], context_norm.t())
            top_val, top_idx = torch.topk(block_sim, knn_k, dim=-1)
            block_rows = torch.arange(start, end, device=embeddings.device).unsqueeze(1).expand(-1, knn_k)
            rows.append(block_rows.reshape(-1))
            cols.append(top_idx.reshape(-1))
            sim_vals.append(top_val.reshape(-1))
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        sim_vals = torch.cat(sim_vals)
        indices = torch.stack((rows, cols))
        size = (n, n)
        # Degree from unit weights on the kept edges (matches official degree_adj).
        ones = torch.ones(indices.shape[1], device=embeddings.device)
        degree = torch.sparse_coo_tensor(indices, ones, size).coalesce()
        row_sum = 1e-7 + torch.sparse.sum(degree, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        diag_idx = torch.arange(n, device=embeddings.device)
        d_mat = torch.sparse_coo_tensor(torch.stack((diag_idx, diag_idx)), r_inv_sqrt, size).coalesce()
        sim_adj = torch.sparse_coo_tensor(indices, sim_vals, size).coalesce()
        norm = torch.sparse.mm(torch.sparse.mm(d_mat, sim_adj), d_mat)
        return norm.coalesce()

    # ======================================================================
    # Propagation helper (official propgt_info).
    # ======================================================================
    @staticmethod
    def _propagate(ego_feat, n_layers, sp_mat, last_layer=False):
        all_feat = [ego_feat]
        for _ in range(n_layers):
            ego_feat = torch.sparse.mm(sp_mat, ego_feat)
            all_feat.append(ego_feat)
        if last_layer:
            return ego_feat
        stacked = torch.stack(all_feat, dim=1)
        return stacked.mean(dim=1, keepdim=False)

    # ======================================================================
    # Forward: representation of shape [n_nodes, 3 * embedding_dim].
    # ======================================================================
    def forward(self):
        # Project trainable modality feature copies to feat_embed_dim.
        trs_item_v_feat = self.image_i_trs(self.image_embedding.weight)
        trs_item_t_feat = self.text_i_trs(self.text_embedding.weight)
        trs_user_v_prefer = self.image_u_trs(self.user_v_prefer)
        trs_user_t_prefer = self.text_u_trs(self.user_t_prefer)

        # ---- Item homography relation learning ----
        item_v_t = torch.cat((trs_item_v_feat, trs_item_t_feat), dim=-1)
        item_id_v_t = torch.cat((self.item_id_embedding.weight, item_v_t), dim=-1)
        item_id_v_t = self._propagate(item_id_v_t, self.n_ii_layers, self.stre_ii_graph, last_layer=True)
        item_id_v_t = F.normalize(item_id_v_t)

        item_id_ii = item_id_v_t[:, :self.embedding_dim]
        gnn_i_v_feat = item_id_v_t[:, self.feat_embed_dim:-self.feat_embed_dim]
        gnn_i_t_feat = item_id_v_t[:, -self.feat_embed_dim:]

        # ---- User homography relation learning ----
        user_v_t = torch.cat((trs_user_v_prefer, trs_user_t_prefer), dim=-1)
        user_id_v_t = torch.cat((self.user_id_embedding.weight, user_v_t), dim=-1)
        user_id_v_t = self._propagate(user_id_v_t, self.n_uu_layers, self.stre_uu_graph, last_layer=True)
        user_id_v_t = F.normalize(user_id_v_t)

        user_id_uu = user_id_v_t[:, :self.embedding_dim]
        gnn_u_v_prefer = user_id_v_t[:, self.embedding_dim:-self.feat_embed_dim]
        gnn_u_t_prefer = user_id_v_t[:, -self.feat_embed_dim:]

        # ---- Item feature attention integration (dims as size-1 tokens) ----
        item_v_feat, _ = self.self_i_attn1(
            gnn_i_v_feat.unsqueeze(2), gnn_i_v_feat.unsqueeze(2),
            gnn_i_v_feat.unsqueeze(2), need_weights=False,
        )
        item_v_feat = self.ly_norm(gnn_i_v_feat + item_v_feat.squeeze(-1))
        item_v_feat = self.prl(item_v_feat)

        item_t_feat, _ = self.self_i_attn2(
            gnn_i_t_feat.unsqueeze(2), gnn_i_t_feat.unsqueeze(2),
            gnn_i_t_feat.unsqueeze(2), need_weights=False,
        )
        item_t_feat = self.ly_norm(gnn_i_t_feat + item_t_feat.squeeze(-1))
        item_t_feat = self.prl(item_t_feat)

        # Cross-attention (text->visual and visual->text).
        i_t2v_feat, _ = self.mutual_i_attn1(
            item_t_feat.unsqueeze(2), item_v_feat.unsqueeze(2),
            item_v_feat.unsqueeze(2), need_weights=False,
        )
        item_t2v_feat = self.ly_norm(item_v_feat + i_t2v_feat.squeeze(-1))
        item_t2v_feat = self.prl(item_t2v_feat)

        i_v2t_feat, _ = self.mutual_i_attn2(
            item_v_feat.unsqueeze(2), item_t_feat.unsqueeze(2),
            item_t_feat.unsqueeze(2), need_weights=False,
        )
        item_v2t_feat = self.ly_norm(item_t_feat + i_v2t_feat.squeeze(-1))
        item_v2t_feat = self.prl(item_v2t_feat)

        user_v_prefer = self.prl(gnn_u_v_prefer)
        user_t_prefer = self.prl(gnn_u_t_prefer)

        # ---- Heterography relation learning (user-item) ----
        item_v_t_feat = torch.cat((item_t2v_feat, item_v2t_feat), dim=-1)   # [n_items, 128]
        user_v_t_prefer = torch.cat((user_v_prefer, user_t_prefer), dim=-1)  # [n_users, 128]
        ego_feat_prefer = torch.cat((user_v_t_prefer, item_v_t_feat), dim=0)
        fin_feat_prefer = self._propagate(ego_feat_prefer, self.num_layers, self.norm_adj)

        ego_id_embed = torch.cat((user_id_uu, item_id_ii), dim=0)            # [n_nodes, 64]
        fin_id_embed = self._propagate(ego_id_embed, self.num_layers, self.norm_adj)

        share_knowldge = self._meta_extra_share(fin_id_embed, fin_feat_prefer)

        fin_v = self.prl(fin_feat_prefer[:, :self.embedding_dim]) + fin_id_embed
        fin_t = self.prl(fin_feat_prefer[:, self.embedding_dim:]) + fin_id_embed
        fin_share = self.prl(share_knowldge) + fin_id_embed

        representation = torch.cat((fin_v, fin_t, fin_share), dim=-1)
        # fin_feat_prefer is needed by the CL / diff losses; return alongside.
        return representation, fin_feat_prefer

    # ======================================================================
    # Meta-network low-rank refinement (official meta_extra_share).
    # ======================================================================
    def _meta_extra_share(self, id_embed, prefer_or_feat):
        u_id_embed = id_embed[:self.n_users, :]
        i_id_embed = id_embed[self.n_users:, :]
        u_v_t = prefer_or_feat[:self.n_users, :]
        i_v_t = prefer_or_feat[self.n_users:, :]

        # Detach the compressed knowledge (meta-knowledge is a fixed target).
        u_knowldge = self.meta_netu(u_v_t).detach()
        i_knowldge = self.meta_neti(i_v_t).detach()

        metau1 = self.mlp_u1(u_knowldge).reshape(-1, self.feat_embed_dim, self.k)
        metau2 = self.mlp_u2(u_knowldge).reshape(-1, self.k, self.feat_embed_dim)
        metai1 = self.mlp_i1(i_knowldge).reshape(-1, self.feat_embed_dim, self.k)
        metai2 = self.mlp_i2(i_knowldge).reshape(-1, self.k, self.feat_embed_dim)

        meta_biasu = torch.mean(metau1, dim=0)
        meta_biasu1 = torch.mean(metau2, dim=0)
        meta_biasi = torch.mean(metai1, dim=0)
        meta_biasi1 = torch.mean(metai2, dim=0)

        low_weightu1 = F.softmax(metau1 + meta_biasu, dim=1)
        low_weightu2 = F.softmax(metau2 + meta_biasu1, dim=1)
        low_weighti1 = F.softmax(metai1 + meta_biasi, dim=1)
        low_weighti2 = F.softmax(metai2 + meta_biasi1, dim=1)

        u_middle = torch.sum(torch.multiply(u_id_embed.unsqueeze(-1), low_weightu1), dim=1)
        u_share = torch.sum(torch.multiply(u_middle.unsqueeze(-1), low_weightu2), dim=1)
        i_middle = torch.sum(torch.multiply(i_id_embed.unsqueeze(-1), low_weighti1), dim=1)
        i_share = torch.sum(torch.multiply(i_middle.unsqueeze(-1), low_weighti2), dim=1)

        return torch.cat((u_share, i_share), dim=0)

    # ======================================================================
    # Losses.
    # ======================================================================
    @staticmethod
    def _ssl_loss(data1, data2, index, ssl_temp):
        """InfoNCE aligning two views of the SAME nodes (official ssl_loss)."""
        index = torch.unique(index)
        emb1 = F.normalize(data1[index], p=2, dim=1)
        emb2 = F.normalize(data2[index], p=2, dim=1)
        pos_score = torch.sum(torch.mul(emb1, emb2), dim=1)
        all_score = torch.mm(emb1, emb2.T)
        pos_score = torch.exp(pos_score / ssl_temp)
        all_score = torch.sum(torch.exp(all_score / ssl_temp), dim=1)
        return -torch.sum(torch.log(pos_score / all_score)) / len(index)

    @staticmethod
    def _cal_diff_loss(feat, ui_index, dim):
        """Orthogonal constraint: zero-mean, row-L2-normalized (v^T @ t)^2."""
        input1 = feat[ui_index, :dim]
        input2 = feat[ui_index, dim:]
        input1 = input1 - torch.mean(input1, dim=0, keepdims=True)
        input2 = input2 - torch.mean(input2, dim=0, keepdims=True)
        n1 = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(n1.expand_as(input1) + 1e-6)
        n2 = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(n2.expand_as(input2) + 1e-6)
        return torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

    def calculate_loss(self, interaction):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        # Consume the dataloader's clean per-user history-avoiding negatives when
        # supplied (len>=3); otherwise fall back to uniform sampling.
        if len(interaction) >= 3:
            neg_items = interaction[2].to(self.device)
        else:
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=self.device)

        # REARM node ids are GLOBAL (items offset by +n_users); our interaction
        # carries LOCAL item ids.
        pos_global = pos_items + self.n_users
        neg_global = neg_items + self.n_users

        representation, fin_feat_prefer = self.forward()

        # BPR: rows [pos, neg] -> pos - neg per (user) pair.
        user_rep = representation[users]
        pos_rep = representation[pos_global]
        neg_rep = representation[neg_global]
        pos_score = torch.sum(user_rep * pos_rep, dim=1)
        neg_score = torch.sum(user_rep * neg_rep, dim=1)
        stacked = torch.stack((pos_score, neg_score), dim=1)  # [N, 2]
        bpr_score = torch.matmul(stacked, self.cal_bpr)
        bpr_loss = -torch.mean(nn.LogSigmoid()(bpr_score))

        # Indices into the [n_nodes, ...] fin_feat_prefer. The official loss()
        # feeds the item-side InfoNCE ``item_tensor.view(-1)`` where the loader
        # yields [pos_item, neg_item] per interaction -> the CL index covers
        # positives AND negatives (ssl_loss dedups via torch.unique, so order/
        # duplicates are immaterial). The diff loss instead consumes
        # ``ui_index[:, 0]`` -> POSITIVES only, duplicates kept.
        cl_item_idx = torch.cat((pos_global, neg_global))
        user_idx = users

        # InfoNCE aligning the v-half and t-half of the interest features.
        i_cl = self._ssl_loss(
            fin_feat_prefer[:, :self.feat_embed_dim],
            fin_feat_prefer[:, -self.feat_embed_dim:], cl_item_idx, self.cl_tmp,
        )
        u_cl = self._ssl_loss(
            fin_feat_prefer[:, :self.feat_embed_dim],
            fin_feat_prefer[:, -self.feat_embed_dim:], user_idx, self.cl_tmp,
        )
        mul_vt_cl_loss = self.cl_loss_weight * (i_cl + u_cl)

        # Orthogonal constraint on the v/t halves (positives only, dups kept).
        u_diff = self._cal_diff_loss(fin_feat_prefer, user_idx, self.feat_embed_dim)
        i_diff = self._cal_diff_loss(fin_feat_prefer, pos_global, self.feat_embed_dim)
        mul_diff_loss = self.diff_loss_weight * (i_diff + u_diff)

        # reg_loss = 0: L2 is realised via the optimizer weight_decay (AdamW).
        return bpr_loss + mul_vt_cl_loss + mul_diff_loss

    def full_sort_predict(self, interaction):
        user = interaction[0].to(self.device)
        representation, _ = self.forward()
        u_reps, i_reps = torch.split(representation, [self.n_users, self.n_items], dim=0)
        return torch.matmul(u_reps[user], i_reps.t())


class _MetaMLP(nn.Module):
    """Two-layer PReLU MLP with L2-normalized output (official MLP)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_pre = nn.Linear(input_dim, output_dim, bias=True)
        self.prl = nn.PReLU()
        self.linear_out = nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, data):
        x = self.prl(self.linear_pre(data))
        x = self.linear_out(x)
        return F.normalize(x, p=2, dim=-1)
