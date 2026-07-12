# coding: utf-8
r"""
TMLP
################################################
Reference:
    Yang et al. "Topological Multi-Layer Perceptron for Multimodal
    Recommendation." AAAI'2025.
    Reference implementation snapshot: src/models/TMLP.py,
    src/utils/graphmlp.py, src/utils/tps.py, src/utils/tools.py.

Faithful port of the official ``TMLP`` model. Per-modality feature towers
(``MLP -> leaky_relu -> MLP1 -> GMLP``) are concatenated with an item-id
embedding into a ``[n_items, 3*d]`` item representation. The user-side
learnable ``preference`` [n_users, 3*d] is stacked on top and the whole
``[n_nodes, 3*d]`` matrix is L2-normalized and passed through two
LightGCN-style symmetric convolutions over the bipartite user-item graph:

    ``x_hat = conv(x) + x + conv(conv(x))``

Despite the "MLP" title, the two graph convolutions remain in the forward
path (they are the topological smoothing). The training objective is BPR plus
a single topology-aware contrastive term ``Ncontrast`` that pulls each batch
item toward the neighbours selected by an offline Topological Pruning Sampling
(TPS) graph.

Two deviations from the official code, both documented inline:

1. **TPS input provenance.** The official TPS pipeline (``src/utils/tps.py``)
   loads ``data/<ds>/adj_0.1.pt`` — a file with NO producer anywhere in the
   repo. The shipped companion artifact (``data/sports/adj_tensor_sampling_5.pt``)
   and the filename ``adj_0.1`` both point to the FREEDOM-style multimodal
   kNN item graph at ``mm_image_weight=0.1``. We therefore build that graph in
   ``__init__`` as the TPS input (documented inference, not verified against an
   upstream artifact).

2. **Vectorized TPS.** The repo computes the 4-term mutual information with a
   pure-Python O(nnz * N) edge loop (``tools.get_mutual_information``) that
   takes hours at dataset scale. The MI terms are closed-form from sparse
   boolean products (``n11 = A_bool @ A_bool.T`` on the edge set; ``n10 =
   supp_i - n11``; ``n01 = supp_j - n11``; ``n00 = N - supp_i - supp_j +
   n11``; probabilities use the *weighted* row sum). The vectorized pipeline
   reproduces the repo's ``get_mutual_information`` -> exp-normalize ->
   ``get_mutli_sim(ratio=1, K)`` -> ``get_L_DA`` output to float precision
   (verified: identical top-K selections, ``<1e-7`` max diff on ``L_DA``). The
   result is cached to ``tps_adj_{knn_k}_{K}.pt`` in the dataset dir.

Dead knobs from the official config are NOT ported: ``alpha2`` (never read),
``dropout: 0.8`` (read but never applied), ``n_ui_layers`` (unused; the two
convs are hard-wired), and the non-TPS FREEDOM ``mm_adj`` cache the official
model builds but never consumes.

Note on the shipped official log: the released Baby-dataset training log in
the official repo was produced by a DIFFERENT variant model/config
(``FREEDOM_gmlp24``, ``lr=5e-3``, ``alpha1=0.6``, ``fc=3``, ``K=9``), not the
canonical ``TMLP.py`` / ``TMLP.yaml`` this file ports. We treat the shipped
``TMLP.py`` + ``TMLP.yaml`` as the canonical release and port those (not the
undocumented ``gmlp24`` variant that produced the log).
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase
from core.utils import build_norm_adj_matrix


class GMLP(nn.Module):
    """Graph MLP feature tower (faithful to ``src/utils/graphmlp.py``).

    ``num_fc_layers`` Linear stack ``input_dim -> hid_dim -> ... -> output_dim``
    with ``act_fn`` + LayerNorm + Dropout after every hidden layer (the final
    layer is linear). Weights xavier-uniform, biases N(0, 1e-6).
    """

    _ACT_FNS = {
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "elu": F.elu,
        "selu": F.selu,
        "softplus": F.softplus,
    }

    def __init__(self, input_dim, hid_dim, dropout, output_dim, num_fc_layers, act_fn):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        for i in range(num_fc_layers):
            in_features = input_dim if i == 0 else hid_dim
            out_features = hid_dim if i < num_fc_layers - 1 else output_dim
            self.fc_layers.append(nn.Linear(in_features, out_features))

        if act_fn not in self._ACT_FNS:
            raise ValueError(f"Unsupported activation function: {act_fn}")
        self.act_fn = self._ACT_FNS[act_fn]

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        for fc in self.fc_layers:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.normal_(fc.bias, std=1e-6)

    def forward(self, x):
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:
                x = self.act_fn(x)
                x = self.layernorm(x)
                x = self.dropout(x)
        return x


class TMLP(RecommenderBase):
    """Topological Multi-Layer Perceptron multimodal recommender (AAAI'25)."""

    def __init__(self, config, dataloader):
        super(TMLP, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.knn_k = config["knn_k"]
        # In-model preference L2 coefficient. The official config fixes this to
        # 0.0 and the preferences stay ``None`` (so this term is always 0); the
        # knob is a plain YAML key, NOT searched (dead in official TMLP).
        self.reg_weight = float(config["reg_weight"])
        self.mm_image_weight = config["mm_image_weight"]
        self.hidden_dim = config["hidden_dim"]
        self.v_dropout = config["v_dropout"]
        self.t_dropout = config["t_dropout"]
        self.num_fc_layers = config["num_fc_layers"]
        self.act_fn = config["act_fn"]
        self.tau1 = config["tau1"]
        self.alpha1 = config["alpha1"]
        self.aggr_mode = config["aggr_mode"]
        # TPS top-K neighbours retained per row.
        self.tps_K = config["tps_K"]

        self.n_nodes = self.n_users + self.n_items

        # Bipartite train interaction matrix.
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)

        # Bipartite convolution operator over the [n_users+n_items] node set.
        # The official ``Base_gcn`` distinguishes aggr=='add' (symmetric
        # D^-1/2 A D^-1/2 -- its default and the only value the official config
        # searches) from any other reduction (unnormalized mean of neighbours ==
        # row-stochastic D^-1 A). Both are static graph operators built once;
        # ``torch.sparse.mm(gcn_adj, x)`` reproduces ``Base_gcn(x, edges)`` for
        # the corresponding mode to float precision. Keeping both live makes the
        # searchable ``aggr_mode`` knob non-inert.
        self.gcn_adj = self._build_gcn_adj(self.aggr_mode)

        # Item-id embedding.
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # NOTE: the official model creates trainable ``image_embedding`` /
        # ``text_embedding`` copies but uses them ONLY to build its (dead) mm_adj
        # cache -- ``forward()`` projects the FROZEN raw ``self.v_feat`` /
        # ``self.t_feat`` (see official TMLP.forward). We therefore do NOT create
        # trainable feature embeddings (they never reach the optimizer in the
        # official forward path) and project the raw features directly; the TPS
        # input graph is built from the detached raw features below.

        # Per-modality projection towers: MLP -> leaky_relu -> MLP1 -> GMLP.
        self.MLP_v = nn.Linear(self.v_feat.shape[1], 2 * self.feat_embed_dim)
        self.MLP_v1 = nn.Linear(2 * self.feat_embed_dim, self.feat_embed_dim)
        self.MLP_t = nn.Linear(self.t_feat.shape[1], 2 * self.feat_embed_dim)
        self.MLP_t1 = nn.Linear(2 * self.feat_embed_dim, self.feat_embed_dim)

        self.vgmlp = GMLP(
            self.embedding_dim,
            self.hidden_dim,
            self.v_dropout,
            output_dim=self.embedding_dim,
            num_fc_layers=self.num_fc_layers,
            act_fn=self.act_fn,
        )
        self.tgmlp = GMLP(
            self.embedding_dim,
            self.hidden_dim,
            self.t_dropout,
            output_dim=self.embedding_dim,
            num_fc_layers=self.num_fc_layers,
            act_fn=self.act_fn,
        )

        # User-side learnable preference, width 3*d (visual+text+id blocks).
        self.preference = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty(self.n_users, self.embedding_dim * 3), gain=1.0
            )
        )

        # Reg preferences stay None (matches official; reg term is always 0).
        self.v_preference, self.t_preference = None, None

        # Build (or load) the TPS-pruned item-item graph used by Ncontrast.
        self.adj_tensor = self._build_or_load_tps_adj(config)

    def _build_gcn_adj(self, aggr_mode):
        """Build the static bipartite convolution operator for ``aggr_mode``.

        ``add``  -> symmetric D^-1/2 A D^-1/2 (official Base_gcn('add') default);
        anything else -> row-stochastic D^-1 A (== PyG mean aggregation, the
        official non-'add' branch). Returns a sparse tensor on ``self.device``.
        """
        if aggr_mode == "add":
            return build_norm_adj_matrix(
                self.interaction_matrix, self.n_users, self.n_items, self.device
            )
        # row-stochastic (mean) normalization D^-1 A over the bipartite graph
        inter_M = self.interaction_matrix
        inter_M_t = inter_M.transpose()
        row = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        col = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        data = np.ones(len(row), dtype=np.float32)
        A = sp.coo_matrix(
            (data, (row, col)), shape=(self.n_nodes, self.n_nodes), dtype=np.float32
        )
        deg = np.asarray(A.sum(axis=1)).flatten() + 1e-7
        D_inv = sp.diags(np.power(deg, -1.0))
        L = sp.coo_matrix(D_inv @ A)
        indices = torch.LongTensor(np.vstack((L.row, L.col)))
        values = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(
            indices, values, torch.Size(L.shape)
        ).to(self.device)

    # ------------------------------------------------------------------
    # TPS graph construction (offline pruning folded into __init__)
    # ------------------------------------------------------------------
    def _build_or_load_tps_adj(self, config):
        """Build the FREEDOM-style mm kNN item graph, prune it with the
        vectorized TPS mutual-information sampler, and cache the result.

        Returns a sparse ``[n_items, n_items]`` row-stochastic (``D^-1 (A+I)``)
        tensor on ``self.device``.
        """
        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        os.makedirs(dataset_path, exist_ok=True)
        cache_file = os.path.join(
            dataset_path, "tps_adj_{}_{}.pt".format(self.knn_k, self.tps_K)
        )
        if os.path.exists(cache_file):
            adj = torch.load(cache_file, map_location="cpu", weights_only=False)
            return adj.coalesce().to(self.device)

        # 1. FREEDOM-style mm kNN item graph at mm_image_weight (the TPS input,
        #    standing in for the repo's producerless ``adj_0.1.pt``). Boolean
        #    support = the kNN neighbour set; values = symmetric laplacian
        #    weights (they only enter TPS via the weighted row sum p_x).
        mm_adj = self._build_mm_knn_adj()  # scipy csr [n_items, n_items]

        # 2. Vectorized TPS pruning -> sparse row-stochastic L_DA.
        adj = self._tps_prune(mm_adj, self.tps_K)

        torch.save(adj, cache_file)
        return adj.coalesce().to(self.device)

    def _build_mm_knn_adj(self):
        """FREEDOM ``get_knn_adj_mat`` (symmetric laplacian, cosine top-k)
        blended ``mm_image_weight*image + (1-w)*text``, returned as scipy csr.
        """
        adj = None
        if self.v_feat is not None:
            image_adj = self._knn_laplacian_csr(self.v_feat.detach())
            adj = image_adj
        if self.t_feat is not None:
            text_adj = self._knn_laplacian_csr(self.t_feat.detach())
            adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        return adj.tocsr()

    def _knn_laplacian_csr(self, mm_embeddings):
        """Cosine kNN top-k adjacency with symmetric D^-1/2 A D^-1/2 weights,
        matching FREEDOM ``get_knn_adj_mat`` / ``compute_normalized_laplacian``.
        Returned as a scipy csr matrix ``[n_items, n_items]``.
        """
        n = mm_embeddings.shape[0]
        context_norm = mm_embeddings / torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        rows = torch.arange(n).unsqueeze(1).expand(-1, self.knn_k).flatten()
        cols = knn_ind.flatten()
        # symmetric normalization by kNN out-degree (== knn_k for every row)
        deg = torch.zeros(n)
        deg.index_add_(0, rows, torch.ones(rows.size(0)))
        r_inv_sqrt = torch.pow(1e-7 + deg, -0.5)
        values = r_inv_sqrt[rows] * r_inv_sqrt[cols]
        data = values.cpu().numpy()
        r = rows.cpu().numpy()
        c = cols.cpu().numpy()
        return sp.coo_matrix((data, (r, c)), shape=(n, n))

    def _tps_prune(self, A, K):
        """Vectorized Topological Pruning Sampling.

        Faithful closed-form reimplementation of ``tps.py`` (K>1 path) +
        ``tools.get_mutual_information`` / ``get_mutli_sim`` / ``get_L_DA``.

        Steps (all on the same sparse support as the official code):
          MI(i,j) over existing edges (i != j) from the 4 boolean-count terms;
          exp on stored entries; row-normalize by stored-row sum;
          per row keep the top-K positive entries until their cumulative
          normalized mass reaches ratio=1; mask A to those selections;
          ``L_DA = D^-1 (A + I)`` with a +1 self-loop in D.

        NOTE: this materializes several dense ``[N, N]`` float64 intermediates
        (``edge_mask``, ``N11``, ``mi``, ``exp_mi``, ``sel``, ``A_masked``) --
        there is NO chunking. Each float64 array costs ``8 * N^2`` bytes:
        ~0.5 GB at N=8k and ~2.6 GB at N=18k -- workable on Baby/Sports/
        Clothing-scale item counts; around 63k+ items (~32 GB per array, e.g.
        Bili-scale catalogs) it will OOM, where a chunked or sparse-only MI
        computation would be required.
        """
        A = A.tocsr()
        N = A.shape[0]
        bias = 1e-10

        Ab = (A > 0).astype(np.float64)  # boolean support
        supp = np.asarray(Ab.sum(axis=1)).reshape(-1)  # |supp_i| (intersection counts)
        d_w = np.asarray(A.sum(axis=1)).reshape(-1)     # weighted row sum drives p_x
        p_x = d_w / N
        p_inv = 1.0 - p_x

        # Pairwise intersection counts on the *edge* support only (dense
        # [N, N] intermediate; no chunking -- see the OOM bound in the
        # docstring above). n11 = |supp_i ∩ supp_j|.
        edge_mask = A.toarray() > 0
        np.fill_diagonal(edge_mask, False)  # repo skips i == j

        N11 = (Ab @ Ab.T).toarray()
        supp_i = supp.reshape(-1, 1)
        supp_j = supp.reshape(1, -1)
        N10 = supp_i - N11
        N01 = supp_j - N11
        N00 = N - supp_i - supp_j + N11

        def _term(n, pa, pb):
            e = n / N
            return e * np.log(e / (pa * pb + bias) + bias)

        mi = (
            _term(N11, p_x.reshape(-1, 1), p_x.reshape(1, -1))
            + _term(N10, p_x.reshape(-1, 1), p_inv.reshape(1, -1))
            + _term(N01, p_inv.reshape(-1, 1), p_x.reshape(1, -1))
            + _term(N00, p_inv.reshape(-1, 1), p_inv.reshape(1, -1))
        )
        mi = np.where(edge_mask, mi, 0.0)

        # exp on stored entries, row-normalize by stored-row sum.
        exp_mi = np.where(edge_mask, np.exp(mi), 0.0)
        row_sum = exp_mi.sum(axis=1, keepdims=True)
        norm_mi = np.divide(
            exp_mi, row_sum, out=np.zeros_like(exp_mi), where=row_sum > 0
        )

        # get_mutli_sim(ratio=1, K): descending sort, take up to K positive
        # entries until cumulative mass >= 1.
        sel = np.zeros((N, N), dtype=bool)
        order = np.argsort(-norm_mi, axis=1, kind="stable")
        for i in range(N):
            cum = 0.0
            for j in order[i, :K]:
                if norm_mi[i, j] > 0:
                    cum += norm_mi[i, j]
                    sel[i, j] = True
                if cum >= 1.0:
                    break

        # Mask the ORIGINAL weighted A by the selected neighbours, then L_DA.
        A_masked = A.toarray() * sel  # keep original weights on kept edges
        A_id = A_masked + np.eye(N)
        deg_loop = A_masked.sum(axis=1) + 1.0  # get_D self-loop (+1)
        d_inv = 1.0 / deg_loop
        L_DA = d_inv.reshape(-1, 1) * A_id  # D^-1 (A + I)

        L_DA = sp.coo_matrix(L_DA)
        indices = torch.LongTensor(np.vstack((L_DA.row, L_DA.col)))
        values = torch.FloatTensor(L_DA.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size((N, N)))

    def pre_epoch_processing(self):
        """No pre-epoch work (matches the official ``pass``)."""
        pass

    # ------------------------------------------------------------------
    # Forward / losses
    # ------------------------------------------------------------------
    def _gcn(self, all_feat):
        """LightGCN-style topological smoothing.

        ``x = F.normalize(cat(preference, all_feat))`` then
        ``x_hat = conv(x) + x + conv(conv(x))`` where conv is the symmetric
        deg-normalized bipartite neighbour-sum ``torch.sparse.mm(norm_adj, .)``.
        """
        x = torch.cat((self.preference, all_feat), dim=0)
        x = F.normalize(x)
        h = torch.sparse.mm(self.gcn_adj, x)       # conv(x)
        h_1 = torch.sparse.mm(self.gcn_adj, h)     # conv(conv(x))
        x_hat = h + x + h_1
        return x_hat

    def forward(self):
        v_feat = self.MLP_v1(F.leaky_relu(self.MLP_v(self.v_feat)))
        t_feat = self.MLP_t1(F.leaky_relu(self.MLP_t(self.t_feat)))
        id_feat = self.item_id_embedding.weight
        v_feat = self.vgmlp(v_feat)
        t_feat = self.tgmlp(t_feat)
        all_feat = torch.cat((v_feat, t_feat, id_feat), dim=1)  # [n_items, 3*d]

        all_rep = self._gcn(all_feat)
        user_rep = all_rep[: self.n_users]
        item_rep = all_rep[self.n_users:]
        return user_rep, item_rep

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        return -torch.mean(maxi)

    def get_feature_dis(self, x):
        """Cosine-similarity matrix with the diagonal zeroed."""
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x / x_norm
        x_dis = x @ x.T
        mask = torch.eye(x_dis.shape[0], device=x.device)
        x_dis = (1 - mask) * x_dis
        return x_dis

    def Ncontrast(self, x_dis, adj_label, tau=1):
        """Topology-aware contrastive loss: pull items toward TPS neighbours.

        ``-log( sum(exp(tau*sim) * adj) / sum(exp(tau*sim)) )``.
        """
        x_dis = torch.exp(tau * x_dis)
        x_dis_sum = torch.sum(x_dis, 1)
        x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
        loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
        return loss

    def _dense_block_from_sparse(self, item_idx):
        """Extract the ``[B, B]`` dense block ``adj_tensor[item_idx][:, item_idx]``
        WITHOUT densifying the full ``[n_items, n_items]`` graph (the official
        hotspot called ``.to_dense()`` on the whole pruned adj every batch).

        Fully vectorized: keep only the COO edges whose BOTH endpoints appear in
        the batch, then scatter their values into every ``(row-slot, col-slot)``
        combination. Duplicate ``item_idx`` entries (pos/neg collisions) are
        replicated to all matching slots, so the result is bit-identical to
        ``adj_tensor.to_dense()[item_idx][:, item_idx]``.
        """
        adj = self.adj_tensor.coalesce()
        r, c = adj.indices()[0], adj.indices()[1]
        v = adj.values()
        n_items = self.n_items
        B = item_idx.shape[0]
        device = item_idx.device

        # CSR-style ragged map: global id -> the batch positions holding it.
        order = torch.argsort(item_idx)
        counts = torch.zeros(n_items, dtype=torch.long, device=device)
        counts.index_add_(0, item_idx, torch.ones(B, dtype=torch.long, device=device))
        starts = torch.zeros(n_items + 1, dtype=torch.long, device=device)
        starts[1:] = torch.cumsum(counts, 0)

        block = torch.zeros(B, B, device=self.adj_tensor.device, dtype=v.dtype)
        keep = (counts[r] > 0) & (counts[c] > 0)
        r, c, v = r[keep], c[keep], v[keep]
        if r.numel() == 0:
            return block

        # Expand each surviving edge over all (row-slot x col-slot) combinations.
        rc, cc = counts[r], counts[c]
        ncomb = rc * cc
        total = int(ncomb.sum())
        e = torch.repeat_interleave(torch.arange(r.numel(), device=device), ncomb)
        off = torch.arange(total, device=device) - torch.repeat_interleave(
            torch.cumsum(ncomb, 0) - ncomb, ncomb
        )
        ri = off // cc[e]
        ci = off % cc[e]
        a = order[starts[r[e]] + ri]
        b = order[starts[c[e]] + ci]
        block[a, b] = v[e]
        return block

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        if len(interaction) >= 3:
            neg_items = interaction[2]
        else:
            neg_items = torch.randint(
                0, self.n_items, pos_items.shape, device=pos_items.device
            )

        ua_embeddings, ia_embeddings = self.forward()

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        # Topology-aware contrastive loss over the batch items. Slice the pruned
        # adjacency SPARSELY -> dense only the [B, B] block for Ncontrast.
        item_idx = torch.cat((pos_items, neg_items))
        ii_dis = self.get_feature_dis(ia_embeddings[item_idx])
        ii_adj = self._dense_block_from_sparse(item_idx)
        ncloss1 = self.alpha1 * self.Ncontrast(ii_dis, ii_adj, tau=self.tau1)

        # Preference L2 (always 0 in the official config: preferences are None).
        reg_embedding_loss_v = (
            (self.v_preference[users] ** 2).mean()
            if self.v_preference is not None
            else 0.0
        )
        reg_embedding_loss_t = (
            (self.t_preference[users] ** 2).mean()
            if self.t_preference is not None
            else 0.0
        )
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)

        return batch_mf_loss + reg_loss + ncloss1

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
