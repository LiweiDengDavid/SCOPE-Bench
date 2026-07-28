# coding: utf-8
r"""
MENTOR
################################################
Reference:
    https://github.com/Jinfeng-Xu/MENTOR
    AAAI'2025: [MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation]

Faithful centralized baseline port. The two multi-level self-supervised tasks
are ported from the official ``src/models/mentor.py`` (``calculate_loss`` /
``forward`` / ``GCN``):

  * cross-modal alignment (``align_loss``, gated by ``alpha``, guided by the ID
    modality) matches the first/second-order statistics of the ID, fused,
    visual and text representations pairwise;
  * feature enhancement (gated by ``beta``) combines a graph-perspective
    contrastive term (``mask_g_loss``, InfoNCE between two random-noise graph
    views at ``temperature``) and a feature-perspective consistency term
    (``mask_f_loss``, cosine agreement between a ``mask_rate``-masked view and an
    MLP-transformed view).

Adaptations for the NexusRec framework (CPU-safe, no external files):
  * modality features loaded MGCN-style (``from_pretrained`` + ``nn.Linear``);
  * the per-modality GCN propagates over a symmetric-normalized sparse U-I
    adjacency (``build_norm_adj_matrix``) instead of torch_geometric message
    passing, so it runs without CUDA / cached user-graph files;
  * the item semantic graph (kNN ``knn_k``, fusion ``lambda_coeff``) enhances
    all modalities including the fused one.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase
from core.utils import (
    build_knn_normalized_graph,
    build_norm_adj_matrix,
    build_sim,
)


class MENTOR(RecommenderBase):
    def __init__(self, config, dataloader):
        super(MENTOR, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.embedding_dim = config["embedding_size"]
        self.n_ui_layers = config["num_ui_layers"]
        self.n_layers = config["num_layers"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.align_weight = float(config["alpha"])
        self.enhance_weight = float(config["beta"])
        self.temperature = config["temperature"]
        self.mask_rate = config["mask_rate"]
        self.reg_weight = float(config["reg_weight"])

        # User-Item bipartite adjacency (symmetric normalized), used by the
        # per-modality GCN message passing.
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = build_norm_adj_matrix(
            self.interaction_matrix, self.n_users, self.n_items, device=self.device
        )

        # ID embeddings (guide modality) for users and items.
        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Modality features (MGCN-style) + projection to embedding_dim.
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # Item semantic (kNN) graph, fused across modalities, enhancing all
        # modalities including the fused one.
        self.mm_adj = self._build_item_semantic_graph()

        # Softmax-normalized weighting for per-user modality fusion (weight_u).
        weight_u = torch.tensor(
            np.random.randn(self.n_users, 2, 1), dtype=torch.float32
        )
        nn.init.xavier_normal_(weight_u)
        self.weight_u = nn.Parameter(weight_u)
        self.weight_u.data = F.softmax(self.weight_u.data, dim=1)

        # Feature-enhancement MLP (feature-perspective SSL view).
        self.enhance_mlp = nn.Linear(2 * self.embedding_dim, 2 * self.embedding_dim)

        # Cached fused user/item representation for full_sort_predict.
        self.result_embed = None

    def _build_item_semantic_graph(self):
        """kNN item-item semantic graph fused across modalities (lambda_coeff)."""
        image_adj = None
        text_adj = None
        if self.v_feat is not None:
            image_adj = build_knn_normalized_graph(
                build_sim(self.image_embedding.weight.detach()),
                topk=self.knn_k,
                is_sparse=True,
                norm_type="sym",
            )
        if self.t_feat is not None:
            text_adj = build_knn_normalized_graph(
                build_sim(self.text_embedding.weight.detach()),
                topk=self.knn_k,
                is_sparse=True,
                norm_type="sym",
            )
        if image_adj is not None and text_adj is not None:
            mm_adj = self.lambda_coeff * image_adj + (1.0 - self.lambda_coeff) * text_adj
        elif image_adj is not None:
            mm_adj = image_adj
        else:
            mm_adj = text_adj
        return mm_adj.coalesce().to(self.device)

    def pre_epoch_processing(self):
        pass

    def _modality_gcn(self, item_feats, perturbed=False):
        """Propagate a modality over the U-I graph, returning fused node reps.

        Mirrors the reference ``GCN``: user preference is a learnable embedding,
        items carry the (projected) modality feature; both are L2-normalized and
        passed through ``n_ui_layers`` of symmetric-normalized graph convolution,
        with the input and each layer summed (residual). ``perturbed`` injects
        SimGCL-style directional noise for the graph-perspective SSL views.
        """
        user_feats = self.user_id_embedding.weight
        x = torch.cat([user_feats, item_feats], dim=0)
        x = F.normalize(x, dim=-1)

        h = x
        out = x
        for _ in range(self.n_ui_layers):
            h = torch.sparse.mm(self.norm_adj, h)
            if perturbed:
                noise = torch.rand_like(h)
                h = h + torch.sign(h) * F.normalize(noise, dim=-1) * 0.1
            out = out + h
        return out

    def _build_item_graph(self, item_rep):
        """Enhance item representations through the semantic graph."""
        h = item_rep
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def forward(self):
        image_feats = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None
        text_feats = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None

        # Per-modality node representations over the U-I graph.
        v_rep = self._modality_gcn(image_feats)
        t_rep = self._modality_gcn(text_feats)
        id_rep = self._modality_gcn(self.item_id_embedding.weight)

        # Noise-perturbed graph views (graph-perspective SSL).
        v_rep_n1 = self._modality_gcn(image_feats, perturbed=True)
        t_rep_n1 = self._modality_gcn(text_feats, perturbed=True)
        v_rep_n2 = self._modality_gcn(image_feats, perturbed=True)
        t_rep_n2 = self._modality_gcn(text_feats, perturbed=True)

        # Concatenated modality views: fused (v||t), guide (id||id), v||v, t||t.
        v_user, v_item = v_rep[: self.n_users], v_rep[self.n_users:]
        t_user, t_item = t_rep[: self.n_users], t_rep[self.n_users:]
        id_user, id_item = id_rep[: self.n_users], id_rep[self.n_users:]

        # Weighted per-user fusion of the visual/text user reps (weight_u).
        stacked_user = torch.stack([v_user, t_user], dim=2)  # [U, d, 2]
        fused_user = (self.weight_u.transpose(1, 2) * stacked_user)  # [U, d, 2]
        fused_user = torch.cat([fused_user[:, :, 0], fused_user[:, :, 1]], dim=1)  # [U, 2d]

        guide_user = torch.cat([id_user, id_user], dim=1)
        v_user_rep = torch.cat([v_user, v_user], dim=1)
        t_user_rep = torch.cat([t_user, t_user], dim=1)

        fused_item = torch.cat([v_item, t_item], dim=1)
        guide_item = torch.cat([id_item, id_item], dim=1)
        v_item_rep = torch.cat([v_item, v_item], dim=1)
        t_item_rep = torch.cat([t_item, t_item], dim=1)

        # Enhance every item view through the semantic graph.
        fused_item = fused_item + self._build_item_graph(fused_item)
        guide_item = guide_item + self._build_item_graph(guide_item)
        v_item_rep = v_item_rep + self._build_item_graph(v_item_rep)
        t_item_rep = t_item_rep + self._build_item_graph(t_item_rep)

        # Noise views (fused).
        n1_user = self._fuse_user(v_rep_n1[: self.n_users], t_rep_n1[: self.n_users])
        n2_user = self._fuse_user(v_rep_n2[: self.n_users], t_rep_n2[: self.n_users])
        n1_item = torch.cat([v_rep_n1[self.n_users:], t_rep_n1[self.n_users:]], dim=1)
        n2_item = torch.cat([v_rep_n2[self.n_users:], t_rep_n2[self.n_users:]], dim=1)
        n1_item = n1_item + self._build_item_graph(n1_item)
        n2_item = n2_item + self._build_item_graph(n2_item)

        result = {
            "fused": torch.cat([fused_user, fused_item], dim=0),
            "guide": torch.cat([guide_user, guide_item], dim=0),
            "v": torch.cat([v_user_rep, v_item_rep], dim=0),
            "t": torch.cat([t_user_rep, t_item_rep], dim=0),
            "n1": torch.cat([n1_user, n1_item], dim=0),
            "n2": torch.cat([n2_user, n2_item], dim=0),
            "fused_user": fused_user,
            "fused_item": fused_item,
        }
        self.result_embed = result["fused"]
        return result

    def _fuse_user(self, v_user, t_user):
        stacked = torch.stack([v_user, t_user], dim=2)
        fused = self.weight_u.transpose(1, 2) * stacked
        return torch.cat([fused[:, :, 0], fused[:, :, 1]], dim=1)

    def _info_nce(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()

    @staticmethod
    def _dist_gap(a, b):
        """First/second-order statistic gap between two representations."""
        return (torch.abs(torch.var(a) - torch.var(b)) +
                torch.abs(torch.mean(a) - torch.mean(b)))

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        # Consume the dataloader's clean per-user history-avoiding negatives when
        # supplied (use_neg_sampling=true emits a 3-tuple), matching the other
        # baselines; only fall back to uniform sampling for the 2-tuple contract.
        if len(interaction) >= 3:
            neg_items = interaction[2]
        else:
            neg_items = torch.randint(
                0, self.n_items, pos_items.shape, device=pos_items.device
            )

        result = self.forward()
        fused = result["fused"]

        # BPR on the fused representation.
        u_e = fused[users]
        pos_e = fused[self.n_users + pos_items]
        neg_e = fused[self.n_users + neg_items]
        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularization on the fused batch reps and fusion weights.
        reg_loss = self.reg_weight * (
            (u_e ** 2).mean() + (pos_e ** 2).mean() + (neg_e ** 2).mean()
            + (self.weight_u ** 2).mean()
        )

        # (1) Multi-level cross-modal alignment guided by the ID modality.
        guide, v, t = result["guide"], result["v"], result["t"]
        align_loss = (
            self._dist_gap(guide, fused)
            + self._dist_gap(guide, v)
            + self._dist_gap(guide, t)
            + self._dist_gap(fused, v)
            + self._dist_gap(fused, t)
            + self._dist_gap(v, t)
        )

        # (2) General feature enhancement: graph-perspective + feature-perspective.
        # Batch-scope the graph-perspective InfoNCE (mirroring MGCN's side/content
        # contrastive term, which indexes by `users` / `pos_items`): index the two
        # perturbed views by the batch BEFORE `_info_nce`, so each similarity
        # matrix is [batch, batch] rather than [n_users, n_users] / [n_items,
        # n_items]. Passing the full blocks would build an O(n_nodes^2) similarity
        # matrix every step and OOM on large datasets (~36k+ users).
        n1, n2 = result["n1"], result["n2"]
        n1_user, n1_item = n1[: self.n_users], n1[self.n_users:]
        n2_user, n2_item = n2[: self.n_users], n2[self.n_users:]
        mask_g_loss = (
            self._info_nce(n1_user[users], n2_user[users], self.temperature)
            + self._info_nce(n1_item[pos_items], n2_item[pos_items], self.temperature)
        )

        # Feature-perspective consistency, also scoped to the batch nodes.
        u_rep = result["fused_user"][users]
        i_rep = result["fused_item"][pos_items]
        u_mlp = self.enhance_mlp(u_rep)
        i_mlp = self.enhance_mlp(i_rep)
        u_mask = F.dropout(u_rep, self.mask_rate, training=self.training)
        i_mask = F.dropout(i_rep, self.mask_rate, training=self.training)
        mask_f_loss = (
            (1 - F.cosine_similarity(u_mask, u_mlp).mean())
            + (1 - F.cosine_similarity(i_mask, i_mlp).mean())
        )
        enhance_loss = mask_g_loss + mask_f_loss

        return (
            bpr_loss
            + reg_loss
            + self.align_weight * align_loss
            + self.enhance_weight * enhance_loss
        )

    def full_sort_predict(self, interaction):
        users = interaction[0]
        result = self.forward()
        fused = result["fused"]
        user_tensor = fused[: self.n_users][users]
        item_tensor = fused[self.n_users:]
        return torch.matmul(user_tensor, item_tensor.t())
