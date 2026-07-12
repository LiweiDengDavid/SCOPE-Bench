# coding: utf-8
r"""
MIG-GT
################################################
Reference:
    https://github.com/CrawlScript/MIG-GT
    AAAI'2025: [Modality-Independent Graph Neural Networks with Global
    Transformers for Multimodal Recommendation]

Faithful centralized baseline. Two pillars, ported from the official repo:

  1. Modality-independent GNNs (mig_gt/layers/mirf_gt.py ``MIGGT.forward`` +
     mig_gt/layers/mgdcf.py ``MGDCF``): a *separate* GNN runs over the shared
     user-item graph for each modality (id / text / visual) with an
     *independent* per-modality hop count (``k_id`` / ``k_text`` / ``k_visual``).
     The per-modality node embeddings are summed into a combined embedding.
     Repo uses DGL message passing; here propagation is expressed with the
     framework's symmetrically-normalized sparse U-I adjacency (mirrors MGCN),
     which is the same GCN operator.

  2. Sampling-based global transformer (mig_gt/layers/mirf_gt.py ``Transformer``
     applied as ``z_transformer`` in ``MIGGT.forward``): for every node, a set
     of ``global_sample_size`` items is uniformly sampled from the whole graph;
     each node's own embedding is prepended as the query token and multi-head
     attention (``n_heads``) reinjects global information. Token 0 is read back
     out as the refined node embedding.

Loss = BPR + optional contrastive (``cl_weight``) + ``reg_weight`` L2, with
negatives sampled internally in ``calculate_loss``.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


class _GlobalTransformer(nn.Module):
    """Sampling-based global transformer (ports mirf_gt.py ``Transformer``).

    Q/K linear projections, multi-head scaled-dot-product attention over the
    sampled global memory, with the residual mixing ``0.1*att + 0.9*query``
    used by the official ``z_transformer``.

    Ported faithfully: the official ``z_transformer`` is called as
    ``z_transformer(memory, memory)`` so query == key == the full memory
    (self token + sampled global tokens). It therefore returns a *per-token*
    output ``[N, 1+C, d]`` (``z_memory_h``), whose token 0 is the refined node
    embedding and whose C sampled-token slots feed the TUR loss.
    """

    def __init__(self, dim, att_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(dim, att_dim)
        self.k_linear = nn.Linear(dim, att_dim)
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.zeros_(self.k_linear.bias)

    def forward(self, memory):
        # memory: [N, 1 + C, dim]; q == k == memory (official z_transformer).
        Q = self.q_linear(memory)
        K = self.k_linear(memory)
        V = memory

        # Split channels across heads and stack them along the batch axis.
        Q_ = torch.cat(Q.split(Q.size(-1) // self.num_heads, dim=-1), dim=0)
        K_ = torch.cat(K.split(K.size(-1) // self.num_heads, dim=-1), dim=0)
        V_ = torch.cat(V.split(V.size(-1) // self.num_heads, dim=-1), dim=0)

        sim = Q_ @ K_.transpose(-2, -1)
        sim = sim / (Q_.size(-1) ** 0.5)
        sim = F.softmax(sim, dim=-1)
        att_h = sim @ V_

        # Merge heads back onto the channel axis.
        att_h = torch.cat(att_h.split(memory.size(0), dim=0), dim=-1)

        # Official residual mixing for z_transformer.
        att_h = att_h * 0.1 + memory * 0.9
        return att_h  # [N, 1+C, dim]


class MIG_GT(RecommenderBase):
    supports_multi_negatives = False

    def __init__(self, config, dataloader):
        super(MIG_GT, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.embedding_dim = config["embedding_size"]
        # Independent per-modality hop counts.
        self.k_id = config["k_id"]
        self.k_text = config["k_text"]
        self.k_visual = config["k_visual"]
        self.n_heads = config["n_heads"]
        self.global_sample_size = config["global_sample_size"]
        # Fixed seed for the deterministic global sample at inference so that
        # full_sort_predict is a stable function of the trained weights.
        self.eval_sample_seed = int(config["eval_sample_seed"])
        # Transformer Unsmooth Regularization weight (MIG-GT's named auxiliary).
        self.tur_weight = float(config["tur_weight"])
        # Ablation-only contrastive (MIG-GT-CL variant); gated OFF by default.
        self.cl_weight = config["cl_weight"]
        # InfoNCE temperature for the ablation CL term (YAML, not a code literal).
        self.cl_temperature = float(config["cl_temperature"])
        self.reg_weight = float(config["reg_weight"])

        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)

        # ID embeddings for users and items.
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Modality feature projections to the shared embedding size (MGCN-style).
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # Symmetrically-normalized U-I adjacency over (users + items).
        self.norm_adj = self._build_norm_adj().to(self.device)

        # One global transformer shared across the fused node embeddings.
        att_dim = max(self.embedding_dim // 4, self.n_heads)
        # Attention dim must be divisible by heads for the split/merge.
        att_dim = (att_dim // self.n_heads) * self.n_heads
        att_dim = max(att_dim, self.n_heads)
        self.z_transformer = _GlobalTransformer(self.embedding_dim, att_dim, self.n_heads)

    def _build_norm_adj(self):
        """Build a sym-normalized sparse adjacency over user+item nodes."""
        n = self.n_users + self.n_items
        adj = sp.dok_matrix((n, n), dtype=np.float32)
        adj = adj.tolil()
        R = self.interaction_matrix.tolil()
        adj[: self.n_users, self.n_users:] = R
        adj[self.n_users:, : self.n_users] = R.T
        adj = adj.todok()

        rowsum = np.array(adj.sum(axis=1)) + 1e-7
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
        norm = d_mat.dot(adj).dot(d_mat).tocoo()

        indices = torch.from_numpy(np.vstack((norm.row, norm.col)).astype(np.int64))
        values = torch.from_numpy(norm.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(norm.shape), dtype=torch.float32)

    def _propagate(self, ego_embeddings, k):
        """k-hop GCN propagation over the shared U-I graph (one modality GNN)."""
        all_embeddings = [ego_embeddings]
        h = ego_embeddings
        for _ in range(k):
            h = torch.sparse.mm(self.norm_adj, h)
            all_embeddings.append(h)
        return torch.stack(all_embeddings, dim=1).mean(dim=1)

    def _global_transformer(self, node_embeddings):
        """Uniform global item sampling + attention (sampling-based transformer).

        Returns the full per-token output ``z_memory_h`` of shape ``[N, 1+C, d]``:
        token 0 is the refined node embedding, tokens 1..C are the transformer
        outputs for the C uniformly-sampled global items. TUR scores all C+1.
        """
        num_nodes = node_embeddings.size(0)
        item_embeddings = node_embeddings[self.n_users:]

        # Clamp so the tiny test graph still has enough items to sample.
        sample_size = min(self.global_sample_size, self.n_items)

        if self.training:
            idx = torch.randint(
                0, self.n_items, (num_nodes, sample_size), device=node_embeddings.device
            )
        else:
            # Deterministic global sample at inference: without this the uniform
            # token sampling reruns on every full_sort_predict call, so a fixed
            # checkpoint yields different scores (hence different NDCG/Recall)
            # each eval and HPO ends up comparing noisy objectives. A fixed-seed
            # generator keeps the global-transformer path (faithful) while making
            # eval a stable function of the weights.
            gen = torch.Generator(device=node_embeddings.device)
            gen.manual_seed(self.eval_sample_seed)
            idx = torch.randint(
                0, self.n_items, (num_nodes, sample_size),
                generator=gen, device=node_embeddings.device,
            )
        sampled = item_embeddings[idx]  # [N, C, D]

        # Prepend each node's own embedding as the self token (index 0).
        memory = torch.cat([node_embeddings.unsqueeze(1), sampled], dim=1)  # [N, 1+C, D]

        z_memory_h = self.z_transformer(memory)  # [N, 1+C, D]
        return z_memory_h

    def forward(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_id_embedding.weight

        # --- Modality-independent GNNs (independent per-modality hop counts) ---
        # ID GNN.
        id_ego = torch.cat([user_emb, item_emb], dim=0)
        combined = self._propagate(id_ego, self.k_id)

        # Text GNN (users seeded with zeros; item side seeded with text features).
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_ego = torch.cat([torch.zeros_like(user_emb), text_feats], dim=0)
            combined = combined + self._propagate(text_ego, self.k_text)

        # Visual GNN.
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_ego = torch.cat([torch.zeros_like(user_emb), image_feats], dim=0)
            combined = combined + self._propagate(image_ego, self.k_visual)

        # --- Sampling-based global transformer reinjects global information ---
        # z_memory_h: [N, 1+C, d]; token 0 = refined node embedding.
        z_memory_h = self._global_transformer(combined)
        refined = z_memory_h[:, 0]

        users, items = torch.split(refined, [self.n_users, self.n_items], dim=0)
        return users, items, z_memory_h

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        if len(interaction) >= 3:
            neg_items = interaction[2]
        else:
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=pos_items.device)

        user_all, item_all, z_memory_h = self.forward()

        u_e = user_all[users]
        pos_e = item_all[pos_items]
        neg_e = item_all[neg_items]

        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        reg_loss = self.reg_weight * (
            u_e.pow(2).sum() + pos_e.pow(2).sum() + neg_e.pow(2).sum()
        ) / users.shape[0]

        loss = bpr_loss + reg_loss

        # --- Transformer Unsmooth Regularization (TUR): MIG-GT's named auxiliary
        # Ports official main.py (~L297-301). For each positive (user, item)
        # edge, score the positive user's refined embedding against the C+1
        # transformer tokens (``z_memory_h``) of the POSITIVE ITEM node, then a
        # cross-entropy forces the true self token (index 0, the neighbor token)
        # to outscore the C uniformly-sampled global tokens. This regularizes
        # the transformer against over-smoothing toward random global items.
        if self.tur_weight > 0:
            pos_user_h = user_all[users]                       # [B, d]
            pos_item_nodes = pos_items + self.n_users          # item node offset
            pos_z_memory_h = z_memory_h[pos_item_nodes]        # [B, 1+C, d]
            unsmooth_logits = (
                pos_user_h.unsqueeze(1) @ pos_z_memory_h.permute(0, 2, 1)
            ).squeeze(1)                                       # [B, 1+C]
            tur_targets = torch.zeros(
                users.shape[0], dtype=torch.long, device=unsmooth_logits.device
            )
            tur_loss = F.cross_entropy(unsmooth_logits, tur_targets)
            loss = loss + self.tur_weight * tur_loss

        # Ablation-only contrastive (MIG-GT-CL variant); gated OFF by default
        # (cl_weight=0 in the base config). NOT part of the base model's loss.
        if self.cl_weight > 0:
            refined_items = item_all
            self_token_items = z_memory_h[self.n_users:, 0]
            cl_loss = self._infonce(refined_items[pos_items], self_token_items[pos_items], self.cl_temperature)
            loss = loss + self.cl_weight * cl_loss

        return loss

    def _infonce(self, view1, view2, temperature):
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        pos_score = torch.exp((view1 * view2).sum(dim=-1) / temperature)
        denom = torch.exp(view1 @ view2.transpose(0, 1) / temperature).sum(dim=1)
        return torch.mean(-torch.log(pos_score / denom))

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_all, item_all, _ = self.forward()
        u_e = user_all[user]
        scores = torch.matmul(u_e, item_all.transpose(0, 1))
        return scores
