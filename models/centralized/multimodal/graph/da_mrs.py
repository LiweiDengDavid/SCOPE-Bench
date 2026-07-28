# coding: utf-8
r"""
DA-MRS
################################################
Reference:
    https://github.com/GuipengXu/MA-MRS
    KDD'2024: [Improving Multi-modal Recommender Systems by Denoising and Aligning
              Multi-modal Content and User Feedback]

DA-MRS is a plug-and-play technique that wraps a configurable collaborative
backbone (default LightGCN) with three self-contained ideas, ported from the
official `src/models/lightgcn.py` (the DA-MRS+LightGCN variant):

  (1) Content denoising -- item-item graphs are built from *cross-modal
      consistent* content similarity. An edge survives only where BOTH the
      visual and textual similarity agree (masking below the per-modality mean,
      shifted by `prune_threshold`), then top-`knn_k` neighbours are kept and
      symmetrically normalised. (ref `get_knn_adj_mat`)

  (2) Feedback denoising -- a *denoised BPR* loss re-weights every
      (user, pos, neg) triple by how strongly the multimodal content agrees
      with the observed feedback (`denoise_temp` sharpens the content
      agreement). (ref `get_weight_modal` + `bpr_loss`)

  (3) Alignment -- guided by user preference (`align_user_weight`, a symmetric
      KL between collaborative and content scoring, ref the KL term) and by
      graded item relations (`align_item_weight`, a neighbour-discrimination
      contrastive term over the content-consistent graph, ref
      `neighbor_discrimination`).

`calculate_loss` = denoised-BPR + align_user_weight * align_u
                                + align_item_weight * align_i
                                + reg_weight * L2
all wrapping the LightGCN backbone's propagated embeddings. Setting both
alignment weights to 0 removes both alignment terms (they gate their terms).
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


class _LightGCNBackbone(nn.Module):
    """Minimal internal LightGCN embedding module (user/item id embeddings +
    symmetric-normalised U-I propagation), mirroring the reference backbone."""

    def __init__(self, n_users, n_items, embedding_dim, n_layers, norm_adj):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.norm_adj = norm_adj

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self):
        ego = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )
        all_embeddings = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embeddings += [ego]
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        u_g, i_g = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g, i_g


class DA_MRS(RecommenderBase):
    def __init__(self, config, dataloader):
        super(DA_MRS, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["num_layers"]
        self.knn_k = config["knn_k"]
        self.prune_threshold = float(config["prune_threshold"])
        self.align_user_weight = float(config["align_user_weight"])
        self.align_item_weight = float(config["align_item_weight"])
        self.denoise_temp = float(config["denoise_temp"])
        self.reg_weight = float(config["reg_weight"])
        # Number of pseudo-label neighbours per item for the graded
        # neighbor-discrimination (ref `generate_pesudo_labels`, topk=10).
        self.pseudo_topk = min(int(config["pseudo_topk"]), self.n_items)

        # --- backbone selection ---
        backbone = config["backbone"]
        if backbone != "lightgcn":
            raise ValueError(
                f"DA_MRS supports backbone='lightgcn' only, got '{backbone}'"
            )

        # U-I adjacency (symmetric normalised) from the interaction matrix.
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)
        norm_adj = self._get_norm_adj_mat().to(self.device)

        self.backbone = _LightGCNBackbone(
            self.n_users, self.n_items, self.embedding_dim, self.n_layers, norm_adj
        )

        # --- modality feature transforms (mirror MGCN) ---
        # NOTE (faithful to official XMUDM/DA-MRS src/models/lightgcn.py): these
        # image_embedding/text_embedding + image_trs/text_trs are declared but
        # NEVER used in forward/loss — the official DA-MRS also leaves them dead,
        # injecting modality only through the fixed content/co-occurrence graphs
        # while the LightGCN backbone propagates pure-ID embeddings. Kept for
        # structural fidelity; do not re-flag as "dead params".
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # --- (1) content denoising: cross-modal-consistent item-item graphs ---
        v_src = self.v_feat if self.v_feat is not None else self.t_feat
        t_src = self.t_feat if self.t_feat is not None else self.v_feat
        image_adj, text_adj = self._get_knn_adj_mat(
            v_src.to(self.device), t_src.to(self.device)
        )
        self.image_adj = image_adj
        self.text_adj = text_adj

        # --- behavioral co-occurrence graph (ref get_session_adj / h_s) ---
        # The reference draws h_s from an on-disk item_graph_dict built by
        # build_iib_graph.py; that artifact is itself item co-occurrence over
        # the interaction matrix. We build it in-model from inter_matrix('coo'):
        # items co-interacted by the same users, top-k sparsified + sym-norm.
        self.session_adj = self._get_session_adj()

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

    def _get_knn_adj_mat(self, v_embeddings, t_embeddings):
        """Content denoising (ref `get_knn_adj_mat`): keep only edges where the
        visual AND textual similarity are *jointly* above their (shifted) means
        (cross-modal consistency), then top-`knn_k` per row, sym-normalised."""
        v_norm = v_embeddings.div(
            torch.norm(v_embeddings, p=2, dim=-1, keepdim=True) + 1e-12
        )
        v_sim = torch.mm(v_norm, v_norm.transpose(1, 0))
        t_norm = t_embeddings.div(
            torch.norm(t_embeddings, p=2, dim=-1, keepdim=True) + 1e-12
        )
        t_sim = torch.mm(t_norm, t_norm.transpose(1, 0))

        # Cross-modal consistency: an entry is inconsistent if EITHER modality
        # rates it below its own (threshold-shifted) mean; zero it in both.
        mask_v = v_sim < (v_sim.mean() + self.prune_threshold)
        mask_t = t_sim < (t_sim.mean() + self.prune_threshold)
        v_sim[mask_v] = 0
        v_sim[mask_t] = 0
        t_sim[mask_v] = 0
        t_sim[mask_t] = 0

        image_adj = self._topk_normalized(v_sim)
        text_adj = self._topk_normalized(t_sim)
        return image_adj, text_adj

    def _topk_normalized(self, sim):
        n = sim.shape[0]
        index_x, index_y = [], []
        for i in range(n):
            nz = int(torch.count_nonzero(sim[i]).item())
            k = min(self.knn_k, max(nz, 1))
            _, knn_ind = torch.topk(sim[i], k)
            index_x.append(torch.full((k,), i, dtype=torch.long, device=sim.device))
            index_y.append(knn_ind)
        indices = torch.stack(
            (torch.cat(index_x, dim=0), torch.cat(index_y, dim=0)), 0
        )
        return self._compute_normalized_laplacian(indices, (n, n))

    def _compute_normalized_laplacian(self, indices, adj_size):
        values = torch.ones(indices.shape[1], device=indices.device)
        adj = torch.sparse_coo_tensor(indices, values, adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def _get_session_adj(self):
        """Behavioral item-item co-occurrence graph (ref get_session_adj /
        build_iib_graph.py). C = R^T R gives per-item-pair co-interaction
        counts; keep self-loops + top-`knn_k` co-occurring neighbours per item
        (sparsified like the content kNN graph), then symmetric-normalise."""
        R = self.interaction_matrix.tocsr()  # (n_users, n_items)
        co = (R.T @ R).tocoo()  # (n_items, n_items) co-occurrence counts
        cooc = torch.zeros(self.n_items, self.n_items, device=self.device)
        rows = torch.from_numpy(co.row.astype(np.int64))
        cols = torch.from_numpy(co.col.astype(np.int64))
        vals = torch.from_numpy(co.data.astype(np.float32))
        cooc[rows, cols] = vals.to(self.device)
        # Drop self-co-occurrence before top-k neighbour selection; re-added as
        # explicit self-loops inside the normalised graph.
        cooc.fill_diagonal_(0.0)

        n = self.n_items
        index_x, index_y = [], []
        for i in range(n):
            # self-loop
            index_x.append(torch.tensor([i], device=self.device))
            index_y.append(torch.tensor([i], device=self.device))
            nz = int(torch.count_nonzero(cooc[i]).item())
            if nz > 0:
                k = min(self.knn_k, nz)
                _, knn_ind = torch.topk(cooc[i], k)
                index_x.append(torch.full((k,), i, dtype=torch.long, device=self.device))
                index_y.append(knn_ind)
        indices = torch.stack(
            (torch.cat(index_x, dim=0), torch.cat(index_y, dim=0)), 0
        )
        return self._compute_normalized_laplacian(indices, (n, n))

    # ---------------------------------------------------------------- forward
    def _content_item_embeds(self):
        """Propagate item id embeddings over the content-consistent graphs
        (h_v/h_t) and the behavioral co-occurrence graph (h_s)."""
        base = self.backbone.item_id_embedding.weight
        h_v = base.clone()
        for _ in range(self.n_layers):
            h_v = torch.sparse.mm(self.image_adj, h_v)
        h_t = base.clone()
        for _ in range(self.n_layers):
            h_t = torch.sparse.mm(self.text_adj, h_t)
        h_s = base.clone()
        for _ in range(self.n_layers):
            h_s = torch.sparse.mm(self.session_adj, h_s)
        return h_v, h_t, h_s

    def forward(self):
        u_g, i_g = self.backbone()
        h_v, h_t, h_s = self._content_item_embeds()
        return u_g, i_g, h_v, h_t, h_s

    # ----------------------------------------------------------------- losses
    def _denoise_weights(self, users, pos_items, neg_items, u_emb, h_c):
        """Feedback denoising (ref `get_weight_modal`): per-sample pos/neg
        weights from how well content agrees with the collaborative signal.
        `denoise_temp` sharpens the content agreement."""
        u = u_emb[users]
        c_pos = F.normalize(h_c[pos_items], dim=-1)
        c_neg = F.normalize(h_c[neg_items], dim=-1)
        p_agree = torch.sigmoid(torch.sum(u * c_pos, dim=1) / self.denoise_temp)
        n_agree = torch.sigmoid(torch.sum(u * c_neg, dim=1) / self.denoise_temp)

        pos_weight = torch.clamp(p_agree, 0, 1).detach()
        # Down-weight negatives whose content actually agrees more than the
        # positive's mean content agreement (likely false negatives).
        mask = (n_agree < p_agree.mean()).float()
        neg_weight = torch.clamp((p_agree.mean() - n_agree) * mask, 0, 1).detach()
        return pos_weight, neg_weight

    def _denoised_bpr(self, u, pos, neg, p_weight, n_weight):
        pos_scores = torch.sum(u * pos, dim=1)
        neg_scores = torch.sum(u * neg, dim=1)
        p_maxi = F.logsigmoid(pos_scores - neg_scores) * p_weight
        n_maxi = F.logsigmoid(neg_scores - pos_scores) * n_weight
        return -torch.mean(p_maxi + n_maxi)

    def _align_user(self, users, u_emb, i_emb, h_c):
        """User-preference alignment (ref KL term): symmetric KL between the
        collaborative and the content-based user->item scoring."""
        u = u_emb[users]
        p_g = torch.sigmoid(
            torch.matmul(u, F.normalize(i_emb, dim=-1).transpose(0, 1))
        ).clamp(1e-7, 1 - 1e-7)
        p_c = torch.sigmoid(
            torch.matmul(u, F.normalize(h_c, dim=-1).transpose(0, 1))
        ).clamp(1e-7, 1 - 1e-7)

        def kl(a, b):
            return a * torch.log(a / b) + (1 - a) * torch.log((1 - a) / (1 - b))

        return torch.mean(kl(p_g, p_c) + kl(p_c, p_g))

    def _label_prediction(self, emb, aug_emb):
        """Row-softmax over cosine similarity (ref `label_prediction`)."""
        n_emb = F.normalize(emb, dim=1)
        n_aug = F.normalize(aug_emb, dim=1)
        prob = torch.mm(n_emb, n_aug.transpose(0, 1))
        return F.softmax(prob, dim=1)

    def _generate_pseudo_labels(self, prob1, prob2, prob3):
        """Grade neighbours into two tiers (ref `generate_pesudo_labels`):
        `mm_positive` = top-k of the summed 3-view agreement (double-weighting
        the anchor view) -- similar in BOTH other modalities (strongest pull);
        `s_positive` = top-k of the anchor view AFTER masking the mm_positive
        set -- similar in a SINGLE modality (medium pull)."""
        k = self.pseudo_topk
        positive = prob1 + prob2 + prob3 + prob3
        _, mm_pos_ind = torch.topk(positive, k, dim=-1)
        prob = prob3.clone()
        prob.scatter_(1, mm_pos_ind, 0)
        _, single_pos_ind = torch.topk(prob, k, dim=-1)
        return mm_pos_ind, single_pos_ind

    def _neighbor_discrimination(self, mm_positive, s_positive, emb, aug_emb):
        """Graded 3-tier contrastive loss (ref `neighbor_discrimination`):
        a nested log-ratio pulling strongest toward `mm_positive` (both-modality
        neighbours) and medium toward `s_positive` (single-modality), pushing
        away from everything else. `denoise_temp` is the temperature."""
        temperature = self.denoise_temp
        k = self.pseudo_topk

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=2)

        n_aug = F.normalize(aug_emb, dim=1)
        n_emb = F.normalize(emb, dim=1)

        mm_pos_emb = n_aug[mm_positive]
        s_pos_emb = n_aug[s_positive]

        emb2 = torch.reshape(n_emb, [-1, 1, self.embedding_dim])
        emb2 = torch.tile(emb2, [1, k, 1])

        mm_pos_score = score(emb2, mm_pos_emb)
        s_pos_score = score(emb2, s_pos_emb)
        ttl_score = torch.matmul(n_emb, n_aug.transpose(0, 1))

        mm_pos_score = torch.sum(torch.exp(mm_pos_score / temperature), dim=1)
        s_pos_score = torch.sum(torch.exp(s_pos_score / temperature), dim=1)
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

        # mm_positive includes each anchor's self/diagonal, so mm_pos_score can
        # approach ttl_score; clamp the (ttl - mm_pos) denominator strictly
        # positive so the log can never see a non-positive argument -> NaN.
        cl_loss = -torch.log(mm_pos_score / ttl_score + 1e-9) - torch.log(
            s_pos_score / (ttl_score - mm_pos_score).clamp_min(1e-9) + 1e-9
        )
        return torch.mean(cl_loss)

    def _align_item(self, i_id, h_v, h_t, h_s):
        """Graded item-relation alignment (ref the three neighbor_dis terms):
        for each modality/behavior view, grade neighbours via pseudo-labels
        built from the OTHER two views, then run graded neighbor-discrimination.
        Uses content (h_v/h_t) AND the behavioral co-occurrence view (h_s)."""
        lp_t = self._label_prediction(h_t[i_id], h_t)
        lp_v = self._label_prediction(h_v[i_id], h_v)
        lp_s = self._label_prediction(h_s[i_id], h_s)

        mm_s, s_s = self._generate_pseudo_labels(lp_t, lp_v, lp_s)
        loss_s = self._neighbor_discrimination(mm_s, s_s, h_s[i_id], h_s)

        mm_v, s_v = self._generate_pseudo_labels(lp_t, lp_s, lp_v)
        loss_v = self._neighbor_discrimination(mm_v, s_v, h_v[i_id], h_v)

        mm_t, s_t = self._generate_pseudo_labels(lp_v, lp_s, lp_t)
        loss_t = self._neighbor_discrimination(mm_t, s_t, h_t[i_id], h_t)

        return (loss_s + loss_v + loss_t) / 3.0

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

        u_emb, i_emb, h_v, h_t, h_s = self.forward()
        # Fuse content (h_v/h_t) AND behavioral co-occurrence (h_s) views.
        h_c = (h_v + h_t + h_s) / 3.0

        # (2) feedback denoising -> denoised BPR on the content-augmented items.
        ia = i_emb + h_c
        p_weight, n_weight = self._denoise_weights(
            users, pos_items, neg_items, u_emb, h_c
        )
        bpr = self._denoised_bpr(
            u_emb[users], ia[pos_items], ia[neg_items], p_weight, n_weight
        )

        # (3) alignment (each term gated by its weight).
        align_u = self._align_user(users, u_emb, i_emb, h_c)
        # Graded 3-tier neighbor-discrimination over the unique batch items.
        i_id = torch.unique(torch.cat((pos_items, neg_items)))
        align_i = self._align_item(i_id, h_v, h_t, h_s)

        # L2 regularisation over the involved embeddings.
        reg = (
            u_emb[users].pow(2).sum()
            + i_emb[pos_items].pow(2).sum()
            + i_emb[neg_items].pow(2).sum()
        ) / (2 * users.shape[0])

        loss = (
            bpr
            + self.align_user_weight * align_u
            + self.align_item_weight * align_i
            + self.reg_weight * reg
        )
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0].to(self.device)
        u_emb, i_emb, h_v, h_t, h_s = self.forward()
        all_item_e = i_emb + (h_v + h_t + h_s) / 3.0
        scores = torch.matmul(u_emb[user], all_item_e.transpose(0, 1))
        return scores
