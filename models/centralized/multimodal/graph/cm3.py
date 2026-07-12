# coding: utf-8
r"""
CM3
################################################
Reference:
    Xin Zhou, Yongjie Wang, Zhiqi Shen. "CM3: Calibrating Multimodal
    Recommendation." arXiv:2508.01226 (2025).
    https://github.com/enoche/CM3

CM3 revisits the alignment/uniformity trade-off in multimodal recommendation.
Two ideas are ported here from the official repo (models/cm3.py):

  * Calibrated uniformity (``get_sim_mat`` + ``uniformity_i``): the uniformity
    repulsion between items is discounted by their multimodal similarity, so
    items with similar attributes are kept proximal and only dissimilar items
    repel each other.
  * Spherical Bezier fusion (the ``mix_f`` slerp block in ``forward``): the text
    and image projections are interpolated along the great-circle arc on the
    unit hypersphere with a Beta-sampled mixing coefficient, keeping the fused
    feature on the same spherical manifold.

Standard pre-extracted ``t_feat``/``v_feat`` are used (the optional MLLM-feature
variant is out of scope).
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


class CM3(RecommenderBase):
    def __init__(self, config, dataloader):
        super(CM3, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['num_ui_layers']
        self.mm_image_weight = config['mm_image_weight']

        # Calibrated-uniformity knobs.
        self.gamma = config['gamma']
        self.min_sim = config['min_sim']
        self.max_sim = config['max_sim']
        # Spherical Bezier mixing distribution.
        self.alpha = config['alpha']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataloader.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        # Trainer may not call pre_epoch_processing before the first loss; keep
        # the (undropped) normalized adjacency available immediately.
        self.masked_adj = self.norm_adj
        self.cur_epoch = 0

        # Per-user preference vectors for the three streams (image / text / id).
        self.theta = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(self.n_users, 3, 1)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            v_feat_dim = self.v_feat.shape[1]
            self.v_preference = nn.Parameter(nn.init.xavier_normal_(
                torch.empty(self.n_users, self.embedding_dim)))
            self.v_MLP = nn.Linear(v_feat_dim, 4 * self.embedding_dim)
            self.v_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)
            self.image_nn = nn.Linear(v_feat_dim, self.embedding_dim)
            nn.init.xavier_normal_(self.image_nn.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            t_feat_dim = self.t_feat.shape[1]
            self.t_preference = nn.Parameter(nn.init.xavier_normal_(
                torch.empty(self.n_users, self.embedding_dim)))
            self.t_MLP = nn.Linear(t_feat_dim, 4 * self.embedding_dim)
            self.t_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)
            self.text_nn = nn.Linear(t_feat_dim, self.embedding_dim)
            nn.init.xavier_normal_(self.text_nn.weight)

        self.id_preference = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(self.n_users, self.embedding_dim)))

        # Item-item multimodal kNN graph (built in-memory; no dataset-file cache).
        image_adj = text_adj = None
        if self.v_feat is not None:
            _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.mm_adj = image_adj
        if self.t_feat is not None:
            _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        self.mm_adj = self.mm_adj.to(self.device)

        # Cache of the last forward's node embeddings for full_sort_predict.
        # Plain attribute (not a parameter/buffer) so forward() can reassign it
        # with a fresh tensor each pass; initialized so prediction works even if
        # called before any forward.
        self.result_embed = nn.init.xavier_normal_(
            torch.empty(self.n_nodes, self.embedding_dim)).to(self.device)

    # ------------------------------------------------------------------ graphs
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True) + 1e-12)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    @staticmethod
    def get_sim_mat(mm_embeddings, min_v=0.0, max_v=1.0):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True) + 1e-12)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        sim = torch.clamp(sim, min=min_v, max=max_v)
        return sim

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        for (r, c), v in data_dict.items():
            A[r, c] = v
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        self.cur_epoch += 1
        # Edge-dropout is unused when dropout <= 0 (paper default); keep the full
        # normalized adjacency.
        self.masked_adj = self.norm_adj

    # ----------------------------------------------------------------- forward
    def forward(self, adj):
        # Modality streams through their MLP encoders.
        tmp_v_feat = self.v_MLP_1(F.leaky_relu(self.v_MLP(self.v_feat)))
        tmp_t_feat = self.t_MLP_1(F.leaky_relu(self.t_MLP(self.t_feat)))

        # Projections used for the spherical fusion.
        txt_emb = self.text_nn(self.t_feat)
        img_emb = self.image_nn(self.v_feat)

        # --- Spherical Bezier fusion (great-circle slerp on the unit sphere) ---
        # eps must be float32-effective: 1e-8 is below fp32 machine epsilon
        # (~1.19e-7), so `1 - 1e-8 == 1.0` and the clamp would be a no-op. When
        # alignment training drives the text/image projections near-parallel,
        # dot->1 -> acos->0 -> sin(theta)->0 -> a/b become NaN and poison the
        # forward. 1e-6 keeps `1 - eps < 1.0` genuinely true in float32.
        v0, v1, eps = txt_emb, img_emb, 1e-6
        v0_norm = F.normalize(v0, p=2, dim=1)
        v1_norm = F.normalize(v1, p=2, dim=1)
        dot = (v0_norm * v1_norm).sum(dim=1).clamp(-1 + eps, 1 - eps)  # avoid NaN
        theta = torch.acos(dot)
        # Belt-and-suspenders: keep the denominator strictly positive so a nearly
        # zero angle can never divide by zero.
        sin_theta = torch.sin(theta).clamp_min(eps)
        if self.training:
            mix_p = torch.distributions.beta.Beta(self.alpha, self.alpha).sample(
                (v0_norm.size(0),)).to(self.device)
        else:
            # Deterministic expected mixing point (E[Beta(a, a)] = 0.5) at eval so
            # the fusion — and therefore full_sort_predict — is a stable function
            # of the trained weights rather than one arbitrary random slerp draw.
            mix_p = torch.full((v0_norm.size(0),), 0.5, device=self.device)
        a = (torch.sin(mix_p * theta) / sin_theta).unsqueeze(1)
        b = (torch.sin((1 - mix_p) * theta) / sin_theta).unsqueeze(1)
        mix_f = a * v0_norm + b * v1_norm

        # Stacked (user-preference, item-feature) representations per stream.
        rep_uv = torch.cat((self.v_preference, tmp_v_feat), dim=0)
        rep_ut = torch.cat((self.t_preference, tmp_t_feat), dim=0)
        rep_sh = torch.cat((self.id_preference, mix_f), dim=0)
        v_x = torch.cat((F.normalize(rep_uv), F.normalize(rep_ut), F.normalize(rep_sh)), 1)

        # User-item GCN over the three concatenated streams.
        ego_emb = v_x
        all_embeddings = [ego_emb]
        for _ in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_emb)
            ego_emb = side_embeddings
            all_embeddings += [ego_emb]
        representation = torch.stack(all_embeddings, dim=0).sum(dim=0)

        item_rep = representation[self.n_users:]
        vt_rep = representation[:self.n_users]
        v_rep = vt_rep[:, :self.embedding_dim]
        t_rep = vt_rep[:, self.embedding_dim: 2 * self.embedding_dim]
        s_rep = vt_rep[:, 2 * self.embedding_dim:]

        _att2 = F.softmax(self.theta, dim=1)
        user_rep = torch.cat((_att2[:, 0, :] * v_rep,
                              _att2[:, 1, :] * t_rep,
                              _att2[:, 2, :] * s_rep), dim=1)

        # Item-item multimodal graph propagation.
        h = item_rep
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        item_rep = item_rep + h

        # NOTE (faithful to official enoche/CM3): scoring uses the UN-normalized
        # result_embed, while only the tensors returned to the loss are
        # L2-normalized. The official CM3 has this same train/inference
        # normalization asymmetry; kept for fidelity (do not "fix" to normalized).
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        return F.normalize(user_rep, dim=-1), F.normalize(item_rep, dim=-1), mix_f

    # -------------------------------------------------------------------- loss
    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    @staticmethod
    def uniformity_i(x, sim, t=2):
        # Calibrated uniformity: repulsion discounted by multimodal similarity.
        d = (torch.pdist(x, p=2).pow(2) - 2 + 2 * sim).mul(-t)
        return d.exp().mean().log()

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        if len(interaction) >= 3:
            neg_items = interaction[2]
        else:
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=pos_items.device)

        ua_embeddings, ia_embeddings, mix_feat = self.forward(self.masked_adj)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        align = self.alignment(u_g_embeddings, pos_i_g_embeddings)

        # The calibrated-uniformity term uses torch.pdist / torch.combinations,
        # which require at least two rows; a degenerate 1-sample batch would make
        # them empty and turn the loss into NaN. Fall back to alignment-only.
        if neg_items.size(0) < 2:
            return align

        neg_i_g_embeddings = ia_embeddings[neg_items]
        sim_mt = self.get_sim_mat(mix_feat[neg_items], min_v=self.min_sim, max_v=self.max_sim)
        idx = torch.combinations(torch.arange(neg_items.size(0), device=neg_items.device))
        sim = sim_mt[idx[:, 0], idx[:, 1]]

        uniform = self.gamma * (self.uniformity(u_g_embeddings)
                                + self.uniformity_i(neg_i_g_embeddings, sim)) / 2
        return align + uniform

    def full_sort_predict(self, interaction):
        # Recompute a fresh eval-mode forward so scoring reflects the current
        # weights deterministically. Reading self.result_embed without this would
        # score on a stale cache left by the last *training* batch's forward
        # (captured under a single random slerp draw), not an eval representation.
        with torch.no_grad():
            self.forward(self.norm_adj)
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix
