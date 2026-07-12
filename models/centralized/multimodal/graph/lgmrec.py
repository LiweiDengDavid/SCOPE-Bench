# coding: utf-8
r"""
LGMRec
################################################
Reference:
    https://github.com/georgeguo-cn/LGMRec
    AAAI'2024: [LGMRec: Local and Global Graph Learning for Multimodal Recommendation]

Ported from the official repo (models/lgmrec.py). LGMRec disentangles a
*local* graph embedding -- LightGCN-style collaborative U-I propagation
(``n_ui_layers``) plus per-modality item-item propagation (``n_mm_layers``) --
from a *global* hypergraph embedding produced by a hyperedge-dependency module
(``hyper_num`` hyperedges, ``n_hyper_layer`` HGNN layers, dropout ``keep_rate``).
The two are fused with weight ``alpha``. Training loss is BPR over the fused
embeddings + ``cl_weight`` * hypergraph contrastive loss + ``reg_weight`` L2.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


class HGNNLayer(nn.Module):
    """Hypergraph neural-network layer (ported verbatim from the LGMRec repo).

    Propagates item embeddings through the item-hyperedge incidence matrix and
    reads back both user- and item-side global embeddings.
    """

    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()
        self.h_layer = n_hyper_layer

    def forward(self, i_hyper, u_hyper, embeds):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.transpose(0, 1), i_ret)
            i_ret = torch.mm(i_hyper, lat)
            u_ret = torch.mm(u_hyper, lat)
        return u_ret, i_ret


class LGMRec(RecommenderBase):
    def __init__(self, config, dataloader):
        super(LGMRec, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.embedding_dim = config['embedding_size']
        # The repo carries a separate feat_embed_dim / cf_model; keep this a
        # faithful LightGCN local encoder and tie the modal-projection width to
        # the id-embedding width (no extra YAML knobs introduced).
        self.feat_embed_dim = config['embedding_size']
        self.n_mm_layer = config['n_mm_layers']
        self.n_ui_layers = config['num_ui_layers']
        self.n_hyper_layer = config['n_hyper_layer']
        self.hyper_num = config['hyper_num']
        self.keep_rate = config['keep_rate']
        self.alpha = config['alpha']
        self.cl_weight = config['cl_weight']
        self.reg_weight = float(config['reg_weight'])
        # Temperature for the gumbel-softmax hyperedge assignment AND the
        # hypergraph contrastive loss (kept in YAML per the no-literals rule).
        self.tau = float(config['tau'])

        self.n_nodes = self.n_users + self.n_items

        self.hgnnLayer = HGNNLayer(self.n_hyper_layer)

        # load dataset info
        self.interaction_matrix = dataloader.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tensor(
            self.interaction_matrix, torch.Size((self.n_users, self.n_items))
        )
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        # init user and item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.drop = nn.Dropout(p=1 - self.keep_rate)

        # load item modal features and define hyperedge embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.v_hyper = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.hyper_num))
            )
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.t_hyper = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.hyper_num))
            )

    def scipy_matrix_to_sparse_tensor(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse_coo_tensor(i, data, shape, dtype=torch.float32).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(
            dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz))
        )
        for (r, c), v in data_dict.items():
            A[r, c] = v
        # symmetric normalization
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tensor(
            L, torch.Size((self.n_nodes, self.n_nodes))
        )

    # collaborative graph embedding (LightGCN)
    def cge(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        cge_embs = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            cge_embs += [ego_embeddings]
        cge_embs = torch.stack(cge_embs, dim=1)
        cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs

    # modality graph embedding
    def mge(self, modal='v'):
        if modal == 'v':
            item_feats = self.image_trs(self.image_embedding.weight)
        else:
            item_feats = self.text_trs(self.text_embedding.weight)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]
        mge_feats = torch.cat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
        return mge_feats

    def _hyper_assign(self, logits):
        """Hyperedge assignment: stochastic Gumbel-softmax during training, a
        deterministic softmax relaxation at inference so full_sort_predict is a
        stable function of the trained weights. The official LGMRec samples fresh
        Gumbel noise on every forward (including eval), which makes eval metrics
        and HPO objective comparisons noisy; inference is made deterministic here
        (same reproducible-eval choice as MIG-GT/CM3)."""
        if self.training:
            return F.gumbel_softmax(logits, self.tau, dim=1, hard=False)
        return F.softmax(logits / self.tau, dim=1)

    def forward(self):
        # hyperedge dependencies constructing
        iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)
        uv_hyper = torch.sparse.mm(self.adj, iv_hyper)
        iv_hyper = self._hyper_assign(iv_hyper)
        uv_hyper = self._hyper_assign(uv_hyper)

        it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
        ut_hyper = torch.sparse.mm(self.adj, it_hyper)
        it_hyper = self._hyper_assign(it_hyper)
        ut_hyper = self._hyper_assign(ut_hyper)

        # CGE: collaborative graph embedding
        cge_embs = self.cge()

        # MGE: modality graph embedding
        v_feats = self.mge('v')
        t_feats = self.mge('t')
        mge_embs = F.normalize(v_feats) + F.normalize(t_feats)
        # local embeddings = collaborative + modality
        lge_embs = cge_embs + mge_embs

        # GHE: global hypergraph embedding
        uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(
            self.drop(iv_hyper), self.drop(uv_hyper), cge_embs[self.n_users:]
        )
        ut_hyper_embs, it_hyper_embs = self.hgnnLayer(
            self.drop(it_hyper), self.drop(ut_hyper), cge_embs[self.n_users:]
        )
        av_hyper_embs = torch.cat([uv_hyper_embs, iv_hyper_embs], dim=0)
        at_hyper_embs = torch.cat([ut_hyper_embs, it_hyper_embs], dim=0)
        ghe_embs = av_hyper_embs + at_hyper_embs

        # local embeddings + alpha * global embeddings
        all_embs = lge_embs + self.alpha * F.normalize(ghe_embs)

        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)
        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs]

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.transpose(0, 1)) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def reg_loss(self, *embs):
        reg = 0.0
        for emb in embs:
            reg = reg + torch.norm(emb, p=2)
        reg = reg / embs[-1].shape[0]
        return reg

    def calculate_loss(self, interaction):
        # Sample negatives internally when the trainer supplies only (users, pos).
        if len(interaction) == 3:
            users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        elif len(interaction) == 2:
            users, pos_items = interaction[0], interaction[1]
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=pos_items.device)
        else:
            raise ValueError(f"Unsupported interaction format with {len(interaction)} elements")

        ua_embeddings, ia_embeddings, hyper_embeddings = self.forward()

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        uv_embs, iv_embs, ut_embs, it_embs = hyper_embeddings
        batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + \
            self.ssl_triple_loss(iv_embs[pos_items], it_embs[pos_items], it_embs)

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = batch_bpr_loss + self.cl_weight * batch_hcl_loss + self.reg_weight * batch_reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.transpose(0, 1))
        return scores
