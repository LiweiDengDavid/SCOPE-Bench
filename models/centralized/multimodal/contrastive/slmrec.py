# coding: utf-8
"""SLMRec multimodal self-supervised recommendation model."""

import logging
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import scipy.sparse as sp

from core.base import RecommenderBase

logger = logging.getLogger("nexusrec")

## Only visual + text features
##

class SLMRec(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)
        self.infonce_criterion = nn.CrossEntropyLoss()
        self.__init_weight(dataloader)

    def _fuse_user_item_views(self, user_views, item_views):
        """Fuse user/item modality views with the configured fusion module."""
        user = self.embedding_user_after_GCN(self.mm_fusion(user_views))
        item = self.embedding_item_after_GCN(self.mm_fusion(item_views))
        return user, item

    @staticmethod
    def _normalize_views(view_a, view_b):
        """Normalize two SSL views before contrastive scoring."""
        view_a = F.normalize(view_a, dim=1)
        view_b = F.normalize(view_b, dim=1)
        return view_a, view_b

    def _ssl_cross_entropy(self, view_a, view_b, criterion=None):
        view_a, view_b = self._normalize_views(view_a, view_b)
        logits = torch.mm(view_a, view_b.T) / self.ssl_temp
        labels = torch.arange(view_b.shape[0], device=self.device, dtype=torch.long)
        loss_fn = criterion if criterion is not None else self.ssl_criterion
        return loss_fn(logits, labels)

    def _fuse_ssl_embeddings(self, user_views_a, item_views_a, user_views_b, item_views_b):
        users_a, items_a = self._fuse_user_item_views(user_views_a, item_views_a)
        users_b, items_b = self._fuse_user_item_views(user_views_b, item_views_b)
        return users_a, items_a, users_b, items_b

    def __init_weight(self, dataloader):
        self.num_users = self.n_users
        self.num_items = self.n_items
        self.embedding_size = self.config['embedding_size']
        self.latent_dim = self.embedding_size  # Latent dimension equals embedding size
        self.num_layers = self.config['num_layers']
        self.fusion_module = self.config['fusion_method']
        self.memory_efficient = self.config['memory_efficient']

        self.create_u_embeding_i()

        self.all_items = self.all_users = None

        train_interactions = dataloader.inter_matrix(form='csr').astype(np.float32)
        coo = self.create_adj_mat(train_interactions).tocoo()
        indices = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        self.norm_adj = torch.sparse_coo_tensor(indices, torch.FloatTensor(coo.data), coo.shape, dtype=torch.float32)
        self.norm_adj = self.norm_adj.to(self.device)
        self.f = nn.Sigmoid()

        if self.config["ssl_task"] == "FAC":
            # Fine and Coarse
            self.g_i_iv = nn.Linear(self.latent_dim, self.latent_dim)
            self.g_v_iv = nn.Linear(self.latent_dim, self.latent_dim)
            self.g_iv_iva = nn.Linear(self.latent_dim, self.latent_dim)
            self.g_iva_ivat = nn.Linear(self.latent_dim, self.latent_dim // 2)
            self.g_t_ivat = nn.Linear(self.latent_dim, self.latent_dim // 2)
            nn.init.xavier_uniform_(self.g_i_iv.weight)
            nn.init.xavier_uniform_(self.g_v_iv.weight)
            nn.init.xavier_uniform_(self.g_iv_iva.weight)
            nn.init.xavier_uniform_(self.g_iva_ivat.weight)
            nn.init.xavier_uniform_(self.g_t_ivat.weight)
            self.ssl_temp = float(self.config["temperature"])
        elif self.config["ssl_task"] in ["FD", "FD+FM"]:
            # Feature dropout
            self.ssl_criterion = nn.CrossEntropyLoss()
            self.ssl_temp = float(self.config["temperature"])
            self.dropout_rate = float(self.config["dropout_rate"])
            self.dropout = nn.Dropout(p=self.dropout_rate)
        elif self.config["ssl_task"] == "FM":
            # Feature Masking
            self.ssl_criterion = nn.CrossEntropyLoss()
            self.ssl_temp = float(self.config["temperature"])

    def compute(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        if self.v_feat is not None:
            self.v_dense_emb = self.v_dense(self.v_feat)  # v=>id
        if self.t_feat is not None:
            self.t_dense_emb = self.t_dense(self.t_feat)  # t=>id

        def compute_graph(u_emb, i_emb):
            all_emb = torch.cat([u_emb, i_emb])
            if self.memory_efficient:
                # Memory-efficient mode: running mean instead of stacking all layers.
                # Equivalent to the standard mean-pool, but uses O(1) extra memory.
                current_emb = all_emb
                running_sum = all_emb  # layer-0 contribution
                for _ in range(self.num_layers):
                    current_emb = torch.sparse.mm(self.norm_adj, current_emb)
                    running_sum = running_sum + current_emb
                light_out = running_sum / (self.num_layers + 1)
            else:
                # Standard mode: store embeddings from all layers
                embs = [all_emb]
                g_droped = self.norm_adj
                for _ in range(self.num_layers):
                    all_emb = torch.sparse.mm(g_droped, all_emb)
                    embs.append(all_emb)
                # Compute mean in a more memory-friendly way
                embs = torch.stack(embs, dim=1)
                light_out = torch.mean(embs, dim=1)
                # Free intermediate variables to release memory
                del embs
            return light_out

        self.i_emb = compute_graph(users_emb, items_emb)
        self.i_emb_u, self.i_emb_i = torch.split(self.i_emb, [self.num_users, self.num_items])
        self.v_emb = compute_graph(users_emb, self.v_dense_emb)
        self.v_emb_u, self.v_emb_i = torch.split(self.v_emb, [self.num_users, self.num_items])
        if self.t_feat is not None:
            self.t_emb = compute_graph(users_emb, self.t_dense_emb)
            self.t_emb_u, self.t_emb_i = torch.split(self.t_emb, [self.num_users, self.num_items])

        return self._fuse_user_item_views(
            [self.i_emb_u, self.v_emb_u, self.t_emb_u],
            [self.i_emb_i, self.v_emb_i, self.t_emb_i],
        )

    def feature_dropout(self, users_idx, items_idx):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        v_dense = self.v_dense_emb
        t_dense = self.t_dense_emb

        def compute_graph(u_emb, i_emb):
            all_emb = torch.cat([u_emb, i_emb])
            ego_emb_sub_1 = all_emb
            ego_emb_sub_2 = all_emb
            # embs = [all_emb]
            embs_sub_1 = [ego_emb_sub_1]
            embs_sub_2 = [ego_emb_sub_2]

            g_droped = self.norm_adj

            for _ in range(self.num_layers):
                ego_emb_sub_1 = self.dropout(torch.sparse.mm(g_droped, ego_emb_sub_1))
                ego_emb_sub_2 = self.dropout(torch.sparse.mm(g_droped, ego_emb_sub_2))
                embs_sub_1.append(ego_emb_sub_1)
                embs_sub_2.append(ego_emb_sub_2)
            embs_sub_1 = torch.stack(embs_sub_1, dim=1)
            embs_sub_2 = torch.stack(embs_sub_2, dim=1)

            light_out_sub_1 = torch.mean(embs_sub_1, dim=1)
            light_out_sub_2 = torch.mean(embs_sub_2, dim=1)

            users_sub_1, items_sub_1 = torch.split(light_out_sub_1, [self.num_users, self.num_items])
            users_sub_2, items_sub_2 = torch.split(light_out_sub_2, [self.num_users, self.num_items])
            return users_sub_1[users_idx], items_sub_1[items_idx], users_sub_2[users_idx], items_sub_2[items_idx]

        i_emb_u_sub_1, i_emb_i_sub_1, i_emb_u_sub_2, i_emb_i_sub_2 = compute_graph(users_emb, items_emb)
        v_emb_u_sub_1, v_emb_i_sub_1, v_emb_u_sub_2, v_emb_i_sub_2 = compute_graph(users_emb, v_dense)
        t_emb_u_sub_1, t_emb_i_sub_1, t_emb_u_sub_2, t_emb_i_sub_2 = compute_graph(users_emb, t_dense)

        users_sub_1, items_sub_1, users_sub_2, items_sub_2 = self._fuse_ssl_embeddings(
            [i_emb_u_sub_1, v_emb_u_sub_1, t_emb_u_sub_1],
            [i_emb_i_sub_1, v_emb_i_sub_1, t_emb_i_sub_1],
            [i_emb_u_sub_2, v_emb_u_sub_2, t_emb_u_sub_2],
            [i_emb_i_sub_2, v_emb_i_sub_2, t_emb_i_sub_2],
        )

        return self._ssl_cross_entropy(users_sub_1, users_sub_2) + self._ssl_cross_entropy(items_sub_1, items_sub_2)

    def feature_masking(self, users_idx, items_idx, dropout=False):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        rand_range = 3
        rand_idx1 = np.random.randint(rand_range)
        rand_idx2 = 0
        while True:
            rand_idx2 = np.random.randint(rand_range)
            if rand_idx2 != rand_idx1:
                break

        v_dense = self.v_dense_emb
        t_dense = self.t_dense_emb

        def compute_graph(u_emb, i_emb, idx):
            all_emb_1 = torch.cat([u_emb, i_emb if rand_idx1 != idx else self._item_feat_zeros])
            all_emb_2 = torch.cat([u_emb, i_emb if rand_idx2 != idx else self._item_feat_zeros])
            ego_emb_sub_1 = all_emb_1
            ego_emb_sub_2 = all_emb_2
            embs_sub_1 = [ego_emb_sub_1]
            embs_sub_2 = [ego_emb_sub_2]
            g_droped = self.norm_adj

            for _ in range(self.num_layers):
                ego_emb_sub_1 = torch.sparse.mm(g_droped, ego_emb_sub_1)
                ego_emb_sub_2 = torch.sparse.mm(g_droped, ego_emb_sub_2)
                if dropout:
                    ego_emb_sub_1 = self.dropout(ego_emb_sub_1)
                    ego_emb_sub_2 = self.dropout(ego_emb_sub_2)
                embs_sub_1.append(ego_emb_sub_1)
                embs_sub_2.append(ego_emb_sub_2)
            embs_sub_1 = torch.stack(embs_sub_1, dim=1)
            embs_sub_2 = torch.stack(embs_sub_2, dim=1)

            light_out_sub_1 = torch.mean(embs_sub_1, dim=1)
            light_out_sub_2 = torch.mean(embs_sub_2, dim=1)

            users_sub_1, items_sub_1 = torch.split(light_out_sub_1, [self.num_users, self.num_items])
            users_sub_2, items_sub_2 = torch.split(light_out_sub_2, [self.num_users, self.num_items])
            return users_sub_1[users_idx], items_sub_1[items_idx], users_sub_2[users_idx], items_sub_2[items_idx]

        # Modality view indices are contiguous in [0, 3): for the 3-modal
        # [i, v, t] case any single view can be masked.
        v_emb_u_sub_1, v_emb_i_sub_1, v_emb_u_sub_2, v_emb_i_sub_2 = compute_graph(users_emb, v_dense, idx=0)
        i_emb_u_sub_1, i_emb_i_sub_1, i_emb_u_sub_2, i_emb_i_sub_2 = compute_graph(users_emb, items_emb, idx=2)
        t_emb_u_sub_1, t_emb_i_sub_1, t_emb_u_sub_2, t_emb_i_sub_2 = compute_graph(users_emb, t_dense, idx=1)

        users_sub_1, items_sub_1, users_sub_2, items_sub_2 = self._fuse_ssl_embeddings(
            [i_emb_u_sub_1, v_emb_u_sub_1, t_emb_u_sub_1],
            [i_emb_i_sub_1, v_emb_i_sub_1, t_emb_i_sub_1],
            [i_emb_u_sub_2, v_emb_u_sub_2, t_emb_u_sub_2],
            [i_emb_i_sub_2, v_emb_i_sub_2, t_emb_i_sub_2],
        )

        return self._ssl_cross_entropy(users_sub_1, users_sub_2) + self._ssl_cross_entropy(items_sub_1, items_sub_2)

    def fac(self, idx):
        x_i_iv = self.g_i_iv(self.i_emb_i[idx])
        x_v_iv = self.g_v_iv(self.v_emb_i[idx])
        v_loss = self._ssl_cross_entropy(x_i_iv, x_v_iv, criterion=self.infonce_criterion)
        x_iv_iva = self.g_iv_iva(x_i_iv)
        x_iva_ivat = self.g_iva_ivat(x_iv_iva)
        x_t_ivat = self.g_t_ivat(self.t_emb_i[idx])
        return v_loss + self._ssl_cross_entropy(x_iva_ivat, x_t_ivat, criterion=self.infonce_criterion)

    def full_sort_predict(self, interaction):
        users = interaction[0]
        # Recompute embeddings for each eval call so ranking uses the current model state.
        all_users, all_items = self.compute()
        users_emb = all_users[users]
        scores = torch.matmul(users_emb, all_items.t())
        return self.f(scores)

    def _get_embeddings(self, users, pos_items, neg_items):
        self.all_users, self.all_items = self.compute()
        users_emb = self.all_users[users]
        pos_emb = self.all_items[pos_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)

        if neg_items is None:
            neg_emb_ego = neg_emb = None
        else:
            neg_emb = self.all_items[neg_items]
            neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def calculate_loss(self, interaction):
        if len(interaction) >= 2:
            users, pos = interaction[0], interaction[1]
        else:
            raise ValueError(f"Unsupported interaction format with {len(interaction)} elements")
        
        main_loss = self.infonce(users, pos)
        ssl_loss = self.compute_ssl(users, pos)
        alpha = self.config['alpha']
        
        if isinstance(alpha, str):
            alpha = float(alpha)
        
        return main_loss + alpha * ssl_loss

    def compute_ssl(self, users, items):
        if self.config["ssl_task"] == "FAC":
            return self.fac(items)
        elif self.config["ssl_task"] == "FD":
            return self.feature_dropout(users.long(), items.long())
        elif self.config["ssl_task"] == "FM":
            return self.feature_masking(users.long(), items.long())
        elif self.config["ssl_task"] == "FD+FM":
            return self.feature_masking(users.long(), items.long(), dropout=True)
        raise ValueError(f"Unsupported ssl_task: {self.config['ssl_task']}")

    def forward(self, *args, **kwargs):
        # SLMRec never routes through a generic forward(): training uses
        # calculate_loss (infonce + compute_ssl) and evaluation uses the
        # overridden full_sort_predict. forward only exists to satisfy the
        # RecommenderBase @abstractmethod contract.
        raise NotImplementedError(
            "SLMRec has no generic forward(); use calculate_loss / full_sort_predict."
        )

    def mm_fusion(self, reps: list):
        # The post-GCN linear layers are sized latent_dim*(mul_modal_cnt+1), so
        # they require concatenation and reject any other fusion method early.
        if self.fusion_module != "concat":
            raise ValueError(
                "SLMRec supports fusion_method 'concat' only "
                f"(post-GCN layers are concat-sized); got {self.fusion_module!r}."
            )
        return torch.cat(reps, dim=1)

    def infonce(self, users, pos):
        users_emb, pos_emb, _, _, _, _ = self._get_embeddings(users.long(), pos.long(), None)
        return self._ssl_cross_entropy(users_emb, pos_emb, criterion=self.infonce_criterion)

    def _align_feat_to_items(self, feat, name):
        """Pad or truncate feature matrix to num_items rows, move to device, normalize.

        Returns (feat_tensor, nn.Linear projector).
        """
        if feat.shape[0] != self.num_items:
            logger.warning("%s rows %d != num_items %d", name, feat.shape[0], self.num_items)
            if feat.shape[0] < self.num_items:
                padding = torch.zeros(self.num_items - feat.shape[0], feat.shape[1], device=feat.device)
                feat = torch.cat([feat, padding], dim=0)
                logger.warning("Padded %s to %s", name, feat.shape)
            else:
                feat = feat[:self.num_items]
                logger.warning("Truncated %s to %s", name, feat.shape)
        feat = feat.to(self.device)
        feat = F.normalize(feat, dim=1)
        dense = nn.Linear(feat.shape[1], self.latent_dim)
        nn.init.xavier_uniform_(dense.weight)
        return feat, dense

    def create_u_embeding_i(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.config["init"] == "xavier":
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        elif self.config["init"] == "normal":
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)

        # load features, updated by enoche
        mul_modal_cnt = 0
        if self.v_feat is not None:
            self.v_feat, self.v_dense = self._align_feat_to_items(self.v_feat, "v_feat")
            mul_modal_cnt += 1
        if self.t_feat is not None:
            self.t_feat, self.t_dense = self._align_feat_to_items(self.t_feat, "t_feat")
            mul_modal_cnt += 1

        self.item_feat_dim = self.latent_dim * (mul_modal_cnt + 1)

        self.embedding_item_after_GCN = nn.Linear(self.item_feat_dim, self.latent_dim)
        self.embedding_user_after_GCN = nn.Linear(self.item_feat_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_item_after_GCN.weight)
        nn.init.xavier_uniform_(self.embedding_user_after_GCN.weight)
        self.register_buffer("_item_feat_zeros", torch.zeros(self.num_items, self.latent_dim))

    def create_adj_mat(self, interaction_csr):
        user_np, item_np = interaction_csr.nonzero()
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv.dot(adj).tocoo()

        adj_type = self.config['adj_type']
        logger.debug("Building adjacency matrix: adj_type=%s", adj_type)
        if adj_type == 'plain':
            adj_matrix = adj_mat
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1)) + 1e-08    # avoid RuntimeWarning: divide by zero encountered in power
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])

        return adj_matrix
