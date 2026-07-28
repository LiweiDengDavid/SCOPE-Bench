import math
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.base import RecommenderBase


class IDFREE(RecommenderBase):
    supports_multi_negatives = True  # calculate_loss stacks all interaction[2:] rows

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)
        self.current_epoch = 0
        self.embedding_dim = config['embedding_size']
        self.gcn_layers = config['num_layers']
        self.use_id = config['use_id']
        self.knn_k = config['knn_k']
        self.dropout = config['dropout_rate']
        self.temperature = config['temperature']
        self.asg = config['asg']
        self.auige = config['auige']
        self.m_alpha = config['m_alpha']
        self.pe = config['pe']
        self.lambda_align = float(config['lambda_align'])
        self.save_intermediate = config['save_intermediate']
        self.save_path = config['paths']['checkpoint'].format(config['model'], config['dataset'])
        self.use_text = config['use_text']
        self.use_image = config['use_image']

        self.intermediate_path = os.path.abspath(self.save_path)
        if not os.path.exists(self.intermediate_path):
            os.makedirs(self.intermediate_path)

        self.n_nodes = self.n_users + self.n_items
        interaction_matrix = dataloader.inter_matrix(form='coo').astype(np.float32)
        self.ui_indices = torch.LongTensor(np.vstack((interaction_matrix.row, interaction_matrix.col))).to(self.device)
        self.base_adj = self.get_base_adj(self.ui_indices.clone())
        if self.use_id:
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
            self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_id_embedding.weight)
        user_visual_adj, item_visual_adj = self._build_modal_knn(self.v_feat, self.use_image, "visual")
        user_text_adj, item_text_adj = self._build_modal_knn(self.t_feat, self.use_text, "text")
        self.base_uu, self.base_ii = self._select_modal_graphs(
            user_visual_adj,
            item_visual_adj,
            user_text_adj,
            item_text_adj,
        )

        self._cached_user_embedding = None
        self._cached_item_embedding = None
        self.i_pe = PositionalEncoding(self.embedding_dim, self.n_items, self.device)
        self.u_pe = PositionalEncoding(self.embedding_dim, self.n_users, self.device)
        if self.use_text or self.use_image:
            self.i_edge_predictor = EdgeMLP(self.base_ii, self.embedding_dim, self.device)
            self.u_edge_predictor = EdgeMLP(self.base_uu, self.embedding_dim, self.device)
        if self.v_feat is not None and self.use_image:
            self.v_mu = MLP(self.v_feat.size(-1), self.embedding_dim)
        if self.t_feat is not None and self.use_text:
            self.t_mu = MLP(self.t_feat.size(-1), self.embedding_dim)

    def _build_modal_knn(self, features, enabled, name):
        if features is None or not enabled:
            return None, None
        item_embeddings = nn.Embedding.from_pretrained(features, freeze=False)
        user_embeddings = self._mean_user_embeddings(features)
        if name == "visual":
            self.v_feat_i = item_embeddings
            self.v_feat_u = user_embeddings
        elif name == "text":
            self.t_feat_i = item_embeddings
            self.t_feat_u = user_embeddings
        else:
            raise ValueError(f"Unknown modality name: {name!r}. Expected 'visual' or 'text'.")
        return self.get_knn_adj(user_embeddings), self.get_knn_adj(item_embeddings.weight)

    def _select_modal_graphs(self, user_visual_adj, item_visual_adj, user_text_adj, item_text_adj):
        if self.use_text and self.use_image:
            return user_visual_adj + user_text_adj, item_visual_adj + item_text_adj
        if self.use_text:
            return user_text_adj, item_text_adj
        if self.use_image:
            return user_visual_adj, item_visual_adj
        raise ValueError("At least one of use_text or use_image must be True.")

    def _mean_user_embeddings(self, embeddings):
        rows = self.ui_indices[0]
        cols = self.ui_indices[1]
        item_embeddings = embeddings[cols]
        user_embedding_sum = torch.zeros((self.n_users, embeddings.size(-1)), device=self.device)
        user_interaction_count = torch.zeros(self.n_users, device=self.device)
        user_embedding_sum.index_add_(0, rows, item_embeddings)
        user_interaction_count.index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
        user_embedding_mean = user_embedding_sum / user_interaction_count.unsqueeze(1)
        user_embedding_mean = torch.nan_to_num(user_embedding_mean, nan=0.0, posinf=0.0, neginf=0.0)
        del user_embedding_sum, user_interaction_count
        return user_embedding_mean

    def get_base_adj(self, ui_indices):
        adj_size = torch.Size((self.n_nodes, self.n_nodes))
        ui_indices[1] += self.n_users
        ui_graph = torch.sparse_coo_tensor(ui_indices, torch.ones_like(self.ui_indices[0], dtype=torch.float32),
                                           adj_size, device=self.device)
        iu_graph = ui_graph.T
        base_adj = ui_graph + iu_graph
        return base_adj

    def get_aug_adj_mat(self, base_adj, uu_graph, ii_graph):
        if uu_graph is None and ii_graph is None:
            return base_adj
        adj_size = torch.Size((self.n_nodes, self.n_nodes))

        uu_graph = torch.sparse_coo_tensor(uu_graph._indices(),
                                           uu_graph._values(), adj_size, device=self.device)
        ii_graph = torch.sparse_coo_tensor(ii_graph._indices() + self.n_users,
                                           ii_graph._values(), adj_size, device=self.device)
        aug_adj = uu_graph + base_adj + ii_graph
        return aug_adj

    def get_knn_adj(self, embeddings):
        context_norm = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        knn_val, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        indices0 = torch.arange(knn_ind.size(0)).unsqueeze(1).expand(-1, self.knn_k).to(self.device)
        indices = torch.stack([indices0.flatten(), knn_ind.flatten()], dim=0)
        adj = torch.sparse_coo_tensor(indices, knn_val.flatten().squeeze(), sim.size())
        return adj

    def _normalize_laplacian(self, adj):
        indices = adj._indices()
        values = adj._values()
        row = indices[0]
        col = indices[1]
        rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
        d_inv_sqrt = torch.clamp(torch.pow(rowsum, -0.5), 0.0, 10.0)
        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = values * row_inv_sqrt * col_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj.shape)

    def sample_adj(self, adj, dropout):
        edge_value = adj._values()
        edge_value[torch.isnan(edge_value)] = 0.
        degree_len = int(edge_value.size(0) * (1. - dropout))
        degree_idx = torch.multinomial(edge_value, degree_len)
        keep_indices = adj._indices()[:, degree_idx]
        new_adj = torch.sparse_coo_tensor(keep_indices, edge_value[degree_idx], adj.shape)
        return self._normalize_laplacian(new_adj)

    def simple_gcn(self, ego_embeddings, norm_adj):
        all_embeddings = [ego_embeddings]
        for _ in range(self.gcn_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        return all_embeddings

    def _project_modal_features(self):
        user_text = item_text = user_visual = item_visual = None
        if self.use_image:
            item_visual = self.v_mu(self.v_feat_i.weight)
            user_visual = self.v_mu(self.v_feat_u)
            if self.pe:
                item_visual = self.i_pe(item_visual)
                user_visual = self.u_pe(user_visual)
        if self.use_text:
            item_text = self.t_mu(self.t_feat_i.weight)
            user_text = self.t_mu(self.t_feat_u)
            if self.pe:
                item_text = self.i_pe(item_text)
                user_text = self.u_pe(user_text)
        return user_text, item_text, user_visual, item_visual

    def _fuse_modal_embeddings(self, user_text, item_text, user_visual, item_visual):
        if self.use_text and self.use_image:
            user_embedding = self.m_alpha * user_text + (1 - self.m_alpha) * user_visual
            item_embedding = self.m_alpha * item_text + (1 - self.m_alpha) * item_visual
        elif self.use_text:
            user_embedding = user_text
            item_embedding = item_text
        else:
            user_embedding = user_visual
            item_embedding = item_visual
        if self.use_id:
            user_embedding = user_embedding + self.user_embedding.weight
            item_embedding = item_embedding + self.item_id_embedding.weight
        return user_embedding, item_embedding

    def _adaptive_graph(self, user_text, item_text, user_visual, item_visual):
        if not self.auige or not (self.use_text or self.use_image):
            return self.base_adj
        if not self.asg:
            return self.get_aug_adj_mat(self.base_adj, self.base_uu, self.base_ii).coalesce()

        if self.training:
            if self.use_text and self.use_image:
                item_adj = self.i_edge_predictor(item_text, item_visual)
                user_adj = self.u_edge_predictor(user_text, user_visual)
            elif self.use_text:
                item_adj = self.i_edge_predictor(item_text, item_text)
                user_adj = self.u_edge_predictor(user_text, user_text)
            else:
                item_adj = self.i_edge_predictor(item_visual, item_visual)
                user_adj = self.u_edge_predictor(user_visual, user_visual)
        else:
            if self._cached_item_embedding is None or self._cached_user_embedding is None:
                raise RuntimeError("_adaptive_graph called in eval mode before any forward() pass.")
            item_adj = self.get_knn_adj(self._cached_item_embedding)
            user_adj = self.get_knn_adj(self._cached_user_embedding)
        return self.get_aug_adj_mat(self.base_adj, user_adj, item_adj).coalesce()

    def forward(self, training):
        u_t_mu, i_t_mu, u_v_mu, i_v_mu = self._project_modal_features()
        u_embedding, i_embedding = self._fuse_modal_embeddings(u_t_mu, i_t_mu, u_v_mu, i_v_mu)
        if not training:
            self._cached_user_embedding = u_embedding
            self._cached_item_embedding = i_embedding
        adj = self._adaptive_graph(u_t_mu, i_t_mu, u_v_mu, i_v_mu)
        if training and self.dropout > 0.:
            norm_adj = self.sample_adj(adj, self.dropout)
        else:
            norm_adj = self._normalize_laplacian(adj)
        all_g_embeddings = torch.cat([u_embedding, i_embedding], dim=0)
        all_g_embeddings = self.simple_gcn(all_g_embeddings, norm_adj)
        u_g_embedding, i_g_embedding = torch.split(all_g_embeddings, [self.n_users, self.n_items], dim=0)
        if training:
            return u_g_embedding, i_g_embedding, u_t_mu, i_t_mu, u_v_mu, i_v_mu
        else:
            return u_g_embedding, i_g_embedding, i_t_mu, i_v_mu

    def sl_loss(self, users, pos_items, neg_items):
        # neg_items must be [N, K, dim] (each user's own K negatives). A 2-D [N, dim]
        # tensor would broadcast against users.unsqueeze(1)=[N,1,dim] into an
        # invalid [N, N] cross-user similarity.
        if neg_items.dim() != 3:
            raise ValueError(
                f"sl_loss expects [N, K, dim] negatives, got shape {tuple(neg_items.shape)}."
            )
        pos_scores = F.cosine_similarity(users, pos_items)
        neg_scores = F.cosine_similarity(users.unsqueeze(1), neg_items, dim=2)
        d = neg_scores - pos_scores.unsqueeze(1)
        return torch.logsumexp(d / self.temperature, dim=1).mean()

    def infonce_loss(self, emb1, emb2):
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        scores = torch.exp(torch.matmul(emb1, emb2.T) / self.temperature)
        pos_sim = scores.diag()
        loss = -torch.log(pos_sim / torch.sum(scores, dim=1)).mean()
        return loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        if len(interaction) >= 3:
            # dataloader emits [users, pos, neg_1, ..., neg_K]; stack the K negative rows
            # into [N, K] so sl_loss uses each user's own negatives.
            neg_items = torch.stack(list(interaction[2:]), dim=1)
        elif len(interaction) == 2:
            num_negatives = self.config["sampling"]["num_negatives"]
            neg_items = torch.randint(0, self.n_items, (users.shape[0], num_negatives), device=users.device)
        else:
            raise ValueError(f"Unexpected interaction format with {len(interaction)} elements")

        u_g_embeddings, i_g_embeddings, u_t_mu, i_t_mu, u_v_mu, i_v_mu = self.forward(True)
        m_loss = 0.
        if self.use_text and self.use_image:
            u_m_loss = self.infonce_loss(u_t_mu[users], u_v_mu[users])
            i_m_loss = self.infonce_loss(i_t_mu[pos_items], i_v_mu[pos_items])
            m_loss = u_m_loss + i_m_loss
        loss_rec = self.sl_loss(u_g_embeddings[users], i_g_embeddings[pos_items], i_g_embeddings[neg_items])
        return loss_rec + self.lambda_align * m_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        u_g_embeddings, i_g_embeddings, i_t_mu, i_v_mu = self.forward(False)
        if self.save_intermediate:
            torch.save(u_g_embeddings, os.path.join(self.intermediate_path, f'u_embeddings_{self.current_epoch}.pt'))
            torch.save(i_g_embeddings, os.path.join(self.intermediate_path, f'i_embeddings_{self.current_epoch}.pt'))
        u_embeddings = u_g_embeddings[user]
        scores = torch.matmul(u_embeddings, i_g_embeddings.transpose(0, 1))
        return scores
    
    def pre_epoch_processing(self, epoch_idx=None):
        if epoch_idx is not None:
            self.current_epoch = epoch_idx
        else:
            if not hasattr(self, 'current_epoch'):
                self.current_epoch = 0
            else:
                self.current_epoch += 1

    def get_resume_state(self):
        """Persist the epoch counter (a plain attr, not in state_dict) so a resumed
        run continues the save_intermediate diagnostic numbering instead of from 0."""
        return {"current_epoch": self.current_epoch}

    def set_resume_state(self, state):
        self.current_epoch = state["current_epoch"]


class EdgeMLP(nn.Module):
    def __init__(self, adj, in_dim, device):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.edge_index = torch.clone(adj._indices())
        self.edge_val = torch.clone(adj._values())
        self.adj_size = adj.size()
        self.device = device

    def forward(self, src_embeddings, dst_embeddings):
        src_hidden = self.fc1(src_embeddings)
        dst_hidden = self.fc2(dst_embeddings)
        src, dst = self.edge_index[0], self.edge_index[1]
        src_edge_hidden, dst_edge_hidden = src_hidden[src], dst_hidden[dst]
        edge_logits = torch.mul(src_edge_hidden, dst_edge_hidden).sum(1).squeeze()
        edge_weights = self.edge_val * torch.sigmoid(edge_logits)
        return torch.sparse_coo_tensor(
            self.edge_index, edge_weights, self.adj_size, device=self.device
        )


class PositionalEncoding(nn.Module):
    def __init__(self, pos_dim, max_len, device):
        super().__init__()
        self.pos_dim = pos_dim
        self.max_len = max_len
        self.device = device

        self.pe = torch.zeros(max_len, pos_dim, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-math.log(10000.0) / pos_dim))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.pe


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, activation="tanh", layer_norm=True):
        super().__init__()
        self.fcs = nn.ModuleList()
        if activation == 'tanh':
            activation_layer = nn.Tanh()
        elif activation == 'siLu':
            activation_layer = nn.SiLU()
        elif activation == "sigmoid":
            activation_layer = nn.Sigmoid()
        elif activation == "softmax":
            activation_layer = nn.Softmax(dim=-1)
        elif activation == "relu":
            activation_layer = nn.ReLU()
        else:
            activation_layer = None

        if input_dim <= output_dim:
            for i in range(num_layers):
                self.fcs.append(nn.Linear(input_dim, output_dim))
                if activation_layer is not None:
                    self.fcs.append(activation_layer)
                if layer_norm:
                    self.fcs.append(nn.LayerNorm(output_dim))

        else:
            step_ratio = (output_dim / input_dim) ** (1 / num_layers)
            dims = [input_dim]
            for i in range(1, num_layers):
                current_dim = input_dim * (step_ratio ** i)
                hidden_dim = self.next_power_of_two(current_dim)
                self.fcs.append(nn.Linear(dims[-1], hidden_dim))
                if activation_layer is not None:
                    self.fcs.append(activation_layer)
                if layer_norm:
                    self.fcs.append(nn.LayerNorm(hidden_dim))

                dims.append(hidden_dim)

            self.fcs.append(nn.Linear(dims[-1], output_dim))
            if activation_layer is not None:
                self.fcs.append(activation_layer)
            if layer_norm:
                self.fcs.append(nn.LayerNorm(output_dim))

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x

    def next_power_of_two(self, n):
        """Return the smallest power of 2 that is >= n."""
        return 2 ** math.ceil(math.log2(n))
