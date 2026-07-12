# coding: utf-8
"""
https://github.com/jing-1/MVGAE
Paper: Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems
IEEE TMM'21
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn.inits import uniform

from core.base import RecommenderBase

EPS = 1e-15
MAX_LOGVAR = 10
# Reparametrization noise scale (reference-faithful constant; see paper / upstream
# https://github.com/jing-1/MVGAE). Kept as a named module constant alongside
# EPS/MAX_LOGVAR rather than an inline literal.
NOISE_SCALE = 0.1


class MVGAE(RecommenderBase):
    def __init__(self, config, dataloader):
        super(MVGAE, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)
        self.experts = ProductOfExperts()
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.num_layers = config['num_layers']
        self.aggr_mode = 'mean'
        self.concate = False
        self.embedding_size = config['embedding_size']
        self.dim_x = self.embedding_size  # Use embedding_size uniformly
        self.beta = config['beta']
        self.collaborative = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.n_items, self.dim_x)))
        # packing interaction in training into edge_index
        train_interactions = dataloader.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        dim_latent = config['latent_dim']
        if self.v_feat is not None:
            self.v_gcn = GCN(self.device, self.v_feat, self.edge_index, self.batch_size, self.num_user, self.num_item, self.dim_x,
                             self.aggr_mode, self.concate, num_layer=self.num_layers, dim_latent=dim_latent, dropout=self.dropout_rate)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.device, self.t_feat, self.edge_index, self.batch_size, self.num_user, self.num_item, self.dim_x,
                             self.aggr_mode, self.concate, num_layer=self.num_layers, dim_latent=dim_latent, dropout=self.dropout_rate)
        self.c_gcn = GCN(self.device, self.collaborative, self.edge_index, self.batch_size, self.num_user, self.num_item,
                         self.dim_x,
                         self.aggr_mode, self.concate, num_layer=self.num_layers, dim_latent=dim_latent, dropout=self.dropout_rate)
        self.result_embed = nn.init.xavier_normal_(torch.rand((self.num_user + self.num_item, self.dim_x))).to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def reparametrize(self, mu, logvar):
        logvar = logvar.clamp(max=MAX_LOGVAR)
        if self.training:
            return mu + torch.randn_like(logvar) * NOISE_SCALE * torch.exp(logvar.mul(0.5))
        else:
            return mu

    def dot_product_decode_neg(self, z, user, neg_items, sigmoid=True):
        # Score each user against its own sampled negative. user and neg_items are
        # batch-aligned [N] in node space.
        neg_values = torch.sum(z[user] * z[neg_items], dim=-1)
        return torch.sigmoid(neg_values) if sigmoid else neg_values

    def dot_product_decode(self, z, edge_index, sigmoid=True):
        value = torch.sum(z[edge_index[0]] * z[edge_index[1]], dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward(self):
        v_mu, v_logvar = self.v_gcn()
        t_mu, t_logvar = self.t_gcn()
        c_mu, c_logvar = self.c_gcn()
        mu = torch.stack([v_mu, t_mu], dim=0)
        logvar = torch.stack([v_logvar, t_logvar], dim=0)

        pd_mu, pd_logvar, _ = self.experts(mu, logvar)
        del mu
        del logvar

        mu = torch.stack([pd_mu, c_mu], dim=0)
        logvar = torch.stack([pd_logvar, c_logvar], dim=0)

        pd_mu, pd_logvar, _ = self.experts(mu, logvar)
        del mu
        del logvar
        z = self.reparametrize(pd_mu, pd_logvar)

        # Amazon-style sparse datasets (all datasets used in this framework) apply
        # sigmoid regularization to the readout embedding.
        self.result_embed = torch.sigmoid(pd_mu)
        return pd_mu, pd_logvar, z, v_mu, v_logvar, t_mu, t_logvar, c_mu, c_logvar

    def recon_loss(self, z, pos_edge_index, user, neg_items):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        # Amazon-style sparse datasets (all datasets used here) apply sigmoid regularization.
        z = torch.sigmoid(z)

        pos_scores = self.dot_product_decode(z, pos_edge_index, sigmoid=True)
        neg_scores = self.dot_product_decode_neg(z, user, neg_items, sigmoid=True)
        loss = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        return loss

    def kl_loss(self, mu, logvar):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logvar`, or based on latent variables from last encoding.
        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logvar (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        logvar = logvar.clamp(max=MAX_LOGVAR)
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    def calculate_loss(self, interaction):
        # Handle different interaction formats
        if len(interaction) == 3:
            user = interaction[0]
            pos_items = interaction[1]
            neg_items = interaction[2]
        elif len(interaction) == 2:
            user = interaction[0]
            pos_items = interaction[1]
            # Generate random negative samples
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=pos_items.device)
        else:
            raise ValueError(f"Unsupported interaction format with {len(interaction)} elements")
        # forward() returns node-space embeddings: user rows first, then item rows.
        # Lift raw item ids into node space so train and eval score the same region.
        pos_items_node = pos_items + self.n_users
        neg_items_node = neg_items + self.n_users
        pos_edge_index = torch.stack([user, pos_items_node], dim=0)
        pd_mu, pd_logvar, z, v_mu, v_logvar, t_mu, t_logvar, c_mu, c_logvar = self.forward()

        z_v = self.reparametrize(v_mu, v_logvar)
        z_t = self.reparametrize(t_mu, t_logvar)
        z_c = self.reparametrize(c_mu, c_logvar)
        recon_loss = self.recon_loss(z, pos_edge_index, user, neg_items_node)
        kl_loss = self.kl_loss(pd_mu, pd_logvar)
        loss_multi = recon_loss + self.beta * kl_loss
        loss_v = self.recon_loss(z_v, pos_edge_index, user, neg_items_node) + self.beta * self.kl_loss(v_mu, v_logvar)
        loss_t = self.recon_loss(z_t, pos_edge_index, user, neg_items_node) + self.beta * self.kl_loss(t_mu, t_logvar)
        loss_c = self.recon_loss(z_c, pos_edge_index, user, neg_items_node) + self.beta* self.kl_loss(c_mu, c_logvar)
        return loss_multi + loss_v + loss_t + loss_c

    def full_sort_predict(self, interaction):
        # Recompute under the trainer's eval mode so result_embed is clean
        # (reparametrize returns mu and dropout is disabled when not training).
        self.forward()
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix


class GCN(torch.nn.Module):
    def __init__(self, device, features, edge_index, batch_size, num_user, num_item, dim_id, aggr_mode, concate,
                 num_layer, dim_latent, dropout):
        super(GCN, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.dropout = dropout

        if self.dim_latent:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(self.num_user, self.dim_latent)))
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            nn.init.xavier_normal_(self.MLP.weight)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_id, aggr=self.aggr_mode, dropout=self.dropout)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(self.num_user, self.dim_feat)))
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_id, aggr=self.aggr_mode, dropout=self.dropout)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode, dropout=self.dropout)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_4 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode, dropout=self.dropout)
        nn.init.xavier_normal_(self.conv_embed_4.weight)
        self.linear_layer4 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer4.weight)
        self.g_layer4 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)
        nn.init.xavier_normal_(self.g_layer4.weight)
        self.conv_embed_5 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode, dropout=self.dropout)
        nn.init.xavier_normal_(self.conv_embed_5.weight)
        self.linear_layer5 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer5.weight)
        self.g_layer5 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)
        nn.init.xavier_normal_(self.g_layer5.weight)

    def forward(self):
        temp_features = self.MLP(self.features) if self.dim_latent else self.features
        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x).to(self.device)

        if self.num_layer > 0:
            h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))
            x_hat = F.leaky_relu(self.linear_layer1(x))
            x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer1(h))
            del x_hat
            del h

        if self.num_layer > 1:
            h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))
            x_hat = F.leaky_relu(self.linear_layer2(x))
            x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer2(h))
            del h
            del x_hat

        mu = F.leaky_relu(self.conv_embed_4(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer4(x))
        mu = self.g_layer4(torch.cat((mu, x_hat), dim=1)) if self.concate else self.g_layer4(mu) + x_hat
        del x_hat

        logvar = F.leaky_relu(self.conv_embed_5(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer5(x))
        logvar = self.g_layer5(torch.cat((logvar, x_hat), dim=1)) if self.concate else self.g_layer5(logvar) + x_hat
        del x_hat
        return mu, logvar


class ProductOfExperts(torch.nn.Module):
    def __init__(self):
        super(ProductOfExperts, self).__init__()
        """Return parameters for product of independent experts.
        See https://arxiv.org/pdf/1410.7827.pdf for equations.
        @param mu: M x D for M experts
        @param logvar: M x D for M experts
        """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar, pd_var


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', dropout=0.1, **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index.long(), num_nodes=x.size(0))
            edge_index = edge_index.long()
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return F.dropout(aggr_out, p=self.dropout, training=self.training)

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
