# coding: utf-8
"""
MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video. 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform

from core.base import RecommenderBase


class MMGCN(RecommenderBase):
    """MMGCN model with modality-specific GCN channels and ID embeddings."""

    def __init__(self, config, dataloader):
        super(MMGCN, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        # Basic parameters
        self.num_user = self.n_users
        self.num_item = self.n_items
        dim_x = config['embedding_size']
        self.num_layers = config['num_layers']
        # MMGCN's GCN stack has three convolution steps in this implementation,
        # so expose that depth as a runtime contract.
        if self.num_layers != 3:
            raise ValueError(
                "MMGCN's GCN depth is fixed at 3 layers in this implementation; "
                f"config['num_layers']={self.num_layers} is not supported. Keep "
                "num_layers=3 (it is excluded from MMGCN's HPO search space)."
            )
        self.aggr_mode = 'mean'
        self.concate = False
        has_id = True
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        # In-model embedding-L2 coefficient, decoupled from optimizer weight_decay.
        self.reg_weight = float(config['reg_weight'])

        # Build interaction edge index
        train_interactions = dataloader.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0

        # Multimodal feature handling - automatically configured
        batch_size = config['train_batch_size']
        dim_latent = config['latent_dim']
        if self.v_feat is not None:
            self.v_gcn = GCN(self.edge_index, batch_size, self.num_user, self.num_item, self.v_feat.size(1), dim_x, self.aggr_mode,
                             self.concate, num_layer=self.num_layers, has_id=has_id, dim_latent=dim_latent, device=self.device)
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = GCN(self.edge_index, batch_size, self.num_user, self.num_item, self.t_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, num_layer=self.num_layers, has_id=has_id, dim_latent=dim_latent, device=self.device)
            self.num_modal += 1

        # ID embedding initialization — registered as nn.Parameter so the optimizer updates them
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_user + self.num_item, dim_x)))
        self.result = None

    def pack_edge_index(self, inter_mat):
        """Pack edge index from interaction matrix"""
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, *args, **kwargs):
        """Forward pass - unified interface"""
        representation = None
        if self.v_feat is not None:
            representation = self.v_gcn(self.v_feat, self.id_embedding)
        if self.t_feat is not None:
            if representation is None:
                representation = self.t_gcn(self.t_feat, self.id_embedding)
            else:
                representation += self.t_gcn(self.t_feat, self.id_embedding)

        representation /= self.num_modal
        self.result = representation
        return representation

    def calculate_loss(self, interaction):
        """Calculate loss - unified interface"""
        # Handle different interaction formats
        if len(interaction) == 3:
            batch_users = interaction[0]
            pos_items = interaction[1] + self.n_users
            neg_items = interaction[2] + self.n_users
        elif len(interaction) == 2:
            batch_users = interaction[0]
            pos_items = interaction[1] + self.n_users
            # Generate random negative samples
            neg_items = torch.randint(0, self.n_items, interaction[1].shape, device=interaction[1].device) + self.n_users
        else:
            raise ValueError(f"Unsupported interaction format with {len(interaction)} elements")

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(F.logsigmoid(torch.matmul(score, self.weight)))
        
        # Regularization loss
        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        """Full-sort prediction - unified interface"""
        self.forward()
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix


class GCN(torch.nn.Module):
    """Graph Convolutional Network module"""
    
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer,
                 has_id, dim_latent=None, device='cpu'):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_user, self.dim_latent)))
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)
        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_user, self.dim_feat)))
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

    def forward(self, features, id_embedding):
        """GCN forward pass"""
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer1(x))
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer1(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer2(x))
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer2(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer3(x))
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer3(h) + x_hat)

        return x


class BaseModel(MessagePassing):
    """Base message passing model"""
    
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, size=None):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
