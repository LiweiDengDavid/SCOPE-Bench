# coding: utf-8
"""GRCN multimodal graph recommendation model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_edge, remove_self_loops, add_self_loops, softmax

from core.base import RecommenderBase

##########################################################################


class SAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        normalize=True,
        bias=True,
        aggr="mean",
        **kwargs,
    ):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, weight_vector, size=None):
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loops=False):
        super(GATConv, self).__init__(aggr="add")  # , **kwargs)
        self.self_loops = self_loops
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_i, x_j, size_i, edge_index_i):
        self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
        self.alpha = softmax(self.alpha, edge_index_i, num_nodes=size_i)
        return x_j * self.alpha.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out


class EGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm):
        super(EGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.aggr_mode = aggr_mode
        self.has_act = has_act
        self.has_norm = has_norm
        self.id_embedding = nn.Parameter(
            nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E)))
        )
        self.conv_embed_1 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)
        self.conv_embed_2 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

        if self.has_norm:
            x = F.normalize(x)

        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector)

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector)
        if self.has_act:
            x_hat_2 = F.leaky_relu_(x_hat_2)

        return x + x_hat_1 + x_hat_2


class CGCN(torch.nn.Module):
    def __init__(
        self,
        features,
        num_user,
        num_item,
        dim_C,
        aggr_mode,
        num_routing,
        has_act,
        has_norm,
        is_word=False,
        device=None,
    ):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing
        self.has_act = has_act
        self.has_norm = has_norm
        self.dim_C = dim_C
        self.device = device
        self.preference = nn.Parameter(
            nn.init.xavier_normal_(torch.rand((num_user, dim_C)))
        )
        self.conv_embed_1 = GATConv(self.dim_C, self.dim_C)
        self.is_word = is_word

        if is_word:
            self.word_tensor = torch.LongTensor(features)
            self.features = nn.Embedding(torch.max(features[1]) + 1, dim_C)
            nn.init.xavier_normal_(self.features.weight)

        else:
            self.dim_feat = features.size(1)
            self.features = features
            self.MLP = nn.Linear(self.dim_feat, self.dim_C)
            nn.init.xavier_normal_(self.MLP.weight)

    def forward(self, edge_index):
        features = F.leaky_relu(self.MLP(self.features))

        if self.has_norm:
            preference = F.normalize(self.preference)
            features = F.normalize(features)

        for i in range(self.num_routing):
            x = torch.cat((preference, features), dim=0)
            x_hat_1 = self.conv_embed_1(x, edge_index)
            preference = preference + x_hat_1[: self.num_user]

            if self.has_norm:
                preference = F.normalize(preference)

        x = torch.cat((preference, features), dim=0)
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

        x_hat_1 = self.conv_embed_1(x, edge_index)

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1)

    def to(self, device):
        """Override to() to correctly handle device transfers."""
        super().to(device)
        if hasattr(self, "word_tensor"):
            self.word_tensor = self.word_tensor.to(device)
        if hasattr(self, "features") and not isinstance(self.features, nn.Embedding):
            self.features = self.features.to(device)
        return self


class GRCN(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)
        self.num_user = self.n_users
        self.num_item = self.n_items
        dim_x = config["embedding_size"]
        self.num_layers = config["num_layers"]
        self.aggr_mode = "add"
        self.weight_mode = "confid"
        self.fusion_mode = "concat"
        has_act = False
        has_norm = True
        is_word = False
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        # In-model embedding-L2 coefficient, decoupled from optimizer weight_decay.
        self.reg_weight = float(config["reg_weight"])
        self.dropout = float(config["dropout_rate"])
        train_interactions = dataloader.inter_matrix(form="coo").astype(np.float32)
        edge_index = torch.tensor(
            self.pack_edge_index(train_interactions), dtype=torch.long
        )
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.id_gcn = EGCN(self.n_users, self.n_items, dim_x, self.aggr_mode, has_act, has_norm)
        self.pruning = True

        num_modalities = 0
        if self.v_feat is not None:
            self.v_feat = self._align_feat_rows(self.v_feat)
            self.v_gcn = CGCN(
                self.v_feat,
                self.n_users,
                self.n_items,
                dim_x,
                self.aggr_mode,
                self.num_layers,
                has_act,
                has_norm,
                device=self.device,
            )
            num_modalities += 1

        if self.t_feat is not None:
            self.t_feat = self._align_feat_rows(self.t_feat)
            self.t_gcn = CGCN(
                self.t_feat,
                self.n_users,
                self.n_items,
                dim_x,
                self.aggr_mode,
                self.num_layers,
                has_act,
                has_norm,
                is_word,
                device=self.device,
            )
            num_modalities += 1

        self.model_specific_conf = nn.Parameter(
            nn.init.xavier_normal_(torch.rand((self.n_users + self.n_items, num_modalities)))
        )

    def _align_feat_rows(self, feat: torch.Tensor) -> torch.Tensor:
        """Pad or truncate feature rows to match the item count."""
        if feat.shape[0] < self.num_item:
            pad_rows = self.num_item - feat.shape[0]
            pad = torch.zeros(
                pad_rows,
                feat.shape[1],
                device=feat.device,
                dtype=feat.dtype,
            )
            return torch.cat([feat, pad], dim=0)
        if feat.shape[0] > self.num_item:
            return feat[: self.num_item]
        return feat

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def forward(self):
        weights = []
        content_reps = []
        edge_index, _ = dropout_edge(self.edge_index, p=self.dropout, training=self.training)

        if self.v_feat is not None:
            v_rep, weight_v = self.v_gcn(edge_index)
            content_reps.append(v_rep)
            weights.append(weight_v)

        if self.t_feat is not None:
            t_rep, weight_t = self.t_gcn(edge_index)
            content_reps.append(t_rep)
            weights.append(weight_t)

        weight = weights[0]
        if len(weights) > 1:
            if self.weight_mode == "mean":
                weight = sum(weights) / len(weights)
            else:
                weight = torch.cat(weights, dim=1)

        content_rep = content_reps[0] if len(content_reps) == 1 else torch.cat(content_reps, dim=1)

        if self.weight_mode == "max":
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)
        elif self.weight_mode == "confid":
            # GRCN graph-refining (Wei et al., ACM MM 2020): each directed edge is
            # weighted by the confidence of its SOURCE node only — NOT the product
            # of both endpoints. edge_index_for_weight = [forward; reverse] is in
            # the same column order as the attention `weight`, so its row-0 (source)
            # gives forward edges user-node confidence and reverse edges item-node
            # confidence, matching the author repo (weiyinwei/GRCN) and MMRec.
            edge_index_for_weight = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
            confidence = self.model_specific_conf[edge_index_for_weight[0]]
            weight = weight * confidence
            weight, _ = torch.max(weight, dim=1, keepdim=True)

        if self.pruning:
            weight = torch.relu(weight)

        id_rep = self.id_gcn(edge_index, weight)

        if self.fusion_mode == "concat":
            representation = torch.cat((id_rep, content_rep), dim=1)
        elif self.fusion_mode == "id":
            representation = id_rep
        elif self.fusion_mode == "mean":
            reps = [id_rep]
            reps.extend(content_reps)
            representation = sum(reps) / len(reps)

        return representation

    def calculate_loss(self, interaction):
        if len(interaction) == 3:
            batch_users = interaction[0]
            pos_items = interaction[1] + self.n_users
            neg_items = interaction[2] + self.n_users
        elif len(interaction) == 2:
            batch_users = interaction[0]
            pos_items = interaction[1] + self.n_users
            neg_items = (
                torch.randint(
                    0, self.n_items, interaction[1].shape, device=interaction[1].device
                )
                + self.n_users
            )
        else:
            raise ValueError(
                f"Unsupported interaction format with {len(interaction)} elements"
            )

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (
            self.id_gcn.id_embedding[user_tensor] ** 2
            + self.id_gcn.id_embedding[item_tensor] ** 2
        ).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        reg_content_loss = torch.zeros(1, device=self.device)
        if self.v_feat is not None:
            reg_content_loss = (
                reg_content_loss + (self.v_gcn.preference[user_tensor] ** 2).mean()
            )
        if self.t_feat is not None:
            reg_content_loss = (
                reg_content_loss + (self.t_gcn.preference[user_tensor] ** 2).mean()
            )

        reg_confid_loss = (self.model_specific_conf**2).mean()

        reg_loss = reg_embedding_loss + reg_content_loss + reg_confid_loss

        reg_loss = self.reg_weight * reg_loss

        return loss + reg_loss

    def full_sort_predict(self, interaction):
        # Recompute under the trainer's eval mode so edge dropout follows eval semantics.
        representation = self.forward()
        user_tensor = representation[: self.n_users]
        item_tensor = representation[self.n_users :]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix
