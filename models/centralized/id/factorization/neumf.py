# coding: utf-8
"""
NeuMF (Neural Matrix Factorization) - ported from RecBole baseline
===================================================================

Directly ported from RecBole, preserving the dual-path architecture and
fusion logic exactly as in the original.

Reference:
    Xiangnan He et al. "Neural Collaborative Filtering." in WWW 2017.

RecBole Reference Implementation:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/neumf.py
"""

import torch
import torch.nn as nn
from core.base import EmbLoss, RecommenderBase


class NeuMF(RecommenderBase):
    """NeuMF baseline model - directly ported from RecBole

    Fully preserves the RecBole dual-path architecture:
    - GMF (Generalized Matrix Factorization) path
    - MLP (Multi-Layer Perceptron) path
    - Final prediction layer fusing both paths

    Serves as the neural collaborative filtering baseline in NexusRec.
    """
    
    def _generate_mlp_hidden_sizes(self):
        """Generate the list of MLP hidden layer sizes from the first-layer
        dimension and the number of layers.

        Returns:
            list: MLP hidden layer dimension list, each layer halved in size
        """
        if self.n_layers <= 0:
            return []
        
        sizes = []
        current_size = self.mlp_hidden_size
        for i in range(self.n_layers):
            sizes.append(current_size)
            current_size = current_size // 2
        return sizes
    
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        
        # Core parameters - aligned with RecBole defaults
        self.latent_dim = config['embedding_size']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']
        self.dropout_prob = config['dropout_rate']

        # MLP layer configuration
        self.n_layers = config['num_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']

        # Generate MLP hidden layer size list
        mlp_hidden_sizes = self._generate_mlp_hidden_sizes() if self.mlp_train else []

        # GMF path - Generalized Matrix Factorization
        if self.mf_train:
            self.user_mf_embedding = nn.Embedding(self.n_users, self.latent_dim)
            self.item_mf_embedding = nn.Embedding(self.n_items, self.latent_dim)

        # MLP path - Multi-Layer Perceptron
        if self.mlp_train:
            self.user_mlp_embedding = nn.Embedding(self.n_users, self.latent_dim)
            self.item_mlp_embedding = nn.Embedding(self.n_items, self.latent_dim)

            # Build MLP layers
            mlp_layers = []
            mlp_size = [2 * self.latent_dim] + mlp_hidden_sizes

            for i in range(len(mlp_size) - 1):
                mlp_layers.append(nn.Linear(mlp_size[i], mlp_size[i + 1]))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(self.dropout_prob))

            self.mlp_layers = nn.Sequential(*mlp_layers)

        # Prediction layer
        if self.mf_train and self.mlp_train:
            predict_size = self.latent_dim + mlp_hidden_sizes[-1]
        elif self.mf_train:
            predict_size = self.latent_dim
        else:
            predict_size = mlp_hidden_sizes[-1]

        self.predict_layer = nn.Linear(predict_size, 1)

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = EmbLoss()
        self.embedding_weight_decay = float(config['embedding_weight_decay'])

        # Weight initialization — aligned with RecBole NeuMF (normal_(0, 0.01))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight.data, 0, 0.01)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0, 0.01)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
    
    def forward(self, user, item):
        """Forward pass - dual-path architecture"""
        outputs = []

        # GMF path
        if self.mf_train:
            user_mf_e = self.user_mf_embedding(user)
            item_mf_e = self.item_mf_embedding(item)
            mf_output = user_mf_e * item_mf_e  # element-wise product
            outputs.append(mf_output)

        # MLP path
        if self.mlp_train:
            user_mlp_e = self.user_mlp_embedding(user)
            item_mlp_e = self.item_mlp_embedding(item)
            mlp_input = torch.cat([user_mlp_e, item_mlp_e], dim=1)
            mlp_output = self.mlp_layers(mlp_input)
            outputs.append(mlp_output)

        # Fuse both paths
        prediction_input = torch.cat(outputs, dim=1)
        prediction = self.predict_layer(prediction_input)

        return prediction.squeeze(-1)
    
    def calculate_loss(self, interaction):
        """Calculate loss - BCE loss (He 2017)"""
        user = interaction[0]      # user IDs
        pos_item = interaction[1]  # positive item IDs
        neg_item = interaction[2]  # negative item IDs

        pos_logit = self.forward(user, pos_item)
        neg_logit = self.forward(user, neg_item)

        logits = torch.cat([pos_logit, neg_logit])
        labels = torch.cat([
            torch.ones(pos_logit.size(0), device=self.device),
            torch.zeros(neg_logit.size(0), device=self.device),
        ])
        mf_loss = self.bce_loss(logits, labels)

        # Regularization loss
        reg_loss = 0.0
        if self.embedding_weight_decay > 0:
            reg_embeddings = []
            if self.mf_train:
                reg_embeddings.extend([
                    self.user_mf_embedding(user),
                    self.item_mf_embedding(pos_item),
                    self.item_mf_embedding(neg_item)
                ])
            if self.mlp_train:
                reg_embeddings.extend([
                    self.user_mlp_embedding(user),
                    self.item_mlp_embedding(pos_item),
                    self.item_mlp_embedding(neg_item)
                ])
            reg_loss = self.reg_loss_fn(*reg_embeddings)

        return mf_loss + self.embedding_weight_decay * reg_loss
    
    def predict(self, interaction):
        """Predict scores"""
        if isinstance(interaction, dict):
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]
        else:
            user = interaction[0]
            item = interaction[1]
            
        return self.forward(user, item)
    
    def full_sort_predict(self, interaction):
        """Full-sort prediction"""
        if isinstance(interaction, dict):
            user = interaction[self.USER_ID]
        else:
            user = interaction[0]

        if user.dim() == 0:
            user = user.unsqueeze(0)

        # Support both single-user and batched full-sort evaluation.
        items = torch.arange(self.n_items, device=self.device)
        user = user.view(-1)
        user_expanded = user.unsqueeze(1).expand(-1, self.n_items).reshape(-1)
        item_expanded = items.unsqueeze(0).expand(user.size(0), -1).reshape(-1)

        scores = self.forward(user_expanded, item_expanded)
        return scores.view(user.size(0), self.n_items)
