import torch
import torch.nn as nn

from core.base import RecommenderBase, xavier_normal_initialization


class FCF(RecommenderBase):
    """
    Federated Collaborative Filtering with shared item commonality.

    The server aggregates the item embedding table. The prediction head remains
    personal, so this implementation exposes an item-only forward contract and
    keeps client personalization outside the shared parameter set.
    """

    def __init__(self, config, dataloader):
        super(FCF, self).__init__(config, dataloader)

        # Embedding dimension size
        self.embed_size = config['embedding_size']

        # Item commonality feature embedding layer
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_size)

        # Output layer: converts embeddings to scores
        self.affine_output = torch.nn.Linear(in_features=self.embed_size, out_features=1)
        # Sigmoid activation maps output to the range [0, 1]
        self.logistic = torch.nn.Sigmoid()

        # Initialize model parameters
        self.apply(xavier_normal_initialization)

    def forward(self, item_indices):
        """Forward pass to predict item scores."""
        # Retrieve item commonality feature embeddings
        item_commonality = self.item_commonality(item_indices)

        # Compute predicted scores via linear layer and sigmoid activation
        pred = self.affine_output(item_commonality)
        rating = self.logistic(pred)

        return rating.squeeze(-1)

    def calculate_loss(self, batch):
        """Calculate loss."""
        _, pos_items, neg_items = batch[0], batch[1], batch[2]

        # Construct positive and negative samples
        items = torch.cat([pos_items, neg_items])
        labels = torch.cat([
            torch.ones(pos_items.size(0), device=self.device),
            torch.zeros(neg_items.size(0), device=self.device)
        ])

        # Predict
        predictions = self.forward(items)

        # BCE matches the probability-valued output returned by forward().
        loss = nn.functional.binary_cross_entropy(predictions, labels)

        return loss

    def full_sort_predict(self, interaction, *args, **kwargs):
        """Predict scores for all items."""
        items = torch.arange(self.n_items, device=self.device)
        scores = self.forward(items)
        return scores.unsqueeze(0)

    def get_shared_parameters(self):
        """Get shared parameters for federated aggregation."""
        return {
            'item_commonality.weight': self.item_commonality.weight
        }

    def get_personal_parameters(self):
        """Get personalized parameters (not participating in federated aggregation)."""
        return {
            'affine_output.weight': self.affine_output.weight,
            'affine_output.bias': self.affine_output.bias
        }
