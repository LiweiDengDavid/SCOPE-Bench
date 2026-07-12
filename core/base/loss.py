# coding: utf-8



import torch
import torch.nn as nn


class BPRLoss(nn.Module):

    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """
    def __init__(self, gamma=1e-10):
        super().__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super().__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if not embeddings:
            raise ValueError("EmbLoss requires at least one embedding tensor")
        emb_loss = torch.zeros(1).to(embeddings[0].device)
        for embedding in embeddings:
            if require_pow:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            else:
                emb_loss += torch.norm(embedding, p=self.norm)
        # Normalize by the first embedding's batch dimension (caller convention:
        # first arg is the primary embedding whose batch size sets the scale).
        num_embs = embeddings[0].shape[0]
        if num_embs > 0:
            emb_loss /= num_embs
        return emb_loss
