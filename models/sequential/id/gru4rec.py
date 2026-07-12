# coding: utf-8
"""
GRU4Rec: Session-based Recommendations with Recurrent Neural Networks
======================================================================

This module implements the GRU4Rec model, one of the pioneering approaches
for session-based sequential recommendation using RNNs.

Paper: Hidasi, B., et al. "Session-based recommendations with recurrent neural networks." 
       arXiv preprint arXiv:1511.06939 (2015).

Author: FedVLR Team
Version: 2.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from core.sequential import SequentialRecommender


class GRU4Rec(SequentialRecommender):
    """
    GRU4Rec model for sequential recommendation.
    
    This model uses a GRU (Gated Recurrent Unit) to encode user interaction
    sequences and predict the next item.
    """
    
    def __init__(self, config, dataloader):
        """
        Initialize GRU4Rec model.
        
        Args:
            config: Configuration dictionary
            dataloader: Data loader instance
        """
        super().__init__(config, dataloader)
        
        # Model configuration
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.embedding_size = config['embedding_size']

        # Loss configuration
        self.loss_type = config['loss_type']
        if self.loss_type not in ('ce', 'bpr', 'top1'):
            raise ValueError(
                f"GRU4Rec supports loss_type 'ce', 'bpr', or 'top1', "
                f"got {self.loss_type!r}."
            )

        # Initialize embeddings
        self.item_embedding = nn.Embedding(
            num_embeddings=self.n_items + 1,  # +1 for padding
            embedding_dim=self.embedding_size,
            padding_idx=0
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Dense projection to embedding space (RecBole style)
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        # Re-zero the PAD (padding_idx=0) row that normal_ just overwrote, so the
        # PAD logit is a benign constant in the CE softmax — matching SASRec/BERT4Rec.
        with torch.no_grad():
            self.item_embedding.weight[self.item_embedding.padding_idx].zero_()

        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize dense projection
        nn.init.xavier_uniform_(self.dense.weight)
        self.dense.bias.data.fill_(0)
    
    def encode_sequence(self, item_seq: torch.Tensor, seq_lens: torch.Tensor,
                       v_feat_seq: Optional[torch.Tensor] = None,
                       t_feat_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode item sequence using GRU.
        
        Args:
            item_seq: Item sequence [batch_size, seq_len]
            seq_lens: Actual sequence lengths [batch_size]
            v_feat_seq: Visual features (not used in GRU4Rec)
            t_feat_seq: Text features (not used in GRU4Rec)
            
        Returns:
            Sequence output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = item_seq.size()
        
        # Embedding lookup
        item_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, embedding_size]
        item_emb = self.dropout(item_emb)
        
        # Pack sequences for efficient RNN processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            item_emb, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # GRU forward pass
        packed_output, hidden = self.gru(packed_input)
        
        # Unpack sequences
        seq_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )
        
        return seq_output
    
    def decode_next(self, seq_representation: torch.Tensor) -> torch.Tensor:
        """
        Decode sequence representation to predict next item via embedding dot product.

        Args:
            seq_representation: Sequence representation [batch_size, hidden_size]

        Returns:
            Item scores [batch_size, n_items+1] — index 0 is PAD (unused),
            index i corresponds to item ID i (1-indexed convention).
        """
        projected = self.dense(seq_representation)  # [B, embedding_size]
        # Include all rows (PAD at 0, items at 1..n_items) so score index == item ID
        all_item_emb = self.item_embedding.weight  # [n_items+1, embedding_size]
        return torch.matmul(projected, all_item_emb.T)  # [B, n_items+1]
    
    def forward(self, item_seq: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GRU4Rec.
        
        Args:
            item_seq: Item sequence [batch_size, seq_len]
            seq_lens: Actual sequence lengths [batch_size]
            
        Returns:
            Item scores [batch_size, num_items]
        """
        # Encode sequence
        seq_output = self.encode_sequence(item_seq, seq_lens)
        
        # Get last hidden state for each sequence
        user_repr = self.gather_indexes(seq_output, seq_lens - 1)
        
        # Decode to get item scores
        scores = self.decode_next(user_repr)
        
        return scores
    
    def calculate_loss(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate loss for GRU4Rec.
        
        Args:
            interaction: Interaction dictionary containing:
                - item_seqs: Item sequences [batch_size, seq_len]
                - targets: Target items [batch_size]
                - seq_lens: Sequence lengths [batch_size]
                - neg_items: Negative items [batch_size] (optional)
                
        Returns:
            Loss tensor
        """
        if self.loss_type == 'bpr':
            if 'neg_items' not in interaction:
                raise ValueError(
                    "GRU4Rec loss_type='bpr' requires sampled neg_items in the "
                    "batch; enable neg_sampling instead of using cross-entropy."
                )
            return self._calculate_bpr_loss(interaction)
        if self.loss_type == 'top1':
            return self._calculate_top1_loss(interaction)
        return self._calculate_ce_loss(interaction)
    
    def _calculate_ce_loss(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate cross-entropy loss."""
        item_seqs = interaction['item_seqs']
        targets = interaction['targets']
        seq_lens = interaction['seq_lens']
        
        # Forward pass
        scores = self.forward(item_seqs, seq_lens)
        
        # Cross-entropy loss
        loss = F.cross_entropy(scores, targets)
        
        return loss
    
    def _calculate_bpr_loss(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate BPR (Bayesian Personalized Ranking) loss."""
        item_seqs = interaction['item_seqs']
        targets = interaction['targets']
        neg_items = interaction['neg_items']
        seq_lens = interaction['seq_lens']
        
        # Forward pass
        scores = self.forward(item_seqs, seq_lens)
        
        # Get scores for positive and negative items
        pos_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
        neg_scores = scores.gather(1, neg_items.unsqueeze(1)).squeeze(1)
        
        # BPR loss
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        return loss
    
    def _calculate_top1_loss(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate TOP1 loss (as used in original GRU4Rec paper).
        
        This is a ranking loss that ensures the target item has the highest score.
        """
        item_seqs = interaction['item_seqs']
        targets = interaction['targets']
        seq_lens = interaction['seq_lens']
        
        # Forward pass
        scores = self.forward(item_seqs, seq_lens)

        # Get scores for target items
        pos_scores = scores.gather(1, targets.unsqueeze(1))  # [batch, 1]

        # TOP1 loss from Hidasi et al. 2015, averaged over ALL negatives j (not
        # just the single hardest one):
        #   L = mean_j [ σ(r_j - r_i) + σ(r_j²) ]
        # The σ(r_j²) term regularizes negative scores toward 0.
        per_item = torch.sigmoid(scores - pos_scores) + torch.sigmoid(scores ** 2)  # [batch, n_items]

        # Mask the PAD column (0) and each sample's target so they are excluded
        # from the negative average.
        neg_mask = torch.ones_like(scores, dtype=torch.bool)
        neg_mask[:, 0] = False  # PAD is never a valid negative
        neg_mask.scatter_(1, targets.unsqueeze(1), False)  # exclude the target item

        per_item = per_item.masked_fill(~neg_mask, 0.0)
        neg_counts = neg_mask.sum(dim=1).clamp(min=1)
        loss = (per_item.sum(dim=1) / neg_counts).mean()

        return loss
    
    def full_sort_predict(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict scores for all items.
        
        Args:
            interaction: Interaction dictionary
            
        Returns:
            Item scores [batch_size, num_items]
        """
        item_seqs = interaction['item_seqs']
        seq_lens = interaction['seq_lens']
        
        return self.forward(item_seqs, seq_lens)
