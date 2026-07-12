# coding: utf-8
"""SASRec sequential recommendation model."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from core.sequential import SequentialRecommender

logger = logging.getLogger("nexusrec")


class PointWiseFeedForward(nn.Module):
    """Position-wise feed-forward network using 1D convolutions."""
    
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(-1, -2)
        outputs = self.conv1(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        return outputs.transpose(-1, -2)


class SASRec(SequentialRecommender):
    """SASRec sequential recommendation model."""
    
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.hidden_size = config['hidden_size']
        self.num_blocks = config['num_layers']
        self.num_heads = config['num_attention_heads']
        self.dropout_rate = config['dropout_rate']
        self.norm_first = config['norm_first']

        self.loss_type = config['loss_type']
        if self.loss_type not in ('ce', 'bpr'):
            raise ValueError(
                f"SASRec supports loss_type 'ce' or 'bpr', got {self.loss_type!r}."
            )
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0
        )
        self.pos_embedding = nn.Embedding(
            self.max_seq_len + 1, self.hidden_size, padding_idx=0
        )
        
        self.emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        for _ in range(self.num_blocks):
            self.attention_layernorms.append(
                nn.LayerNorm(self.hidden_size, eps=1e-8)
            )
            self.attention_layers.append(
                nn.MultiheadAttention(
                    self.hidden_size, 
                    self.num_heads, 
                    dropout=self.dropout_rate,
                    batch_first=True
                )
            )
            
            self.forward_layernorms.append(
                nn.LayerNorm(self.hidden_size, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(self.hidden_size, self.dropout_rate)
            )
        
        self.last_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)
        self._init_weights()

        logger.info(f"Initialized SASRec with hidden_size={self.hidden_size}, "
                    f"num_blocks={self.num_blocks}, num_heads={self.num_heads}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param.data)
        if hasattr(self.item_embedding, 'weight'):
            self.item_embedding.weight.data[0, :] = 0
        if hasattr(self.pos_embedding, 'weight'):
            self.pos_embedding.weight.data[0, :] = 0
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create the causal attention mask."""
        return ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))

    def encode_sequence(self, item_seq: torch.Tensor, seq_lens: torch.Tensor,
                       v_feat_seq: Optional[torch.Tensor] = None,
                       t_feat_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode an item sequence with self-attention."""
        _ = seq_lens, v_feat_seq, t_feat_seq
        batch_size, seq_len = item_seq.size()
        seqs = self.item_embedding(item_seq)
        positions = torch.arange(1, seq_len + 1, device=self.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        positions = positions * (item_seq != 0).long()
        pos_embs = self.pos_embedding(positions)
        seqs = seqs + pos_embs
        seqs = self.emb_dropout(seqs)
        causal_mask = self.create_causal_mask(seq_len)
        for i in range(self.num_blocks):
            if self.norm_first:
                normed_seqs = self.attention_layernorms[i](seqs)
                attn_out, _ = self.attention_layers[i](
                    normed_seqs, normed_seqs, normed_seqs, 
                    attn_mask=causal_mask,
                    need_weights=False
                )
                seqs = seqs + attn_out
                normed_seqs = self.forward_layernorms[i](seqs)
                ff_out = self.forward_layers[i](normed_seqs)
                seqs = seqs + ff_out
            else:
                attn_out, _ = self.attention_layers[i](
                    seqs, seqs, seqs,
                    attn_mask=causal_mask,
                    need_weights=False
                )
                seqs = self.attention_layernorms[i](seqs + attn_out)
                ff_out = self.forward_layers[i](seqs)
                seqs = self.forward_layernorms[i](seqs + ff_out)
        return self.last_layernorm(seqs)
    
    def calculate_loss(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Loss is scored against the last-position representation. Default 'ce' is
        # the RecBole/community-standard full-catalogue softmax cross-entropy over
        # the whole item table (mirrors GRU4Rec/HM4SR; col 0 is PAD and is never a
        # positive target). 'bpr' is the opt-in single-negative pairwise variant.
        item_seqs = interaction['item_seqs']
        seq_lens = interaction['seq_lens']
        targets = interaction['targets']
        log_feats = self.encode_sequence(item_seqs, seq_lens)
        final_feats = self.gather_indexes(log_feats, seq_lens - 1)
        if self.loss_type == 'ce':
            scores = torch.matmul(final_feats, self.item_embedding.weight.t())
            return F.cross_entropy(scores, targets)
        # bpr requires a sampled negative and has no implicit CE fallback.
        if 'neg_items' not in interaction:
            raise ValueError(
                "SASRec loss_type='bpr' requires sampled neg_items in the batch; "
                "enable neg_sampling or use loss_type='ce'."
            )
        neg_items = interaction['neg_items']
        pos_logits = (final_feats * self.item_embedding(targets)).sum(dim=-1)
        neg_logits = (final_feats * self.item_embedding(neg_items)).sum(dim=-1)
        return -F.logsigmoid(pos_logits - neg_logits).mean()
    
    def full_sort_predict(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        item_seqs = interaction['item_seqs']
        seq_lens = interaction['seq_lens']
        log_feats = self.encode_sequence(item_seqs, seq_lens)
        final_feats = self.gather_indexes(log_feats, seq_lens - 1)
        return torch.matmul(final_feats, self.item_embedding.weight.t())
