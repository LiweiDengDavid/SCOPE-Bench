# coding: utf-8
"""HM4SR multimodal sequential recommendation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, List, Tuple

from core.sequential import SequentialRecommender


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        pos_emb = self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
        return x + pos_emb


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TransformerEncoder(nn.Module):
    """Multi-layer transformer encoder."""
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class AlignMoE(nn.Module):
    """Alignment Mixture of Experts for initial modal alignment."""
    
    def __init__(self, d_model: int, expert_num: int = 4):
        super().__init__()
        self.expert_num = expert_num
        self.d_model = d_model
        
        # Gates for each modality
        self.gate_id = nn.Linear(d_model, expert_num)
        self.gate_txt = nn.Linear(d_model, expert_num)
        self.gate_img = nn.Linear(d_model, expert_num)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Linear(d_model * 3, d_model * 3) 
            for _ in range(expert_num)
        ])
        
        # Modality weights
        self.modal_weights = nn.Parameter(torch.ones(3))

    def forward(self, multimodal_emb: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            multimodal_emb: [batch_size, seq_len, d_model * 3] (id + txt + img)
        Returns:
            List of enhanced embeddings for each modality
        """
        batch_size, seq_len, _ = multimodal_emb.size()
        
        # Split modalities
        id_emb = multimodal_emb[:, :, :self.d_model]
        txt_emb = multimodal_emb[:, :, self.d_model:2*self.d_model]
        img_emb = multimodal_emb[:, :, 2*self.d_model:]
        
        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(multimodal_emb).unsqueeze(-1))
        expert_outputs = torch.cat(expert_outputs, dim=-1)  # [batch, seq, d_model*3, expert_num]
        
        # Gate weights
        id_weights = F.softmax(self.gate_id(id_emb), dim=-1)   # [batch, seq, expert_num]
        txt_weights = F.softmax(self.gate_txt(txt_emb), dim=-1)
        img_weights = F.softmax(self.gate_img(img_emb), dim=-1)
        
        # Weighted expert outputs
        id_output = torch.sum(
            expert_outputs[:, :, :self.d_model, :] * id_weights.unsqueeze(2), 
            dim=-1
        )
        txt_output = torch.sum(
            expert_outputs[:, :, self.d_model:2*self.d_model, :] * txt_weights.unsqueeze(2), 
            dim=-1
        )
        img_output = torch.sum(
            expert_outputs[:, :, 2*self.d_model:, :] * img_weights.unsqueeze(2), 
            dim=-1
        )
        
        # Apply modality weights
        modal_weights = F.softmax(self.modal_weights, dim=0)
        return [
            modal_weights[0] * id_output,
            modal_weights[1] * txt_output,
            modal_weights[2] * img_output
        ]


class TemporalMoE(nn.Module):
    """Temporal Mixture of Experts for time-aware modeling."""
    
    def __init__(self, d_model: int, expert_num: int = 4):
        super().__init__()
        self.expert_num = expert_num
        self.d_model = d_model
        
        # Time embedding components
        self.time_gate = nn.Linear(d_model * 2, expert_num)  # time features + position
        self.absolute_proj = nn.Linear(1, d_model)
        self.rel_proj = nn.Linear(1, d_model)

        # Expert parameters (simplified as element-wise scaling)
        self.experts = nn.ParameterList([
            nn.Parameter(torch.ones(d_model * 3)) 
            for _ in range(expert_num)
        ])

    def get_time_embedding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Generate time-aware embeddings.
        Args:
            positions: [batch_size, seq_len] - sequence positions or timestamps
        """
        batch_size, seq_len = positions.size()
        
        # Absolute time embedding (simplified)
        abs_emb = torch.cos(self.absolute_proj(positions.unsqueeze(-1).float()))
        
        # Relative embedding: log(offset from first position in sequence).
        # Using cumulative offset instead of consecutive differences avoids
        # log(1)=0 degeneration when positions are sequential integers (0,1,2,...).
        # For sequential positions [0,1,...,N-1]: offsets=[1,2,...,N-1], log=[0,0.69,1.10,...].
        # For actual timestamps: offsets = elapsed time since first event (log-scaled).
        rel_emb = torch.zeros_like(abs_emb)
        if seq_len > 1:
            offsets = (positions[:, 1:] - positions[:, 0:1]).float().clamp(min=1)
            rel_features = torch.log(offsets).unsqueeze(-1)
            rel_emb[:, 1:] = self.rel_proj(rel_features)
        
        return torch.cat([abs_emb, rel_emb], dim=-1)

    def forward(self, multimodal_emb: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            multimodal_emb: [batch_size, seq_len, d_model * 3]
            positions: [batch_size, seq_len] - sequence positions
        """
        # Time-aware gating
        time_emb = self.get_time_embedding(positions)
        gate_weights = F.softmax(self.time_gate(time_emb), dim=-1)
        
        # Expert mixing
        enhanced_emb = torch.zeros_like(multimodal_emb)
        for i, expert in enumerate(self.experts):
            expert_out = multimodal_emb * expert.unsqueeze(0).unsqueeze(0)
            enhanced_emb += expert_out * gate_weights[:, :, i].unsqueeze(-1)
        
        # Split back to modalities
        id_emb = enhanced_emb[:, :, :self.d_model]
        txt_emb = enhanced_emb[:, :, self.d_model:2*self.d_model]
        img_emb = enhanced_emb[:, :, 2*self.d_model:]
        
        return id_emb, txt_emb, img_emb


class HM4SR(SequentialRecommender):
    """HM4SR hierarchical multimodal sequential recommendation model."""
    
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)
        self.hidden_size = config['hidden_size']
        self.n_layers = config['num_layers']
        self.n_heads = config['num_attention_heads']
        self.inner_size = config['inner_size']
        self.dropout_prob = config['dropout_rate']
        self.start_expert_num = config['start_expert_num']
        self.temporal_expert_num = config['temporal_expert_num']
        self.temperature = config['temperature']
        self.use_contrastive = config['use_contrastive']
        self.ssl_weight = config['ssl_weight']
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_encoding = PositionalEncoding(self.hidden_size, self.max_seq_len)
        self._init_multimodal_projections()
        self.align_moe = AlignMoE(self.hidden_size, self.start_expert_num)
        self.temporal_moe = TemporalMoE(self.hidden_size, self.temporal_expert_num)
        self.id_encoder = TransformerEncoder(
            self.n_layers, self.hidden_size, self.n_heads, self.inner_size, self.dropout_prob
        )
        self.txt_encoder = TransformerEncoder(
            self.n_layers, self.hidden_size, self.n_heads, self.inner_size, self.dropout_prob
        )
        self.img_encoder = TransformerEncoder(
            self.n_layers, self.hidden_size, self.n_heads, self.inner_size, self.dropout_prob
        )
        
        self.id_ln = nn.LayerNorm(self.hidden_size)
        self.txt_ln = nn.LayerNorm(self.hidden_size)
        self.img_ln = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _init_multimodal_projections(self):
        """Initialize multimodal feature projections."""
        if hasattr(self, 'v_feat') and self.v_feat is not None:
            self.visual_projection = nn.Linear(self.v_feat.size(1), self.hidden_size)
        else:
            self.visual_projection = nn.Linear(self.hidden_size, self.hidden_size)
        if hasattr(self, 't_feat') and self.t_feat is not None:
            self.text_projection = nn.Linear(self.t_feat.size(1), self.hidden_size)
        else:
            self.text_projection = nn.Linear(self.hidden_size, self.hidden_size)

    def _project_item_features(
        self,
        item_seq: torch.Tensor,
        feat_name: str,
        projection: nn.Linear,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """Project one modality or fall back to the ID embedding."""
        feat = getattr(self, feat_name)
        if feat is None:
            return projection(fallback)
        # item_seq carries the +1-shifted PAD convention (1..n_items, 0 = PAD),
        # but feature rows are 0-indexed by raw item id. Map shifted id s -> raw
        # row s-1; PAD (0) clamps to row 0 and is dropped by the padding mask.
        feat_index = (item_seq - 1).clamp(min=0)
        return projection(feat[feat_index].to(item_seq.device).float())

    @staticmethod
    def _score_targets(seq_output: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Score all candidate items from one sequence representation."""
        return torch.matmul(seq_output, item_embeddings.transpose(0, 1))

    def get_multimodal_embeddings(self, item_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get ID, text, and image embeddings for one item sequence."""
        id_emb = self.item_embedding(item_seq)
        visual_emb = self._project_item_features(item_seq, 'v_feat', self.visual_projection, id_emb)
        text_emb = self._project_item_features(item_seq, 't_feat', self.text_projection, id_emb)
        return id_emb, text_emb, visual_emb
    
    def encode_sequence(self, item_seq: torch.Tensor, seq_lens: torch.Tensor,
                       v_feat_seq: Optional[torch.Tensor] = None,
                       t_feat_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode one item sequence with hierarchical MoE."""
        _ = v_feat_seq, t_feat_seq
        batch_size, seq_len = item_seq.size()
        id_emb, txt_emb, img_emb = self.get_multimodal_embeddings(item_seq)
        id_emb = self.position_encoding(id_emb)
        txt_emb = self.position_encoding(txt_emb)
        img_emb = self.position_encoding(img_emb)
        multimodal_concat = torch.cat([id_emb, txt_emb, img_emb], dim=-1)
        align_enhanced = self.align_moe(multimodal_concat)
        id_emb += align_enhanced[0]
        txt_emb += align_enhanced[1]
        img_emb += align_enhanced[2]
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        multimodal_concat = torch.cat([id_emb, txt_emb, img_emb], dim=-1)
        id_emb, txt_emb, img_emb = self.temporal_moe(multimodal_concat, positions)
        id_emb = self.dropout(self.id_ln(id_emb))
        txt_emb = self.dropout(self.txt_ln(txt_emb))
        img_emb = self.dropout(self.img_ln(img_emb))
        mask = self.create_attention_mask(item_seq)
        id_encoded = self.id_encoder(id_emb, mask)
        txt_encoded = self.txt_encoder(txt_emb, mask)
        img_encoded = self.img_encoder(img_emb, mask)
        id_final = self.gather_last_relevant_hidden_states(id_encoded, seq_lens)
        txt_final = self.gather_last_relevant_hidden_states(txt_encoded, seq_lens)
        img_final = self.gather_last_relevant_hidden_states(img_encoded, seq_lens)
        return id_final + txt_final + img_final
    
    def create_attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Create causal attention mask with padding."""
        _, seq_len = item_seq.size()
        padding_mask = (item_seq != 0).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=item_seq.device)).unsqueeze(0).unsqueeze(0)
        return padding_mask * causal_mask
    
    def gather_last_relevant_hidden_states(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """Gather hidden states at the last relevant position."""
        indices = (seq_lens - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1)
        indices = indices.expand(-1, -1, hidden_states.size(-1))
        return torch.gather(hidden_states, 1, indices).squeeze(1)
    
    def calculate_loss(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Single-path next-item CE on the framework's held-out `targets`,
        # plus the optional contrastive term.
        item_seqs = interaction['item_seqs']
        seq_lens = interaction['seq_lens']
        targets = interaction['targets']
        seq_output = self.encode_sequence(item_seqs, seq_lens)
        scores = self._score_targets(seq_output, self.item_embedding.weight)
        loss = self.criterion(scores, targets)
        if self.use_contrastive:
            loss = loss + self.ssl_weight * self.contrastive_learning_loss(seq_output, targets)
        return loss
    
    def contrastive_learning_loss(self, seq_output: torch.Tensor, target_items: torch.Tensor) -> torch.Tensor:
        """Compute contrastive learning loss."""
        seq_output = F.normalize(seq_output, dim=1)
        target_emb = F.normalize(self.item_embedding(target_items), dim=1)
        pos_logits = (seq_output * target_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)
        neg_logits = torch.matmul(seq_output, target_emb.transpose(0, 1)) / self.temperature
        mask = torch.eye(seq_output.size(0), device=seq_output.device, dtype=torch.bool)
        neg_logits = neg_logits.masked_fill(mask, float('-inf'))
        neg_logits = torch.exp(neg_logits).sum(dim=1)
        return (-torch.log(pos_logits / (pos_logits + neg_logits))).mean()
    
    def full_sort_predict(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        item_seq = interaction['item_seqs']
        seq_lens = interaction['seq_lens']
        seq_output = self.encode_sequence(item_seq, seq_lens)
        return self._score_targets(seq_output, self.item_embedding.weight)
