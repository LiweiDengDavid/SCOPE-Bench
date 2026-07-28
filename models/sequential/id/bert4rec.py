# coding: utf-8
"""BERT4Rec sequential recommendation model."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple

from core.sequential import SequentialRecommender

logger = logging.getLogger("nexusrec")


class BertConfig:
    """Configuration for BERT4Rec."""
    
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 num_attention_heads: int = 2,
                 intermediate_size: int = 256,
                 hidden_act: str = "gelu",
                 hidden_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 max_position_embeddings: int = 50,
                 initializer_range: float = 0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.hidden_size}) must be divisible by "
                f"number of attention heads ({self.num_attention_heads})"
            )


class BertEmbeddings(nn.Module):
    """Item, position, and token-type embeddings."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        
        self.item_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
    
    def forward(self, input_ids: torch.Tensor, 
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, seq_len = input_ids.size()
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        item_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = item_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_dropout_rate)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    """Output layer for self-attention."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
    
    def forward(self, hidden_states: torch.Tensor, 
                input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """Complete attention layer (self-attention + output)."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for attention layer."""
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    """Intermediate (feed-forward) layer."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # Activation function
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = F.gelu
        elif config.hidden_act == "relu":
            self.intermediate_act_fn = F.relu
        elif config.hidden_act == "tanh":
            self.intermediate_act_fn = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for intermediate layer."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """Output layer for Transformer block."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
    
    def forward(self, hidden_states: torch.Tensor, 
                input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Complete Transformer layer."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for Transformer layer."""
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """Stack of Transformer layers."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_layers)])
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertModel(nn.Module):
    """BERT encoder for recommendation sequences."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Create extended attention mask for self-attention.
        
        Args:
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Extended mask [batch_size, 1, seq_len, seq_len]
        """
        extended_attention_mask = attention_mask[:, None, None, :].float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        return self.encoder(embedding_output, extended_attention_mask)


class BERT4Rec(SequentialRecommender):
    """BERT4Rec sequential recommendation model."""
    
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.bert_config = BertConfig(
            vocab_size=self.n_items + 2,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_attention_heads=config['num_attention_heads'],
            intermediate_size=config['intermediate_size'],
            hidden_act=config['hidden_act'],
            hidden_dropout_rate=config['hidden_dropout_rate'],
            attention_dropout_rate=config['attention_dropout_rate'],
            max_position_embeddings=config['max_seq_len'] + 1,
            initializer_range=config['initializer_range']
        )
        
        # Vocab layout (data is +1-shifted): 0 = PAD, 1..n_items = real items,
        # n_items+1 = MASK. vocab_size = n_items + 2 covers indices 0..n_items+1.
        self.pad_token_id = 0
        self.mask_token_id = self.n_items + 1
        self.mask_prob = config['mask_prob']
        self.max_predictions_per_seq = config['max_predictions_per_seq']
        # BERT4Rec supports only MLM training; read the declared loss_type and
        # raise on any other value so the config knob is honoured.
        self.loss_type = config['loss_type']
        if self.loss_type != 'mlm':
            raise ValueError(
                f"BERT4Rec supports loss_type 'mlm' only, got {self.loss_type!r}"
            )
        # MLM corruption split (Devlin et al.): of the masked positions, a
        # fraction become [MASK], a fraction become a random real item, and the
        # remainder are left unchanged. Kept in YAML (experiment-affecting
        # sampling numerics), not hardcoded.
        self.mask_replace_with_mask_prob = config['mask_replace_with_mask_prob']
        self.mask_replace_with_random_prob = config['mask_replace_with_random_prob']
        self._mask_random_threshold = (
            self.mask_replace_with_mask_prob + self.mask_replace_with_random_prob
        )
        self.bert = BertModel(self.bert_config)
        self.output_ffn = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.bert_config.hidden_size, eps=1e-12)
        self.lm_head = nn.Linear(self.bert_config.hidden_size, self.n_items + 2)
        # Tie the output head to the input item-embedding matrix (Sun et al.:
        # "shared item embedding matrix in the input and output layer"). Both
        # are [n_items + 2, hidden] over the same vocab (PAD/items/MASK); the
        # per-item output bias stays an independent parameter.
        self.lm_head.weight = self.bert.embeddings.item_embeddings.weight

        logger.info(
            "Initialized BERT4Rec with hidden_size=%d, num_layers=%d, vocab_size=%d",
            self.bert_config.hidden_size,
            self.bert_config.num_layers,
            self.bert_config.vocab_size,
        )

    def _prediction_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states into vocabulary logits."""
        hidden_states = self.output_ffn(hidden_states)
        hidden_states = self.output_gelu(hidden_states)
        hidden_states = self.output_ln(hidden_states)
        return self.lm_head(hidden_states)
    
    def encode_sequence(self, item_seq: torch.Tensor, seq_lens: torch.Tensor,
                       v_feat_seq: Optional[torch.Tensor] = None,
                       t_feat_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = v_feat_seq, t_feat_seq
        attention_mask = self.get_attention_mask(item_seq, seq_lens)
        # The data layer already applies the +1 PAD shift (real items 1..n_items,
        # index 0 = PAD), so item_seq is used directly as vocab ids: 0 = PAD,
        # 1..n_items = real items, n_items+1 = MASK. No model-side shift.
        return self.bert(
            input_ids=item_seq,
            attention_mask=attention_mask,
        )
    
    def create_masked_lm_predictions(self, item_seq: torch.Tensor, 
                                   seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = item_seq.size()
        masked_input = item_seq.clone()
        masked_lm_positions = torch.full((batch_size, self.max_predictions_per_seq), 
                                       -1, dtype=torch.long, device=item_seq.device)
        masked_lm_labels = torch.full((batch_size, self.max_predictions_per_seq), 
                                    -1, dtype=torch.long, device=item_seq.device)
        
        for i in range(batch_size):
            seq_len_i = seq_lens[i].item()
            
            valid_positions = list(range(seq_len_i))
            num_to_predict = min(self.max_predictions_per_seq, 
                               max(1, int(seq_len_i * self.mask_prob)))
            np.random.shuffle(valid_positions)
            masked_positions = valid_positions[:num_to_predict]
            
            for j, pos in enumerate(masked_positions):
                masked_lm_positions[i, j] = pos
                # Data is already +1-shifted (real items 1..n_items), so the
                # vocab label is the item id verbatim — no model-side shift.
                masked_lm_labels[i, j] = item_seq[i, pos]

                prob = np.random.random()
                if prob < self.mask_replace_with_mask_prob:
                    masked_input[i, pos] = self.mask_token_id
                elif prob < self._mask_random_threshold:
                    # Draw a real item id (1..n_items inclusive); 0 is PAD and
                    # would be misread as padding by encode_sequence. randint's
                    # upper bound is exclusive, so use n_items + 1.
                    masked_input[i, pos] = np.random.randint(1, self.n_items + 1)
        
        return masked_input, masked_lm_positions, masked_lm_labels
    
    def calculate_loss(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        item_seqs = interaction['item_seqs']
        seq_lens = interaction['seq_lens']
        targets = interaction['targets']
        # Append each sample's target after its prefix before masking (RecBole
        # reconstruct_train_data protocol): the FULL train context becomes a
        # model input and the last train item a maskable label — the sliding
        # window alone re-masks strict prefixes only, so neither ever occurs.
        # The +1 slot fits the max_seq_len + 1 position capacity reserved for
        # the eval-time appended MASK.
        num_rows = item_seqs.size(0)
        item_seqs = torch.cat((item_seqs, item_seqs.new_zeros(num_rows, 1)), dim=1)
        item_seqs[torch.arange(num_rows, device=item_seqs.device), seq_lens] = targets
        seq_lens = seq_lens + 1
        masked_input, masked_lm_positions, masked_lm_labels = self.create_masked_lm_predictions(
            item_seqs,
            seq_lens,
        )
        sequence_output = self.encode_sequence(masked_input, seq_lens)
        prediction_scores = self._prediction_logits(sequence_output)
        batch_size = masked_lm_positions.size(0)
        flat_positions = masked_lm_positions.view(-1)
        valid_positions = flat_positions >= 0
        if valid_positions.sum() == 0:
            raise ValueError(
                "BERT4Rec.calculate_loss received a batch with no maskable "
                "positions (all sequences empty) — check the data pipeline."
            )

        batch_indices = torch.arange(batch_size, device=item_seqs.device).unsqueeze(1)
        batch_indices = batch_indices.expand_as(masked_lm_positions).reshape(-1)
        valid_batch_indices = batch_indices[valid_positions]
        valid_positions_indices = flat_positions[valid_positions]
        predicted_scores = prediction_scores[valid_batch_indices, valid_positions_indices]
        target_labels = masked_lm_labels.view(-1)[valid_positions]
        return F.cross_entropy(predicted_scores, target_labels)
    
    def full_sort_predict(self, interaction: Dict[str, torch.Tensor]) -> torch.Tensor:
        item_seqs = interaction['item_seqs']
        seq_lens = interaction['seq_lens']
        batch_size, batch_width = item_seqs.size()
        # Append a [MASK] after each user's history and predict at that slot. Size the
        # eval tensor from each user's OWN history length (capped at the model's
        # max_seq_len+1 position capacity, which exists precisely to hold the appended
        # MASK), NOT the batch-local padded width. Using the batch width would force the
        # batch-longest user — whose history may be far below max_seq_len — to drop its
        # oldest item, making its scores depend on which users share the eval batch.
        target_width = min(int(seq_lens.max().item()) + 1, self.max_seq_len + 1)
        masked_seqs = item_seqs.new_zeros((batch_size, target_width))
        copy_width = min(batch_width, target_width)
        masked_seqs[:, :copy_width] = item_seqs[:, :copy_width]
        mask_positions = seq_lens.clamp(max=target_width - 1)
        masked_seqs[torch.arange(batch_size, device=item_seqs.device), mask_positions] = self.mask_token_id
        new_seq_lens = (seq_lens + 1).clamp(max=target_width)
        sequence_output = self.encode_sequence(masked_seqs, new_seq_lens)
        user_repr = self.gather_indexes(sequence_output, mask_positions)
        all_scores = self._prediction_logits(user_repr)
        # Data is +1-shifted (real items 1..n_items, index 0 = PAD), so logit
        # column c already scores item c. The evaluator reads column j as item j
        # (column 0 = PAD, masked there). Keep columns 0..n_items (drop only the
        # MASK column at n_items+1); width n_items+1 matches the eval contract.
        return all_scores[:, :self.n_items + 1]
