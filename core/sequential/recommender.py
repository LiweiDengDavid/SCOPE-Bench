# coding: utf-8
"""
Sequential Recommender Base Class for NexusRec
=============================================

This module provides the base class for all sequential recommendation models,
extending RecommenderBase with sequence-specific functionality.

Key Features:
- Sequence encoding interface (implemented by each concrete model)
- Attention mask generation
- Index gathering for last-position pooling
- Shared eval-time forward dispatcher

Author: NexusRec Team
Version: 2.0.0
"""

import torch
from typing import Optional
import logging

from ..base.recommender import RecommenderBase

logger = logging.getLogger("nexusrec")


class SequentialRecommender(RecommenderBase):
    """
    Base class for sequential recommendation models.
    
    This class extends RecommenderBase with sequence-specific functionality
    and provides common interfaces for sequential models.
    """
    
    def __init__(self, config, dataloader):
        """
        Initialize sequential recommender.
        
        Args:
            config: Configuration dictionary
            dataloader: Data loader instance
        """
        super().__init__(config, dataloader)
        
        # Sequential-specific configuration (read by subclasses, e.g. GRU4Rec
        # uses dropout_rate; SASRec/HM4SR use hidden_size).
        self.max_seq_len = config["max_seq_len"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]

        # NOTE: the base class deliberately does NOT create item_embedding /
        # layer_norm / dropout modules. Every concrete model owns its own
        # embedding table (SASRec/GRU4Rec/HM4SR reassign self.item_embedding;
        # BERT4Rec embeds via its nested BERT) and its own norm/dropout, so a
        # base-created copy received no gradient and only bloated every
        # checkpoint with a phantom [n_items+1, hidden] table.

        logger.info(f"Initialized SequentialRecommender with max_seq_len={self.max_seq_len}, "
                   f"hidden_size={self.hidden_size}")

    def get_attention_mask(self, item_seq: torch.Tensor, seq_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate attention mask for sequences.
        
        Args:
            item_seq: Item sequence tensor [batch_size, seq_len]
            seq_lens: Actual sequence lengths [batch_size] (optional)
            
        Returns:
            Attention mask [batch_size, seq_len]
        """
        batch_size, seq_len = item_seq.size()
        
        if seq_lens is not None:
            # Create mask based on actual sequence lengths
            mask = torch.arange(seq_len, device=item_seq.device).expand(
                batch_size, seq_len
            ) < seq_lens.unsqueeze(1)
        else:
            # Create mask based on non-padding tokens (assuming 0 is padding)
            mask = (item_seq != 0)
        
        return mask
    
    def gather_indexes(self, output: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
        """
        Gather hidden states from output using indices.
        
        Args:
            output: Output tensor [batch_size, seq_len, hidden_size]
            gather_index: Indices to gather [batch_size]
            
        Returns:
            Gathered output [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = output.size()

        # Gather indices must point inside the current sequence length.
        assert ((gather_index >= 0) & (gather_index < seq_len)).all(), \
            "gather_indexes received an out-of-range index"

        # Expand gather_index for gathering
        gather_index = gather_index.view(batch_size, 1, 1).expand(-1, 1, hidden_size)
        
        # Gather output
        gathered_output = torch.gather(output, dim=1, index=gather_index)
        
        return gathered_output.squeeze(1)
    
    def encode_sequence(self, item_seq: torch.Tensor, seq_lens: torch.Tensor,
                       v_feat_seq: Optional[torch.Tensor] = None,
                       t_feat_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode item sequence to get sequence representation.

        This is an abstract method that should be implemented by subclasses.

        Args:
            item_seq: Item sequence [batch_size, seq_len]
            seq_lens: Actual sequence lengths [batch_size]
            v_feat_seq: Visual feature sequence [batch_size, seq_len, v_dim] (optional)
            t_feat_seq: Text feature sequence [batch_size, seq_len, t_dim] (optional)

        Returns:
            Sequence representation [batch_size, seq_len, hidden_size]
        """
        raise NotImplementedError("encode_sequence must be implemented by subclasses")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Eval-time dispatcher shared by attention/transformer subclasses.

        Satisfies the ``forward`` abstractmethod and delegates to
        ``full_sort_predict``. Subclasses with a stateful forward (e.g. GRU4Rec)
        override this with their real implementation.
        """
        if len(args) >= 2:
            item_seqs, seq_lens = args[0], args[1]
            interaction = {'item_seqs': item_seqs, 'seq_lens': seq_lens}
            return self.full_sort_predict(interaction)
        elif 'interaction' in kwargs:
            return self.full_sort_predict(kwargs['interaction'])
        else:
            raise ValueError("Invalid forward call. Expected item_seqs and seq_lens or interaction dict.")
