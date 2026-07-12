# coding: utf-8
"""
Sequential Dataset for NexusRec
===============================

This module implements sequential dataset handling for sequence-based recommendation models.
It supports various data splitting strategies and provides efficient sequence data management.

Key Features:
- Automatic sequence construction from interaction data
- Multiple splitting strategies (leave-one-out, time-split, etc.)
- Multimodal sequence support
- Efficient batch processing

Author: NexusRec Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("nexusrec")


class SequentialDataset:
    """Sequential dataset for handling user interaction sequences."""
    
    def __init__(self, config, inter_feat: pd.DataFrame,
                 user_num: int = None, item_num: int = None):
        """
        Initialize sequential dataset.

        Args:
            config: Configuration dictionary
            inter_feat: Interaction features DataFrame
            user_num: Number of users
            item_num: Number of items
        """
        self.config = config
        self.inter_feat = inter_feat
        self.user_num = user_num
        self.item_num = item_num

        # Sequential configuration (flattened from sequential: group by config.py)
        self.max_seq_len = config["max_seq_len"]
        self.min_seq_len = config["min_seq_len"]
        self.split_strategy = config["split_method"]
        
        # Field names
        self.user_id_field = config["USER_ID_FIELD"]
        self.item_id_field = config["ITEM_ID_FIELD"]
        self.time_field = config["TIME_FIELD"]
        
        logger.info(f"SequentialDataset field mapping: user_id_field={self.user_id_field}, "
                   f"item_id_field={self.item_id_field}, time_field={self.time_field}")
        logger.info(f"Available columns: {list(self.inter_feat.columns)}")
        
        # Build user sequences
        self.user_seq = {}
        self.user_seq_len = {}
        self._user_id_list = []  # cached for O(1) __getitem__
        self._build_sequences()
        self._user_id_list = list(self.user_seq.keys())

        # Split data for train/valid/test
        self._split_sequences()
        
        # Compute statistics
        self._compute_statistics()

    @staticmethod
    def _sequence_entry(item_seq):
        return {
            'item_seq': item_seq,
            'seq_len': len(item_seq),
        }

    @staticmethod
    def _target_entry(context_seq, targets):
        return {
            'item_seq': context_seq,
            'target': targets[0],
            'targets': targets,
            'num_targets': len(targets),
            'seq_len': len(context_seq),
        }

    def _assign_train_entry(self, user_id, item_seq):
        if len(item_seq) >= 1:
            self.train_seq[user_id] = self._sequence_entry(item_seq)

    def _assign_eval_entry(self, split_store, user_id, context_seq, targets):
        if len(context_seq) >= 1 and targets:
            split_store[user_id] = self._target_entry(context_seq, targets)

    def _apply_leave_one_out_user(self, user_id, item_seq):
        seq_len = len(item_seq)
        if seq_len >= 3:
            train_context = item_seq[:-2]
            self._assign_train_entry(user_id, train_context)
            self._assign_eval_entry(self.valid_seq, user_id, train_context, [item_seq[-2]])
            self._assign_eval_entry(self.test_seq, user_id, item_seq[:-1], [item_seq[-1]])
            return len(train_context) >= 1
        if seq_len >= 2:
            train_context = item_seq[:-1]
            self._assign_train_entry(user_id, train_context)
            self._assign_eval_entry(self.test_seq, user_id, train_context, [item_seq[-1]])
            return len(train_context) >= 1
        return False
        
    def _build_sequences(self):
        """Build user interaction sequences from interaction data."""
        logger.info("Building user interaction sequences...")
        
        # Sort by user and timestamp
        if self.time_field in self.inter_feat.columns:
            sorted_inter = self.inter_feat.sort_values(
                by=[self.user_id_field, self.time_field], kind="stable"
            )
        else:
            # If no timestamp, use original order
            sorted_inter = self.inter_feat.sort_values(
                by=[self.user_id_field], kind="stable"
            )
        
        # Group by user and create sequences
        for user_id, group in sorted_inter.groupby(self.user_id_field):
            # PAD convention: index 0 is reserved exclusively for padding. Shift
            # real item ids by +1 at the data boundary so downstream
            # train/valid/test sequences all share the same item-id space.
            item_seq = [item_id + 1 for item_id in group[self.item_id_field].tolist()]
            
            # Filter sequences by length
            if len(item_seq) >= self.min_seq_len:
                self.user_seq[user_id] = {
                    'item_seq': item_seq,
                    'seq_len': len(item_seq)
                }
                
                # Add timestamp sequence if available
                if self.time_field in group.columns:
                    self.user_seq[user_id]['time_seq'] = group[self.time_field].tolist()
                
                # Store sequence length
                self.user_seq_len[user_id] = len(item_seq)
        
        self.active_user_num = len(self.user_seq)
        logger.info(f"Built sequences for {self.active_user_num} users (original user_num={self.user_num})")
        
    def _split_sequences(self):
        """Split sequences into train/valid/test sets."""
        logger.info(f"Splitting sequences using {self.split_strategy} strategy...")
        
        self.train_seq = {}
        self.valid_seq = {}
        self.test_seq = {}
        
        if self.split_strategy == 'leave_one_out':
            self._leave_one_out_split()
        elif self.split_strategy == 'ratio_split':
            self._ratio_based_split()
        elif self.split_strategy == 'temporal_ratio':
            self._temporal_ratio_split()
        elif self.split_strategy == 'hybrid':
            self._hybrid_split()
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
        
        logger.info(f"Split complete - Train: {len(self.train_seq)}, "
                   f"Valid: {len(self.valid_seq)}, Test: {len(self.test_seq)}")
    
    def _leave_one_out_split(self):
        """Leave-one-out splitting: last item for test, second last for valid."""
        training_users = 0
        total_users = len(self.user_seq)
        
        for user_id, seq_data in self.user_seq.items():
            if self._apply_leave_one_out_user(user_id, seq_data['item_seq']):
                training_users += 1
        
        logger.info(f"Training users with valid sequences: {training_users}/{total_users}")
    
    def _ratio_based_split(self):
        """Ratio-based splitting (e.g., 8:1:1) with single valid/test targets.

        Test target is always the last item (leave-one-out semantics for test).
        The valid target is the item at ``valid_end`` and MUST sit strictly
        before the test target (``valid_end < seq_len - 1``); otherwise the
        validation context/target would be identical to the test pair,
        leaking the test answer into early stopping.

        Bare ``int()`` truncation collapses the indices for short sequences
        (e.g. seq_len=5 -> train_end=valid_end=4), so we mirror the
        ``_temporal_ratio_split`` ``max()`` guards and fall back to plain
        leave-one-out for sequences too short to carve a distinct valid target.
        """
        train_ratio = self.config['train_ratio']
        valid_ratio = self.config['valid_ratio']

        for user_id, seq_data in self.user_seq.items():
            item_seq = seq_data['item_seq']
            seq_len = seq_data['seq_len']

            train_end = max(1, int(seq_len * train_ratio))
            valid_end = max(train_end + 1, int(seq_len * (train_ratio + valid_ratio)))

            # A distinct valid target requires train_end < valid_end < seq_len-1
            # (the last item is reserved for test). When the sequence is too
            # short for that, fall back to leave-one-out instead of dropping it
            # (length 3..5) or leaking the test target into validation (6..10).
            if train_end >= 1 and valid_end < seq_len - 1:
                self._assign_train_entry(user_id, item_seq[:train_end])
                self._assign_eval_entry(
                    self.valid_seq, user_id, item_seq[:valid_end], [item_seq[valid_end]]
                )
                self._assign_eval_entry(
                    self.test_seq, user_id, item_seq[:-1], [item_seq[-1]]
                )
            else:
                self._apply_leave_one_out_user(user_id, item_seq)
    
    def _temporal_ratio_split_user(self, user_id, item_seq, seq_len,
                                    train_ratio, valid_ratio):
        """Apply temporal ratio split to a single user's sequence.

        Returns True if the user was successfully split, False otherwise.
        Populates self.train_seq, self.valid_seq, self.test_seq for this user.
        """
        train_end = max(1, int(seq_len * train_ratio))
        valid_end = max(train_end + 1, int(seq_len * (train_ratio + valid_ratio)))
        valid_end = min(valid_end, seq_len)

        if train_end < 1 or train_end >= seq_len:
            return False

        # A distinct valid window requires at least one item left for test
        # AFTER it (valid_end < seq_len). When the sequence is too short for
        # that (seq_len 3-5, where valid_end collapses to seq_len), the test
        # fallback would copy item_seq[-1] into BOTH valid and test, leaking the
        # test answer into early stopping. Mirror _ratio_based_split: fall back
        # to leave-one-out instead of leaking.
        if valid_end < seq_len:
            self._assign_train_entry(user_id, item_seq[:train_end])
            self._assign_eval_entry(
                self.valid_seq, user_id, item_seq[:train_end], item_seq[train_end:valid_end]
            )
            self._assign_eval_entry(
                self.test_seq, user_id, item_seq[:valid_end], item_seq[valid_end:]
            )
            return True
        return self._apply_leave_one_out_user(user_id, item_seq)

    def _temporal_ratio_split(self):
        """Temporal ratio split: per-user chronological split with multi-target test.

        Unlike ratio_split which keeps only one target, this strategy returns
        the full validation and test subsequences so that downstream components
        can perform streaming or batch evaluation over multiple targets.

        References:
            - RecBole RS mode with group_by=user (per-user ratio split)
            - MMRec 80/10/10 preprocessing for dense users
        """
        train_ratio = self.config['train_ratio']
        valid_ratio = self.config['valid_ratio']

        for user_id, seq_data in self.user_seq.items():
            self._temporal_ratio_split_user(
                user_id, seq_data['item_seq'], seq_data['seq_len'],
                train_ratio, valid_ratio,
            )

    def _hybrid_split(self):
        """Hybrid split: leave-one-out for short sequences, temporal_ratio for long.

        Users with fewer than ``hybrid_threshold`` interactions use leave-one-out
        (safe for sparse users). Users at or above the threshold use temporal_ratio
        to produce multi-target test sets for richer evaluation.

        Reference: MMRec preprocessing/1splitting.ipynb
        """
        threshold = self.config['hybrid_threshold']
        n_leave, n_temporal = 0, 0

        for user_id, seq_data in self.user_seq.items():
            item_seq = seq_data['item_seq']
            seq_len = seq_data['seq_len']

            if seq_len < threshold:
                self._apply_leave_one_out_user(user_id, item_seq)
                n_leave += 1
            else:
                train_ratio = self.config['train_ratio']
                valid_ratio = self.config['valid_ratio']
                self._temporal_ratio_split_user(
                    user_id, item_seq, seq_len, train_ratio, valid_ratio,
                )
                n_temporal += 1

        logger.info(f"Hybrid split: {n_leave} users leave-one-out, "
                    f"{n_temporal} users temporal-ratio (threshold={threshold})")

    def _compute_statistics(self):
        """Compute dataset statistics."""
        all_seq_lens = list(self.user_seq_len.values())
        
        self.statistics = {
            'num_users': self.user_num,
            'num_items': self.item_num,
            'num_interactions': sum(all_seq_lens),
            'avg_seq_len': np.mean(all_seq_lens) if all_seq_lens else 0,
            'max_seq_len': max(all_seq_lens) if all_seq_lens else 0,
            'min_seq_len': min(all_seq_lens) if all_seq_lens else 0,
            'density': sum(all_seq_lens) / (self.user_num * self.item_num) if self.user_num and self.item_num else 0
        }
        
        logger.info(f"Dataset statistics: {self.statistics}")
    
    def __len__(self):
        """Return number of users with sequences."""
        return len(self.user_seq)
    
    def __getitem__(self, idx):
        """Get sequence data by index."""
        if idx >= len(self.user_seq):
            raise IndexError(f"Index {idx} out of range")
        
        user_id = self._user_id_list[idx]
        return user_id, self.user_seq[user_id]
    
    def get_user_num(self) -> int:
        """Get the number of users in the sequential dataset."""
        return self.user_num
    
    def get_item_num(self) -> int:
        """Get the number of items in the sequential dataset."""
        return self.item_num

    def compute_item_pop_freq(self) -> np.ndarray:
        """Per-item training frequency for the Novelty metric.

        Counts item occurrences across the TRAIN-split sequences (mirroring the
        centralized path, which counts training interactions) into an array of
        length ``item_num + 1`` indexed by global item id (index 0 = PAD = 0),
        then normalizes to sum to 1. Aligns with the evaluator's ``topk_index``
        ids in 1..item_num (the evaluator drops the PAD column at index 0).
        """
        counts = np.zeros(self.item_num + 1, dtype=np.float64)
        for entry in self.train_seq.values():
            for item_id in entry["item_seq"]:
                if 0 < item_id <= self.item_num:
                    counts[item_id] += 1
        total = counts.sum()
        if total > 0:
            counts /= total
        return counts
