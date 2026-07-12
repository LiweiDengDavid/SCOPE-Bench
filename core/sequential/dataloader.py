# coding: utf-8
"""
Sequential DataLoader for NexusRec
==================================

This module implements the data loader for sequential recommendation,
handling batch generation, padding, and data augmentation.

Author: NexusRec Team
Version: 2.0.0
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
import random
import logging

import numpy as np

logger = logging.getLogger("nexusrec")


def _seed_dataloader_worker(worker_id):
    """Re-seed NumPy and Python ``random`` inside each DataLoader worker.

    PyTorch seeds torch and Python's ``random`` per worker from the base seed,
    but NOT NumPy. Defined at module level so it is picklable under the 'spawn'
    start method (the default on macOS). This makes any NumPy/random use in
    collation or augmentation reproducible across runs with the same seed.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SequentialTorchDataset(Dataset):
    """PyTorch dataset wrapper around the framework sequential dataset."""
    
    def __init__(self, sequential_dataset, mode: str = 'train', 
                 augmentation: bool = False, neg_sampling: bool = True):
        """
        Initialize sequential dataset wrapper.
        
        Args:
            sequential_dataset: framework SequentialDataset instance
            mode: 'train', 'valid', or 'test'
            augmentation: Whether to apply data augmentation
            neg_sampling: Whether to perform negative sampling
        """
        self.dataset = sequential_dataset
        self.mode = mode
        self.augmentation = augmentation
        self.neg_sampling = neg_sampling
        
        # Get appropriate sequences based on mode
        if mode == 'train':
            self.sequences = self.dataset.train_seq
        elif mode == 'valid':
            self.sequences = self.dataset.valid_seq
        elif mode == 'test':
            self.sequences = self.dataset.test_seq
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.user_ids = list(self.sequences.keys())
        self.max_seq_len = self.dataset.max_seq_len
        self.item_num = self.dataset.item_num
        self._negative_pools = {}
        # Built only when sampling is active (train mode with neg_sampling on);
        # eval-mode instances never sample, so they skip the per-user set build.
        self._user_histories = None
        if self.neg_sampling:
            # Exclude against the TRAIN split only: user_seq is the full
            # pre-split sequence including the held-out LOO valid/test targets,
            # so excluding it would leak the eval answers into the sampler
            # (the centralized sampler also excludes train-only history).
            self._user_histories = {
                user_id: set(seq_data['item_seq'])
                for user_id, seq_data in self.dataset.train_seq.items()
            }

        # Augmentation config from YAML (via dataset)
        self.config = self.dataset.config
        self.augment_strategies = self.config["augmentation_strategies"]
        self.augment_mask_ratio = self.config["augmentation_mask_ratio"]
        self.augment_max_mask = self.config["augmentation_max_mask"]
        self.augment_min_seq_len = self.config["augmentation_min_seq_len"]

        # Build sliding-window training samples (RecBole-compatible protocol).
        # Each position i in [1, len(seq)) produces one sample:
        #   input = seq[:i],  target = seq[i]
        # This is the standard data augmentation for SASRec / BERT4Rec.
        self._train_samples = None
        if mode == 'train':
            self._train_samples = self._build_sliding_window_samples()
            logger.info(
                "Built %d sliding-window training samples from %d users",
                len(self._train_samples), len(self.user_ids),
            )

    def _build_sliding_window_samples(self):
        """Expand per-user sequences into (user_id, input_seq, target) triples."""
        samples = []
        for user_id in self.user_ids:
            seq_data = self.sequences[user_id]
            item_seq = (
                seq_data['item_seq'].copy()
                if isinstance(seq_data['item_seq'], list)
                else seq_data['item_seq'].tolist()
            )
            for i in range(1, len(item_seq)):
                samples.append((user_id, item_seq[:i], item_seq[i]))
        return samples

    @staticmethod
    def _normalize_item_seq(item_seq):
        return item_seq.copy() if isinstance(item_seq, list) else item_seq.tolist()

    def _truncate_sequence(self, item_seq: List[int]) -> List[int]:
        if len(item_seq) > self.max_seq_len:
            return item_seq[-self.max_seq_len:]
        return item_seq

    def __len__(self):
        if self._train_samples is not None:
            return len(self._train_samples)
        return len(self.user_ids)

    def __getitem__(self, idx):
        # --- Training: sliding-window samples ---
        if self._train_samples is not None:
            user_id, input_seq, target = self._train_samples[idx]

            if self.augmentation:
                input_seq = self._augment_sequence(list(input_seq))
            input_seq = self._truncate_sequence(input_seq)

            neg_item = None
            if self.neg_sampling:
                neg_item = self._sample_negative(user_id)

            return {
                'user_id': user_id,
                'item_seq': list(input_seq),
                'target': target,
                'neg_item': neg_item,
                'seq_len': len(input_seq),
            }

        # --- Eval: one sample per user ---
        user_id = self.user_ids[idx]
        seq_data = self.sequences[user_id]
        full_seq = self._normalize_item_seq(seq_data['item_seq'])
        input_seq = self._truncate_sequence(full_seq)
        target = seq_data['target']
        targets = seq_data['targets']
        num_targets = seq_data['num_targets']

        return {
            'user_id': user_id,
            'item_seq': input_seq,
            # Untruncated context so the evaluator can filter the FULL history,
            # not just the (truncated) model window fed via 'item_seq'.
            'full_item_seq': full_seq,
            'target': target,
            'targets': targets,
            'num_targets': num_targets,
            'neg_item': None,
            'seq_len': len(input_seq),
        }
    
    def _augment_sequence(self, seq: List[int]) -> List[int]:
        """Apply random data augmentation (crop / mask / reorder) to a sequence."""
        strategy = random.choice(self.augment_strategies)

        min_len = self.augment_min_seq_len

        if strategy == 'crop' and len(seq) > min_len:
            crop_len = random.randint(min_len, len(seq))
            start_idx = random.randint(0, len(seq) - crop_len)
            return seq[start_idx:start_idx + crop_len]

        elif strategy == 'mask' and len(seq) > 2:
            seq = seq.copy()
            num_mask = min(int(len(seq) * self.augment_mask_ratio), self.augment_max_mask)
            mask_indices = random.sample(range(len(seq)), num_mask)
            for idx in mask_indices:
                # Real items occupy 1..item_num (index 0 is PAD). randint's upper
                # bound is inclusive, so item_num must be reachable.
                seq[idx] = random.randint(1, self.item_num)  # 0 is PAD token
            return seq

        elif strategy == 'reorder' and len(seq) > min_len:
            seq = seq.copy()
            reorder_len = min(min_len, len(seq) // 2)
            start_idx = random.randint(0, len(seq) - reorder_len)
            subsequence = seq[start_idx:start_idx + reorder_len]
            random.shuffle(subsequence)
            seq[start_idx:start_idx + reorder_len] = subsequence
            return seq

        return seq
    
    def _sample_negative(self, user_id: int) -> int:
        if user_id not in self._negative_pools:
            interacted_items = self._user_histories[user_id]
            # Real items occupy 1..item_num inclusive (index 0 is PAD); range's
            # stop is exclusive, so use item_num + 1 to cover the top item.
            candidate_pool = list(set(range(1, self.item_num + 1)) - interacted_items)
            if not candidate_pool:
                raise ValueError(f"User {user_id} has no valid sequential negative items.")
            # Cap the cached per-user pool, mirroring the centralized sampler's
            # memory cap (core/data/dataloader.py _build_user_negative_pool);
            # an uncapped ~item_num list per user is tens of GB at catalog scale.
            neg_pool_cap = self.config["neg_pool_cap"]
            if len(candidate_pool) > neg_pool_cap:
                candidate_pool = np.random.choice(
                    candidate_pool,
                    size=neg_pool_cap,
                    replace=False,
                ).tolist()
            self._negative_pools[user_id] = candidate_pool
        return random.choice(self._negative_pools[user_id])


def sequential_data_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(sample['seq_len'] for sample in batch)
    user_ids = torch.zeros(batch_size, dtype=torch.long)
    item_seqs = torch.zeros((batch_size, max_len), dtype=torch.long)
    target_ids = torch.zeros(batch_size, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)

    has_neg = batch[0]['neg_item'] is not None
    if has_neg:
        neg_items = torch.zeros(batch_size, dtype=torch.long)

    has_multi_targets = 'targets' in batch[0]
    max_num_targets = max(sample['num_targets'] for sample in batch) if has_multi_targets else 0
    if has_multi_targets:
        targets_padded = torch.zeros((batch_size, max_num_targets), dtype=torch.long)
        num_targets_tensor = torch.zeros(batch_size, dtype=torch.long)

    has_full_hist = 'full_item_seq' in batch[0]
    if has_full_hist:
        max_full_len = max(len(sample['full_item_seq']) for sample in batch)
        full_item_seqs = torch.zeros((batch_size, max_full_len), dtype=torch.long)
        full_seq_lens = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        user_ids[i] = sample['user_id']
        seq_len = sample['seq_len']
        seq_lens[i] = seq_len
        target_ids[i] = sample['target']

        if seq_len > 0:
            item_seqs[i, :seq_len] = torch.tensor(sample['item_seq'], dtype=torch.long)

        if has_neg:
            neg_items[i] = sample['neg_item']

        if has_multi_targets:
            t_list = sample['targets']
            nt = len(t_list)
            num_targets_tensor[i] = nt
            targets_padded[i, :nt] = torch.tensor(t_list, dtype=torch.long)

        if has_full_hist:
            full_seq = sample['full_item_seq']
            full_len = len(full_seq)
            full_seq_lens[i] = full_len
            if full_len > 0:
                full_item_seqs[i, :full_len] = torch.tensor(full_seq, dtype=torch.long)

    result = {
        'user_ids': user_ids,
        'item_seqs': item_seqs,
        'targets': target_ids,
        'seq_lens': seq_lens,
    }

    if has_neg:
        result['neg_items'] = neg_items

    if has_multi_targets:
        result['targets_list'] = targets_padded
        result['num_targets'] = num_targets_tensor

    if has_full_hist:
        result['full_item_seqs'] = full_item_seqs
        result['full_seq_lens'] = full_seq_lens

    return result


class SequentialDataLoader:
    """Data loader for sequential recommendation."""
    
    def __init__(self, config, dataset, mode: str = 'train', 
                 batch_size: Optional[int] = None, shuffle: Optional[bool] = None):
        """
        Initialize sequential data loader.
        
        Args:
            config: Configuration dictionary
            dataset: framework SequentialDataset instance
            mode: 'train', 'valid', or 'test'
            batch_size: Batch size (overrides config if provided)
            shuffle: Whether to shuffle data (overrides default if provided)
        """
        self.config = config
        self.dataset = dataset
        self.mode = mode
        
        if batch_size is not None:
            self.batch_size = batch_size
        elif mode == 'train':
            self.batch_size = config["train_batch_size"]
        else:
            self.batch_size = config["eval_batch_size"]

        if shuffle is not None:
            self.shuffle = shuffle
        else:
            self.shuffle = (mode == 'train')
        self.augmentation = (mode == 'train') and config['data_augmentation']
        self.neg_sampling = (mode == 'train') and config['neg_sampling']
        self.torch_dataset = SequentialTorchDataset(
            dataset, 
            mode=mode,
            augmentation=self.augmentation,
            neg_sampling=self.neg_sampling
        )

        if mode == 'train' and len(self.torch_dataset) < self.batch_size:
            logger.warning(
                "Training batch size (%d) is larger than sliding-window sample count (%d). Reducing batch size.",
                self.batch_size,
                len(self.torch_dataset),
            )
            self.batch_size = len(self.torch_dataset)
        
        logger.info(
            f"SequentialTorchDataset for {mode} mode has {len(self.torch_dataset)} samples"
        )
        drop_last = (mode == 'train') and (len(self.torch_dataset) > self.batch_size)
        # Pin shuffle order and per-worker RNG to config['seed'] so the sequential
        # data pipeline is reproducible even with num_workers > 0.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(config['seed']))
        self.dataloader = DataLoader(
            self.torch_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=sequential_data_collate,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            drop_last=drop_last,
            generator=loader_generator,
            worker_init_fn=_seed_dataloader_worker,
        )
        
        logger.info(f"Created SequentialDataLoader for {mode} mode: "
                   f"batch_size={self.batch_size}, shuffle={self.shuffle}, "
                   f"augmentation={self.augmentation}, neg_sampling={self.neg_sampling}, "
                   f"drop_last={drop_last}, dataloader_batches={len(self.dataloader)}")
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Number of batches."""
        return len(self.dataloader)
    
    def get_dataloader(self):
        """Get the underlying PyTorch DataLoader."""
        return self.dataloader
    
    def pretrain_setup(self):
        """Framework hook for pre-training setup."""
        # For sequential models, no special pretrain setup is needed
        pass

    def get_resume_state(self):
        """Capture the DataLoader shuffle generator for bit-faithful resume.

        The loader uses an explicit ``torch.Generator`` (seeded once at
        construction) for shuffling; it advances across epochs and is NOT the
        global torch RNG, so it must be captured/restored separately.
        """
        return {"generator": self.dataloader.generator.get_state()}

    def set_resume_state(self, state):
        """Restore the shuffle generator state captured at the epoch boundary."""
        self.dataloader.generator.set_state(state["generator"])
