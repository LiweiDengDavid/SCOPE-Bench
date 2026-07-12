# coding: utf-8
"""
NexusRec Data Loaders
======================

Wraps :class:`RecDataset` into iterable batches for training and evaluation.
"""
import math
import logging

import numpy as np
import torch
from scipy.sparse import coo_matrix

from .dataset import RecDataset


class AbstractDataLoader(object):
    """Base class for all NexusRec data loaders.

    Wraps a :class:`RecDataset` and exposes iterable batches for training
    and evaluation.  Concrete subclasses implement the batch-building logic
    for specific recommendation paradigms (centralized, federated, sequential).

    Args:
        config: Runtime configuration dictionary.
        dataset: The :class:`RecDataset` to iterate over.
        additional_dataset: Secondary dataset used during evaluation
            (e.g. training interactions for seen-item filtering). Defaults to ``None``.
        batch_size (int): Number of interactions per batch. Defaults to ``1``.
        neg_sampling (bool): Whether to perform negative sampling. Defaults to ``False``.
        shuffle (bool): Whether to shuffle on each iteration. Defaults to ``False``.

    Attributes:
        dataset: Primary dataset.
        additional_dataset: Secondary dataset (evaluation only).
        batch_size (int): Interactions per batch.
        step (int): Pointer increment per batch (equals ``batch_size``).
        shuffle (bool): Shuffle flag.
        sparsity (float): Interaction sparsity of the primary dataset.
    """

    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, neg_sampling=False, shuffle=False):
        self.config = config
        self.logger = logging.getLogger("nexusrec")
        self.dataset = dataset
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.device = config['device']
        self.pr = 0
        self.inter_pr = 0

    def pretrain_setup(self):
        """This function can be used to deal with some problems after essential args are initialized,
        such as the batch-size-adaptation when neg-sampling is needed, and so on. By default, it will do nothing.
        """
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        """This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')


class TrainDataLoader(AbstractDataLoader):
    """
    General dataloader with negative sampling.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=None,
                         batch_size=batch_size, neg_sampling=True, shuffle=shuffle)
        self.dataset_bk = self.dataset.copy(self.dataset.df)

        # special for training dataloader
        self.history_items_per_u = dict()
        self.all_items = self.dataset.df[self.dataset.iid_field].unique().tolist()
        self.all_items_set = set(self.all_items)
        self.all_item_len = len(self.all_items)

        # Per-user negative sample pools: user_id -> list of available negative items
        self._user_neg_cache = {}
        sampling = config["sampling"]
        if sampling['use_neg_sampling']:
            self.sample_func = self._get_neg_sample
        else:
            self.sample_func = self._get_non_neg_sample

        self._get_history_items_u()

    def pretrain_setup(self):
        """Restore the working df to the un-shuffled backup so the first epoch
        starts from a deterministic order (when shuffle=True). Negatives are NOT
        fixed here: __iter__ clears the per-user negative cache and resamples them
        every epoch, and positives are cumulatively reshuffled per epoch.
        """
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)
        # Note: negatives are sampled from all_items_set (a set), so sorting or
        # shuffling the all_items list here would not affect sampling at all.

    def __iter__(self):
        # Resample each user's capped negative pool once per epoch so negatives
        # are not frozen to a single random subset for the whole training run.
        self._user_neg_cache = {}
        return super().__iter__()

    def get_resume_state(self):
        """Capture iteration-order state for bit-faithful training resume.

        The working df is shuffled cumulatively each epoch (``_shuffle`` ->
        ``dataset.shuffle``) and reset to the backup by ``pretrain_setup``, so the
        global RNG alone does not reproduce a mid-run epoch's batch order — the
        df's current row order must be restored explicitly.
        """
        return {"dataset_df": self.dataset.df.copy()}

    def set_resume_state(self, state):
        """Restore the cumulatively-shuffled working df captured at the epoch boundary."""
        self.dataset = self.dataset.copy(state["dataset_df"].copy())

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.

        Sparse matrix has shape (user_num, item_num).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.dataset.uid_field or not self.dataset.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.dataset.df, self.dataset.uid_field,
                                          self.dataset.iid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (pandas.DataFrame): Feature where src and tgt exist.
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(self.dataset.user_num, self.dataset.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        return self.sample_func()

    def _get_neg_sample(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        # to tensor
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).long().to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).long().to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        u_ids = cur_data[self.config['USER_ID_FIELD']]
        # sampling negative items only in the dataset (train); neg_ids is [K, N] (K = num_negatives)
        neg_ids = self._sample_neg_ids(u_ids).to(self.device)
        # merge negatives -> rows [users, pos_items, neg_1, ..., neg_K] -> (2 + K) x N
        batch_tensor = torch.cat((batch_tensor, neg_ids))
        return batch_tensor

    def _get_non_neg_sample(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        # to tensor
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).long().to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).long().to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        return batch_tensor

    def _sample_neg_ids(self, u_ids):
        # Honor num_negatives: draw K negatives per positive. Returns a [K, N]
        # tensor; K=1 preserves the single-negative [1, N] batch contract.
        num_negatives = self.config["num_negatives"]
        neg_ids = [self._sample_k_neg_for_user(u, num_negatives) for u in u_ids]
        return torch.as_tensor(np.asarray(neg_ids)).long().t().contiguous()

    def _sample_k_neg_for_user(self, user_id, k):
        available_items = self._build_user_negative_pool(user_id)
        if available_items.size == 0:
            raise ValueError(f"User {user_id} has no available negative items in the training item set.")
        replace = available_items.size < k
        return np.random.choice(available_items, size=k, replace=replace)

    def _build_user_negative_pool(self, user_id):
        if user_id in self._user_neg_cache:
            return self._user_neg_cache[user_id]

        user_history = self.history_items_per_u.get(user_id, set())
        available_items = list(self.all_items_set - user_history)
        neg_pool_cap = self.config["neg_pool_cap"]
        if len(available_items) > neg_pool_cap:
            available_items = np.random.choice(
                available_items,
                size=neg_pool_cap,
                replace=False,
            ).tolist()
        self._user_neg_cache[user_id] = np.asarray(available_items)
        return self._user_neg_cache[user_id]

    def _get_history_items_u(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u, u_ls in uid_freq:
            self.history_items_per_u[u] = set(u_ls.values)
        return self.history_items_per_u


class EvalDataLoader(AbstractDataLoader):
    """
        additional_dataset: history dataset masked at evaluation (the train
            split, or train+valid for the test loader under
            test_history_mask='train_valid')
        train_dataset: pure train split used as the popularity base for
            Novelty/item-bucket metrics; defaults to additional_dataset
    """

    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False, train_dataset=None):
        if shuffle:
            raise ValueError(
                "EvalDataLoader does not support shuffle=True: evaluation batches "
                "are served from tensors precomputed at construction, so shuffling "
                "the underlying df would silently change nothing."
            )
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, neg_sampling=False, shuffle=shuffle)

        if additional_dataset is None:
            raise ValueError('additional_dataset (training data) is required for evaluation')
        # Popularity base for Novelty/item-bucket metrics, decoupled from the
        # masking history: the test loader's additional_dataset may be
        # train+valid, but popularity must stay train-only so valid and test
        # metrics share one base (read by topk_kernel.compute_item_pop_count).
        self.train_dataset = train_dataset if train_dataset is not None else additional_dataset
        self.eval_items_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []
        # History entries dropped from the mask because the same (user, item)
        # pair also appears in the current eval split (cross-split re-purchase).
        self.cross_split_dup_count = 0

        self.eval_u = self.dataset.df[self.dataset.uid_field].unique()
        # special for eval dataloader
        self.pos_items_per_u = self._get_pos_items_per_u(self.eval_u).to(self.device)
        self._get_eval_items_per_u(self.eval_u)
        # to device
        self.eval_u = torch.tensor(self.eval_u).long().to(self.device)
        if self.cross_split_dup_count:
            self.logger.info(
                "EvalDataLoader: excluded %d cross-split duplicate (user, item) "
                "history entr%s from the eval mask so the current split's own "
                "targets stay rankable.",
                self.cross_split_dup_count,
                "y" if self.cross_split_dup_count == 1 else "ies",
            )

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        raise NotImplementedError(
            "EvalDataLoader cannot shuffle: iteration reads the eval_u/"
            "pos_items_per_u tensors precomputed in __init__, so shuffling the "
            "df alone would silently do nothing (and reordering eval_u without "
            "the mask tensors would corrupt evaluator alignment)."
        )

    def _next_batch_data(self):
        inter_cnt = sum(self.train_pos_len_list[self.pr: self.pr + self.step])
        batch_users = self.eval_u[self.pr: self.pr + self.step]
        batch_mask_matrix = self.pos_items_per_u[:, self.inter_pr: self.inter_pr + inter_cnt].clone()
        # Non-in-place offset to avoid corrupting self.pos_items_per_u
        batch_mask_matrix = torch.stack([
            batch_mask_matrix[0] - self.pr,
            batch_mask_matrix[1],
        ])
        self.inter_pr += inter_cnt
        self.pr += self.step

        return [batch_users, batch_mask_matrix]

    def _get_pos_items_per_u(self, eval_users):
        """
        history items in training dataset.
        masking out positive items in evaluation
        :return:
        user_id - item_ids matrix
        [[0, 0, ... , 1, ...],
         [0, 1, ... , 0, ...]]
        """
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        if isinstance(self.additional_dataset, RecDataset):
            additional_dataset = self.additional_dataset.df
        else:
            additional_dataset = self.additional_dataset

        uid_freq = additional_dataset.groupby(uid_field)[iid_field]
        uid_groups = uid_freq.groups
        # The shipped CSVs contain re-purchase rows where the same (user, item)
        # pair appears in the history AND the current eval split; masking those
        # items would set the split's own targets to -inf (structurally
        # unreachable regardless of model quality), so they are excluded here.
        eval_freq = self.dataset.df.groupby(uid_field)[iid_field]
        u_ids = []
        i_ids = []
        for i, u in enumerate(eval_users):
            if u in uid_groups:
                u_ls = uid_freq.get_group(u).values
                keep = ~np.isin(u_ls, eval_freq.get_group(u).values)
                self.cross_split_dup_count += int(u_ls.size - keep.sum())
                u_ls = u_ls[keep]
            else:
                u_ls = np.array([], dtype=int)
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i] * i_len)
            i_ids.extend(u_ls)
        return torch.tensor([u_ids, i_ids]).long()

    def _get_eval_items_per_u(self, eval_users):
        """
        get evaluated items for each u
        :return:
        """
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        uid_groups = uid_freq.groups
        for u in eval_users:
            if u in uid_groups:
                u_ls = uid_freq.get_group(u).values
            else:
                u_ls = np.array([], dtype=int)
            self.eval_len_list.append(len(u_ls))
            self.eval_items_per_u.append(u_ls)
        self.eval_len_list = np.asarray(self.eval_len_list)

    # return pos_items for each u
    def get_eval_items(self):
        return self.eval_items_per_u

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_users(self):
        return self.eval_u.cpu()
