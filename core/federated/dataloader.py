from abc import ABC, abstractmethod

from ..data import EvalDataLoader, TrainDataLoader
from ..data.dataset import RecDataset


class FederatedDataLoaderBase(ABC):
    """Protocol for federated data loaders.

    A federated data loader manages per-user sub-loaders rather than
    yielding batches directly.  The trainer iterates over users and
    fetches the corresponding per-user loader for local training.

    Concrete loaders must expose:
      - ``loaders``: mapping from user_id → per-user :class:`TrainDataLoader`
      - ``user_set``: ordered list of user IDs managed by this loader
    """

    @property
    @abstractmethod
    def loaders(self) -> dict:
        """Map from user_id to the per-user data loader."""

    @property
    @abstractmethod
    def user_set(self) -> list:
        """Ordered list of user IDs managed by this loader."""


class FederatedDataLoader(FederatedDataLoaderBase):
    """Federated learning data loader.

    Args:
        config (Config): Configuration object.
        dataset (Dataset): Dataset object.
        batch_size (int, optional): Batch size. Defaults to 1.
        neg_sampling (bool, optional): Whether to perform negative sampling. Defaults to False.
        shuffle (bool, optional): Whether to shuffle data. Defaults to False.
        stage (str, optional): Stage, one of 'train', 'valid', 'test'. Defaults to 'train'.
        additional_dataset (Dataset, optional): Additional dataset used during evaluation. Defaults to None.
        train_dataset (Dataset, optional): Pure train split used as the popularity
            base for Novelty/item-bucket metrics. Falls back to
            ``additional_dataset`` (correct for the valid stage, whose
            additional IS the train split). Defaults to None.
    """

    def __init__(
        self,
        config,
        dataset,
        batch_size=1,
        neg_sampling=False,
        shuffle=False,
        stage="train",
        additional_dataset=None,
        train_dataset=None,
    ):
        self.config = config
        self.dataset = dataset

        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling

        self.additional_dataset = additional_dataset
        self.train_dataset = (
            train_dataset if train_dataset is not None else additional_dataset
        )
        self.stage = stage

        # P2 improvement: add per-user dataset cache to avoid repeated creation
        self._user_datasets_cache = {}
        self._user_loaders_cache = {}

        # Initialize data loader
        self.data_loader = self._get_federated_loader()

        # List of user IDs for iteration
        self.user_ids = list(self.data_loader.keys())
        self.user_idx = 0

        # Pre-compute eval data (avoids repeated groupby per evaluation call)
        self._eval_items = None
        self._eval_len_list = None

    def _get_federated_loader(self):
        """Get the federated data loader.

        Returns:
            dict: Mapping from user ID to data loader.
        """
        if self.stage == "train":
            return self._get_train_loader()
        elif self.stage in ["valid", "test", "eval"]:
            return self._get_eval_loader()
        else:
            raise ValueError(f"Invalid stage '{self.stage}': expected 'train', 'valid', or 'test'.")

    def _get_user_datasets(self, filter_condition=None):
        """Get per-user datasets with stage-level caching.

        Args:
            filter_condition (callable, optional): Function to filter users.

        Returns:
            dict: Mapping from user ID to dataset.
        """
        # Cache the unfiltered stage dataset and apply caller filters after retrieval.
        cache_key = f"stage_{self.stage}"

        # Check cache
        if cache_key in self._user_datasets_cache:
            cached_datasets = self._user_datasets_cache[cache_key]
            if filter_condition:
                return {uid: ds for uid, ds in cached_datasets.items() if filter_condition(uid)}
            return cached_datasets

        user_datasets = {}

        # Single groupby pass instead of per-user DataFrame filtering (O(N) vs O(N*U))
        grouped = self.dataset.df.groupby(self.dataset.uid_field)
        for user_id, user_df in grouped:
            user_dataset = RecDataset.from_dataframe(
                self.config, user_df,
                item_num=self.dataset.item_num,
                user_num=1,
            )
            user_datasets[user_id] = user_dataset

        # P2 improvement: cache results
        self._user_datasets_cache[cache_key] = user_datasets

        # Apply filter condition if provided
        if filter_condition:
            return {uid: ds for uid, ds in user_datasets.items() if filter_condition(uid)}

        return user_datasets

    def _get_train_loader(self):
        """Get training data loaders - P2 improvement: adds cache support.

        Returns:
            dict: Mapping from user ID to training data loader.
        """
        # P2 improvement: Check data loader cache
        cache_key = "train_loader"
        if cache_key in self._user_loaders_cache:
            return self._user_loaders_cache[cache_key]

        user_datasets = self._get_user_datasets()
        user_loader = {}

        for user_id, user_dataset in user_datasets.items():
            loader = TrainDataLoader(
                self.config,
                user_dataset,
                batch_size=self.config["train_batch_size"],
                shuffle=self.shuffle,
            )
            # Each client sees one user's positives; negatives still come from the global item universe.
            loader.all_items = list(range(self.dataset.item_num))
            loader.all_items_set = set(loader.all_items)
            loader.all_item_len = len(loader.all_items)
            loader._user_neg_cache = {}
            user_loader[user_id] = loader

        # P2 improvement: cache data loaders
        self._user_loaders_cache[cache_key] = user_loader
        return user_loader

    def _get_eval_loader(self):
        """Get evaluation data loaders - P2 improvement: adds cache support.

        Returns:
            dict: Mapping from user ID to evaluation data loader.
        """
        assert (
            self.additional_dataset is not None
        ), "additional_dataset should not be None in eval dataloader"

        cache_key = "eval_loader"
        if cache_key in self._user_loaders_cache:
            return self._user_loaders_cache[cache_key]

        # Precompute user membership for O(1) filtering.
        valid_users = set(
            self.additional_dataset.df[self.additional_dataset.uid_field].unique()
        )

        def filter_condition(user_id):
            return user_id in valid_users

        user_datasets = self._get_user_datasets(filter_condition)
        user_loader = {}

        # Pre-group additional dataset for efficient per-user slicing
        additional_grouped = self.additional_dataset.df.groupby(
            self.additional_dataset.uid_field
        )

        for user_id, user_dataset in user_datasets.items():
            user_additional_df = additional_grouped.get_group(user_id)

            # Build the per-user eval dataset directly from the grouped dataframe.
            user_additional_dataset = RecDataset.from_dataframe(
                self.config, user_additional_df,
                item_num=self.additional_dataset.item_num,
                user_num=1,
            )

            user_loader[user_id] = EvalDataLoader(
                self.config,
                user_dataset,
                batch_size=self.config["eval_batch_size"],
                additional_dataset=user_additional_dataset,
            )

        # P2 improvement: cache evaluation loaders
        self._user_loaders_cache[cache_key] = user_loader
        return user_loader

    def __iter__(self):
        """Iterator method.

        Returns:
            self: Returns self.
        """
        self.user_idx = 0
        if self.shuffle:
            import random

            # Reset to the canonical construction order before shuffling so the
            # permutation is a pure function of the seed, not of prior iteration
            # history (the HPO manager reuses one cached loader across trials).
            self.user_ids = list(self.data_loader.keys())
            random.shuffle(self.user_ids)
        return self

    def __next__(self):
        """Get the data loader for the next user.

        Returns:
            tuple: (user_id, data_loader)

        Raises:
            StopIteration: When all users have been iterated.
        """
        if self.user_idx >= len(self.user_ids):
            raise StopIteration

        user_id = self.user_ids[self.user_idx]
        loader = self.data_loader[user_id]
        self.user_idx += 1

        return user_id, loader

    def __len__(self):
        """Get the number of users.

        Returns:
            int: Number of users.
        """
        return len(self.user_ids)

    def pretrain_setup(self):
        """Pre-training setup, resets random state."""
        # Reset iterator
        self.user_idx = 0
        # Re-shuffle user IDs if shuffling is enabled
        if self.shuffle:
            import random

            # Reset to the canonical construction order before shuffling so a
            # re-seeded shuffle is reproducible across cached-loader HPO trials
            # (pretrain_setup runs before resume's set_resume_state, which then
            # overrides user_ids, so federated resume stays bit-faithful).
            self.user_ids = list(self.data_loader.keys())
            random.shuffle(self.user_ids)

        # Apply setup to each per-user data loader as well
        for user_id, loader in self.data_loader.items():
            if hasattr(loader, "pretrain_setup"):
                loader.pretrain_setup()

    @property
    def loaders(self):
        """Get all data loaders.

        Returns:
            dict: Mapping from user ID to data loader.
        """
        return self.data_loader

    def get_resume_state(self):
        """Capture per-client iteration order for bit-faithful federated resume.

        Each per-user TrainDataLoader shuffles its local df cumulatively; the
        global RNG alone does not pin a mid-run round's per-client batch order
        (pretrain_setup resets every client loader to its backup), so each
        client's working df must be captured/restored. The user-iteration order
        is captured too for completeness.
        """
        return {
            "user_ids": list(self.user_ids),
            "client_loaders": {
                user_id: loader.get_resume_state()
                for user_id, loader in self.data_loader.items()
            },
        }

    def set_resume_state(self, state):
        """Restore per-client iteration order captured at the round boundary."""
        self.user_ids = list(state["user_ids"])
        for user_id, loader_state in state["client_loaders"].items():
            self.data_loader[user_id].set_resume_state(loader_state)

    @property
    def user_set(self):
        """Get the set of users.

        Returns:
            list: List of user IDs.
        """
        return self.user_ids

    @property
    def item_num(self):
        """Total item count, forwarded from underlying dataset for evaluator compatibility."""
        return self.dataset.item_num

    # Evaluation interface methods
    def get_eval_items(self):
        """Get the list of evaluation items (cached after first call)."""
        if self.stage not in ["valid", "test", "eval"]:
            return []
        if self._eval_items is None:
            grouped = self.dataset.df.groupby(self.dataset.uid_field)[self.dataset.iid_field]
            self._eval_items = [
                grouped.get_group(uid).values.tolist() if uid in grouped.groups else []
                for uid in self.user_ids
            ]
        return self._eval_items

    def get_eval_len_list(self):
        """Get the list of evaluation item counts per user (cached after first call)."""
        if self.stage not in ["valid", "test", "eval"]:
            return []
        if self._eval_len_list is None:
            counts = self.dataset.df.groupby(self.dataset.uid_field).size()
            self._eval_len_list = counts.reindex(self.user_ids, fill_value=0).astype(int).tolist()
        return self._eval_len_list

    def get_eval_users(self):
        """Get the list of evaluation users.

        Returns:
            list: List of user IDs.
        """
        import torch
        return torch.tensor(self.user_ids).long()
