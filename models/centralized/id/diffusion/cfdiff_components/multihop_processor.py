"""
Multi-hop data processor for CFDiff model
Handles construction and caching of multi-hop neighbor information
"""

import torch
import numpy as np
from scipy import sparse
import logging
from typing import List, Optional


class MultiHopProcessor:
    """
    Multi-hop neighbor processor for CFDiff

    Builds and caches multi-hop adjacency matrices for efficient
    multi-hop neighbor feature extraction during training/inference.
    """

    def __init__(
        self,
        dataset,
        max_hops: int = 3,
        cache_enabled: bool = True,
    ):
        """
        Initialize multi-hop processor

        Args:
            dataset: RecDataset instance
            max_hops: maximum number of hops to compute
            cache_enabled: whether the multi-hop adjacency matrices stay resident
                           in memory between lookups; with False they are rebuilt
                           per lookup. Residency only — the produced features are
                           identical either way.
        """
        self.dataset = dataset
        self.max_hops = max_hops
        self.cache_enabled = cache_enabled

        # Get dataset dimensions
        self.n_users = dataset.get_user_num()
        self.n_items = dataset.get_item_num()

        # Cache for adjacency matrices
        self.adjacency_cache = {}
        self.user_interactions_cache = {}

        # Cache for normalization factors and link counts (as per paper)
        self.normalization_factors = {}  # N_{h-1,h} for each hop
        self.link_counts = {}  # c^(h): incoming link counts for each hop

        # Build from the true user-item interaction matrix first.
        # Prebuilt adjacency artifacts in datasets/ may encode item-item graphs
        # that are useful for other models but do not replace CFDiff's
        # user-item interaction source of truth.
        self._build_interaction_matrix()

        # Pre-compute multi-hop adjacencies eagerly only when they stay resident;
        # otherwise they are rebuilt on demand in get_multihop_features().
        if cache_enabled:
            self._precompute_multihop_adjacencies()

        logging.info(
            f"MultiHopProcessor initialized: {self.n_users} users, {self.n_items} items, max_hops={max_hops}"
        )
        logging.info(f"Cached adjacencies: {list(self.adjacency_cache.keys())}")

    def _build_interaction_matrix(self):
        """Build user-item interaction matrix from dataset.

        Raises if no interaction data is available — the model cannot function
        without a real interaction graph.
        """
        if (
            hasattr(self.dataset, "df")
            and hasattr(self.dataset, "splitting_label")
            and self.dataset.splitting_label in getattr(self.dataset.df, "columns", [])
        ):
            train_data = self.dataset.df[
                self.dataset.df[self.dataset.splitting_label] == 0
            ]
            users = train_data[self.dataset.uid_field].values
            items = train_data[self.dataset.iid_field].values
            logging.info(
                f"Building interaction matrix from {len(users)} training interactions"
            )
        elif hasattr(self.dataset, "df"):
            users = self.dataset.df[self.dataset.uid_field].values
            items = self.dataset.df[self.dataset.iid_field].values
            logging.info(
                f"Building interaction matrix from {len(users)} total interactions"
            )
        else:
            raise RuntimeError(
                "MultiHopProcessor requires a dataset with interaction data (dataset.df). "
                "Cannot build interaction matrix."
            )

        data = np.ones(len(users))
        self.interaction_matrix = sparse.coo_matrix(
            (data, (users, items)), shape=(self.n_users, self.n_items)
        ).tocsr()

        self._cache_user_interactions()

        logging.info(
            f"Built interaction matrix: {self.interaction_matrix.shape}, nnz={self.interaction_matrix.nnz}"
        )

    def _cache_user_interactions(self):
        """Cache user interactions for fast lookup"""
        for user_id in range(self.n_users):
            # Get items interacted by this user
            if user_id < self.interaction_matrix.shape[0]:
                interactions = self.interaction_matrix[user_id].indices
                self.user_interactions_cache[user_id] = interactions.tolist()
            else:
                self.user_interactions_cache[user_id] = []

    def _precompute_multihop_adjacencies(self):
        """
        Pre-compute multi-hop adjacency matrices with normalization factors.

        Following CF-Diff paper formula:
        u^(h) = (1 / N_{h-1,h}) × r(G(u,h), c^(h))

        Computes:
        - Adjacency matrices for each hop
        - Normalization factors N_{h-1,h}
        - Link counts c^(h) (incoming links from previous hop)
        """
        # Convert to binary matrix for adjacency computation
        binary_matrix = (self.interaction_matrix > 0).astype(float)

        # Store 1-hop (direct interactions)
        self.adjacency_cache[1] = binary_matrix

        # Compute normalization factor for 1-hop: total interactions
        self.normalization_factors[1] = binary_matrix.sum()

        # Link counts for 1-hop: number of users interacting with each item
        self.link_counts[1] = np.array(binary_matrix.sum(axis=0)).flatten()

        if self.max_hops >= 2:
            # 2-hop: user -> item -> user (collaborative neighbors)
            user_user_adj = binary_matrix.dot(binary_matrix.T)
            user_user_adj.setdiag(0)  # Remove self-loops
            self.adjacency_cache[2] = user_user_adj

            # Normalization factor N_{1,2}: edges from 1-hop to 2-hop
            # This is the total number of user-user connections through items
            self.normalization_factors[2] = user_user_adj.sum() / 2  # Divide by 2 for symmetric matrix

            # Link counts c^(2): for each user, how many 1-hop neighbors connect to them
            # This is the row sum (number of similar users)
            self.link_counts[2] = np.array(user_user_adj.sum(axis=1)).flatten()

        if self.max_hops >= 3:
            # 3-hop: user -> user -> item (items of similar users)
            three_hop_adj = self.adjacency_cache[2].dot(binary_matrix)

            # Normalize by considering the path weights
            # Keep as weighted matrix for better representation
            self.adjacency_cache[3] = three_hop_adj

            # Normalization factor N_{2,3}: edges from 2-hop to 3-hop
            self.normalization_factors[3] = three_hop_adj.sum()

            # Link counts c^(3): for each item, how many 2-hop paths lead to it
            self.link_counts[3] = np.array(three_hop_adj.sum(axis=0)).flatten()

        logging.info(f"Pre-computed adjacency matrices with normalization for {self.max_hops} hops")
        for hop, adj in self.adjacency_cache.items():
            norm_factor = self.normalization_factors.get(hop, 1.0)
            logging.info(
                f"  Hop {hop}: shape={adj.shape}, nnz={adj.nnz}, "
                f"N_{{{hop-1},{hop}}}={norm_factor:.2f}"
            )

    def get_user_interactions(self, user_id: int) -> List[int]:
        """
        Get items that a user has interacted with

        Args:
            user_id: user index

        Returns:
            List of item indices the user has interacted with
        """
        return self.user_interactions_cache.get(user_id, [])

    def get_multihop_features(
        self, users: torch.Tensor, items: torch.Tensor, device: str = None
    ) -> Optional[torch.Tensor]:
        """
        Get multi-hop neighbor features for given user-item pairs (optimized with caching)

        Args:
            users: user indices [batch_size]
            items: item indices [batch_size]
            device: target device for tensors

        Returns:
            Multi-hop features tensor [batch_size, feature_dim] or None
        """
        if self.max_hops <= 1:
            return None

        if device is None:
            device = users.device

        # Adjacencies are required for correctness whenever max_hops > 1; rebuild
        # them on demand if they are not resident (cache_enabled=False or first use).
        if not self.adjacency_cache:
            self._precompute_multihop_adjacencies()

        features_tensor = self._get_batch_multihop_features(users, items, device)

        # cache_enabled controls residency only: drop the tables after the lookup
        # so the next call recomputes them — the features are identical either way.
        if not self.cache_enabled:
            self.adjacency_cache = {}
            self.normalization_factors = {}
            self.link_counts = {}

        return features_tensor

    def _get_batch_multihop_features(
        self, users: torch.Tensor, items: torch.Tensor, device: str
    ) -> torch.Tensor:
        """
        Batch process multi-hop features for better performance.

        Optimization: Process multiple users together using vectorized operations.

        Args:
            users: user indices [batch_size]
            items: item indices [batch_size]
            device: target device for tensors

        Returns:
            Multi-hop features tensor [batch_size, feature_dim]
        """
        batch_size = users.size(0)
        feature_dim = self.n_items * (self.max_hops - 1)

        # Pre-allocate output tensor
        all_features = torch.zeros(batch_size, feature_dim, dtype=torch.float32)

        # Convert to numpy for efficient processing
        user_ids = users.cpu().numpy()

        # Process each hop level in batches
        for hop in range(2, self.max_hops + 1):
            if hop not in self.adjacency_cache:
                continue

            hop_idx = hop - 2  # Index in concatenated features
            start_idx = hop_idx * self.n_items
            end_idx = (hop_idx + 1) * self.n_items

            adj_matrix = self.adjacency_cache[hop]

            # Batch extract features for all users at this hop
            for i, user_id in enumerate(user_ids):
                if user_id < adj_matrix.shape[0]:
                    # Extract hop neighbors
                    hop_neighbors = adj_matrix[user_id]

                    # Convert to dense array
                    if sparse.issparse(hop_neighbors):
                        hop_neighbors = hop_neighbors.toarray().flatten()
                    else:
                        hop_neighbors = np.asarray(hop_neighbors).flatten()

                    # Handle dimension conversion for hop 2 (user-user to user-item)
                    if hop == 2 and hop_neighbors.shape[0] == self.n_users:
                        if hasattr(self, 'interaction_matrix'):
                            user_similarities_sparse = sparse.csr_matrix(hop_neighbors.reshape(1, -1))
                            user_item_features = user_similarities_sparse.dot(
                                self.interaction_matrix
                            ).toarray().flatten()
                            hop_neighbors = user_item_features[:self.n_items]
                        else:
                            hop_neighbors = np.zeros(self.n_items)

                    # Resize if needed
                    if hop_neighbors.shape[0] != self.n_items:
                        if hop_neighbors.shape[0] > self.n_items:
                            hop_neighbors = hop_neighbors[:self.n_items]
                        else:
                            padded = np.zeros(self.n_items)
                            padded[:hop_neighbors.shape[0]] = hop_neighbors
                            hop_neighbors = padded

                    # Apply normalization (as per paper formula)
                    if hop in self.link_counts:
                        link_counts_h = self.link_counts[hop]
                        if len(link_counts_h) == len(hop_neighbors):
                            hop_neighbors = hop_neighbors * (1.0 + np.log1p(link_counts_h))

                    if hop in self.normalization_factors:
                        norm_factor = self.normalization_factors[hop]
                        if norm_factor > 0:
                            hop_neighbors = hop_neighbors / norm_factor

                    # Store in output tensor
                    all_features[i, start_idx:end_idx] = torch.FloatTensor(hop_neighbors)

        return all_features.to(device)
