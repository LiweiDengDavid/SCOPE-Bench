# coding: utf-8
"""
Graph Utility Functions
=======================

Shared graph construction and Laplacian helpers used by graph and multimodal
recommendation models.
"""

from __future__ import annotations

import torch


def _scatter_add(src, index, dim_size, dim=0):
    """Accumulate src into a zero tensor of shape (dim_size,) along dim."""
    result = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    result.index_add_(dim, index, src)
    return result


def build_knn_neighbourhood(adj, topk):
    """Build a top-k weighted neighbourhood adjacency matrix."""
    topk = min(topk, adj.shape[1])
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    return torch.zeros_like(adj).scatter_(-1, knn_ind, knn_val)


def compute_normalized_laplacian(adj):
    """Compute the symmetric normalized Laplacian for a dense adjacency."""
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


def build_sim(context):
    """Build a cosine-similarity matrix from a feature matrix."""
    import torch.nn.functional as F

    context_norm = F.normalize(context, p=2, dim=-1)
    return torch.mm(context_norm, context_norm.transpose(1, 0))


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization="none"):
    """Compute a sparse Laplacian representation."""
    row, col = edge_index[0], edge_index[1]
    deg = _scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == "sym":
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == "rw":
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization="none"):
    """Compute a dense Laplacian representation."""
    if normalization == "sym":
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    if normalization == "rw":
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.0
        d_mat_inv = torch.diagflat(d_inv)
        return torch.mm(d_mat_inv, adj)
    return adj


def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    """Construct a top-k normalized graph in dense or sparse form."""
    device = adj.device
    topk = min(topk, adj.shape[1])
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)

    if is_sparse:
        rows, cols = [], []
        for row_idx, cols_idx in enumerate(knn_ind):
            rows.extend([row_idx] * len(cols_idx))
            cols.extend(cols_idx.tolist())

        edge_index = torch.tensor([rows, cols], dtype=torch.long).to(device)
        edge_weight = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(
            edge_index, edge_weight, num_nodes=adj.shape[0], normalization=norm_type
        )
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)

    weighted_adjacency_matrix = torch.zeros_like(adj).scatter_(-1, knn_ind, knn_val)
    return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)



def build_norm_adj_matrix(interaction_matrix, n_users, n_items, device=None):
    """Build symmetric normalized adjacency matrix for user-item bipartite graph.

    Computes D^{-0.5} * A * D^{-0.5} where A is the bipartite adjacency matrix.
    Used by LightGCN, SGL, FREEDOM, BM3, DiffMM and other GNN models.

    Args:
        interaction_matrix: scipy COO matrix of user-item interactions.
        n_users: number of users.
        n_items: number of items.
        device: target torch device (optional).

    Returns:
        torch.sparse_coo_tensor on *device* (or CPU if device is None).
    """
    import numpy as np
    import scipy.sparse as sp

    inter_M = interaction_matrix
    inter_M_t = inter_M.transpose()

    row = np.concatenate([inter_M.row, inter_M_t.row + n_users])
    col = np.concatenate([inter_M.col + n_users, inter_M_t.col])
    data = np.ones(len(row), dtype=np.float32)

    n_nodes = n_users + n_items
    A = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)

    # Symmetric Laplacian normalization
    diag = np.array(A.sum(axis=1)).flatten() + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D @ A @ D

    L = sp.coo_matrix(L)
    indices = torch.LongTensor(np.array([L.row, L.col]))
    values = torch.FloatTensor(L.data)
    result = torch.sparse_coo_tensor(indices, values, torch.Size(L.shape))
    if device is not None:
        result = result.to(device)
    return result
