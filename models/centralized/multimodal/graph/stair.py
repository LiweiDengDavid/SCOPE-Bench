# coding: utf-8
r"""
STAIR
################################################
Reference:
    Chen et al. "STAIR: Manipulating Collaborative and Multimodal Information
    for E-Commerce Recommendation." AAAI'2025.
    Official code: https://github.com/yizhenzhong/STAIR  (``main.py`` -- the model,
    the ``CoachForSTAIR`` coach, and the config all live there;
    ``optimizers/AdamW.py`` = the used ``AdamWSEvo``; ``optimizers/utils.py`` =
    ``Smoother``).

Faithful port of the official ``STAIR`` (freerec) model. STAIR is a fully
deterministic spectral graph recommender with three staircase mechanisms keyed to
the embedding dimension:

  * **Whitened init.** Each modality's raw features are column-centered, SVD'd,
    and rescaled ``U[:, :D] * sqrt(n_items / D)`` (``whitening``). The item
    embeddings are initialized to the ``num_neighbors``-weighted sum of the
    whitened modalities (text 5, visual 1); the user embeddings to
    ``R @ mfeats`` where ``R`` is the per-user-mean-normalized (row-stochastic)
    user->item interaction matrix.

  * **FSC (Forward Staircase Convolution).** ``encode()`` applies an
    ``L``-hop per-embedding-dimension Neumann/geometric LightGCN smoothing over
    the sym-normalized joint user-item adjacency ``Adj``. The per-dimension mix
    ``beta[d] = 1 - beta3[d]`` with ``beta3[d] = 0.1 + 0.9 * (d/D)^gamma`` makes
    smoothing a *staircase across embedding dimensions* (leading dims heavily
    smoothed, trailing dims near-raw). This is NOT epoch/layer staging: there is
    a single ordinary training loop and no epoch schedule.

  * **BSC (Backward Staircase Convolution).** Lives inside the custom optimizer
    ``AdamWSEvo``: the item param group carries a ``Smoother(mAdj, beta=beta3,
    L, aggr='neumann')``; each step does decoupled weight decay, a bias-corrected
    Adam delta, then Neumann-smooths that delta over the multimodal item kNN
    graph ``mAdj`` with the per-dimension ``beta3`` (trailing dims strongly
    smoothed -- the complementary staircase) before ``param -= lr * delta``. The
    user group is plain AdamW. Because Adam is nonlinear, smoothing the raw
    gradient is NOT equivalent -- the smoothing must happen on the Adam delta, so
    it cannot be a loss/model-side term and genuinely requires a custom
    optimizer. STAIR is the sole model in this repo that supplies its own
    optimizer, via the additive ``build_optimizer`` factory hook.

The objective is a single ``BPR`` term on einsum scores with one negative.

freerec / PyG operations are mapped to plain torch (all documented inline):
``to_normalized_adj('sym')`` -> LightGCN sym-normalized bipartite adjacency from
``inter_matrix('coo')``; ``get_knn_graph`` -> ``torch.topk`` on L2-normalized
features (self excluded); ``coalesce(reduce='sum')`` -> unique linearized edge
ids + ``scatter_reduce('sum')``; ``to_undirected(reduce='max')`` -> symmetrize
via unique linearized ids + ``scatter_reduce('amax')``; ``to_normalized('sym')``
-> D^-1/2 A D^-1/2; ``to_normalized('left')`` -> row-degree (D^-1) scaling.

All literals (``embedding_dim``, ``num_layers=3``, ``gamma``, ``num_neighbors
'5-1'``, ``weight_decay``) live in ``configs/models/STAIR.yaml``. The whitening
SVD and ``mAdj`` are cached to the dataset directory (both derive only from the
raw features / interactions).
"""

import math
import os
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer

from core.base import RecommenderBase
from core.utils import build_norm_adj_matrix


# ======================================================================
# Smoother -- faithful copy of the official ``optimizers/utils.py``.
# Only the 'neumann' aggregation is used by STAIR; 'momentum'/'average' are
# retained verbatim for fidelity. Plain torch, ships inside stair.py.
# ======================================================================
class Smoother:
    r"""Per-dimension graph smoother over a fixed sparse adjacency.

    ``__call__(features)`` returns the Neumann/geometric smoothing
    ``sum_{l=0}^{L} (beta * A)^l features * (1 - beta) / (1 - beta^(L+1))`` where
    ``beta`` is a per-embedding-dimension vector (broadcast over rows). This is
    the same closed form as the FSC in ``STAIR.encode``, applied here to the
    Adam delta (BSC).
    """

    def __init__(self, A: torch.Tensor, beta, L: int, aggr: str) -> None:
        self.Adj = A
        self.beta = beta
        self.L = L
        self.aggr = aggr

    def aggregate(self, features: torch.Tensor):
        return self.Adj @ features

    @torch.no_grad()
    def __call__(self, features: torch.Tensor):
        smoothed = features
        if self.aggr == "neumann":
            norm_correction = 1 - self.beta ** (self.L + 1)
            for _ in range(self.L):
                features = self.aggregate(features) * self.beta
                smoothed = smoothed + features
            smoothed = smoothed.mul(1 - self.beta).div(norm_correction)
        elif self.aggr == "momentum":
            for _ in range(self.L):
                smoothed = self.aggregate(smoothed) * self.beta + features * (1 - self.beta)
        elif self.aggr == "average":
            smoothed = features / (self.L + 1)
            for _ in range(self.L):
                features = self.aggregate(features)
                smoothed += features / (self.L + 1)
        else:
            raise ValueError(
                f"aggr should be average|neumann|momentum but {self.aggr} received ..."
            )
        return smoothed


# ======================================================================
# AdamWSEvo -- faithful copy of the official ``optimizers/AdamW.py``.
# Item group: decoupled wd -> bias-corrected Adam delta -> Smoother(delta) ->
# param -= lr*delta (BSC). Groups without a smoother fall back to standard AdamW.
# Plain torch, ships inside stair.py.
# ======================================================================
class AdamWSEvo(Optimizer):
    r"""AdamW with a per-group **S**\ moothed-**Evo**\ lution update.

    A param group carrying a callable ``smoother`` gets the STAIR BSC update
    (``update_embeddings``): the bias-corrected Adam delta is graph-smoothed
    before it is subtracted from the parameter. Groups without a smoother use the
    stock AdamW functional -- byte-for-byte the reference PyTorch AdamW math, so
    the user embeddings behave exactly as under ``torch.optim.AdamW``.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=False, maximize=False,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def update_embeddings(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        maximize: bool,
        smoother: Callable,
    ):
        r"""BSC update: decoupled wd, bias-corrected Adam delta, then the delta
        (concatenated across all params in the group) is Neumann-smoothed over
        ``mAdj`` with per-dim ``beta3`` before ``param -= lr * delta``."""
        deltas = []

        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]

            step_t += 1

            # Decoupled (AdamW) weight decay.
            param.mul_(1 - lr * weight_decay)

            mgrad = grad
            vgrad = grad.pow(2)

            exp_avg.mul_(beta1).add_(mgrad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).add_(vgrad, alpha=1 - beta2)

            step = step_t.item()
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            bias_correction2_sqrt = math.sqrt(bias_correction2)

            numer = exp_avg.div(bias_correction1)
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            deltas.append(numer.div(denom))

        counts = [delta.size(0) for delta in deltas]
        deltas = smoother(torch.cat(deltas, dim=0))
        deltas = torch.split(deltas, counts)

        for i, param in enumerate(params):
            delta = deltas[i] * lr
            param.add_(delta.neg())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            smoother = group.get("smoother", None)

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("AdamWSEvo does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

            if smoother is not None:
                self.update_embeddings(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    state_steps,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                    smoother=smoother,
                )
            else:
                _adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    state_steps,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                )
        return loss


def _adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Stock single-tensor AdamW step (reference PyTorch math) for groups without
    a smoother -- the STAIR user group. Kept identical to
    ``torch.optim.AdamW`` so non-smoothed params behave exactly as standard."""
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        step_t += 1
        param.mul_(1 - lr * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step = step_t.item()
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)


class STAIR(RecommenderBase):
    r"""STAIR multimodal graph recommender (AAAI'25).

    Single-negative BPR (``supports_multi_negatives = False``). Supplies its own
    optimizer through ``build_optimizer`` (the additive factory hook) so the BSC
    Adam-delta smoothing runs in the optimizer step.
    """

    supports_multi_negatives = False

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.embedding_dim = config["embedding_size"]
        self.num_layers = config["num_layers"]
        self.gamma = float(config["gamma"])
        # ``num_neighbors`` is the per-modality kNN degree AND the init blend
        # weight, in feature-file order (text, visual) == (self.t_feat,
        # self.v_feat). Official default '5-1' -> text 5, visual 1.
        self.num_neighbors = [int(x) for x in str(config["num_neighbors"]).split("-")]

        # ``beta3[d] = 0.1 + 0.9 * (d/D)^gamma`` -- the per-embedding-dimension
        # smoothing profile. ``beta3`` is the (1 - beta_j) used by BSC directly
        # and by FSC as ``beta = 1 - beta3``. Built here from config["gamma"].
        self.beta3 = (
            0.1
            + 0.9
            * (torch.arange(self.embedding_dim, dtype=torch.float) / self.embedding_dim).pow(
                self.gamma
            )
        ).to(self.device)

        # Bipartite train interaction matrix (COO), used for Adj, mAdj-free R,
        # and the user init.
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)

        # ``to_normalized_adj('sym')`` -> LightGCN sym-normalized joint UI adjacency.
        self.register_buffer(
            "Adj",
            build_norm_adj_matrix(
                self.interaction_matrix, self.n_users, self.n_items, self.device
            ),
        )

        # Collaborative embeddings (overwritten by the whitened init in prepare()).
        self.user_embedding = torch.nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.embedding_dim)
        torch.nn.init.normal_(self.user_embedding.weight, std=1e-4)
        torch.nn.init.normal_(self.item_embedding.weight, std=1e-4)

        # Whitened init + multimodal item kNN graph ``mAdj`` (BSC operand). Both
        # cached to the dataset directory.
        self.prepare(config)

        from core.base.loss import BPRLoss

        self.criterion = BPRLoss()

    # ------------------------------------------------------------------
    # Init: whitening, mAdj, whitened item/user embeddings (with caching)
    # ------------------------------------------------------------------
    def whitening(self, feats: torch.Tensor) -> torch.Tensor:
        r"""Column-center -> SVD -> ``U[:, :D] * sqrt(n_items / D)``.

        Faithful to the official ``whitening``. ``full_matrices=False`` makes
        ``U`` shape ``[n_items, min(n_items, F)]``; ``embedding_dim`` must not
        exceed that rank (guaranteed at dataset scale: F >> D). No defensive
        clamp -- a misconfiguration should fail fast.
        """
        feats = feats - feats.mean(0, keepdim=True)
        u, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return u[:, : self.embedding_dim] * math.sqrt(self.n_items / self.embedding_dim)

    def _knn_edges(self, features: torch.Tensor, k: int):
        r"""``get_knn_graph(symmetric=False)``: for each row, the top-``k`` columns
        by cosine similarity (self excluded), returned as DIRECTED edges
        ``[2, N*k]`` (row 0 = source i, row 1 = destination j)."""
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.0)
        _, knn_ind = torch.topk(sim, k, dim=-1)  # (N, k)
        n = features.shape[0]
        src = torch.arange(n, device=features.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = knn_ind.reshape(-1)
        return torch.stack([src, dst], dim=0)

    def _build_m_adj(self, mfeats: List[torch.Tensor]) -> torch.Tensor:
        r"""Multimodal item kNN graph ``mAdj`` (the BSC operand).

        Faithful pipeline: per-modality directed kNN (weight 1) with degree
        ``num_neighbors`` -> ``coalesce(sum)`` (merge duplicate directed edges) ->
        ``to_undirected(max)`` (symmetrize, keep max of the two directions) ->
        ``to_normalized('sym')`` (D^-1/2 A D^-1/2). Returned as a sparse CSR
        tensor ``[n_items, n_items]``.
        """
        n = self.n_items
        edge_index = torch.cat(
            [self._knn_edges(feats, k) for feats, k in zip(mfeats, self.num_neighbors)],
            dim=1,
        )
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float, device=edge_index.device)

        # coalesce(reduce='sum'): merge duplicate directed (i, j) edges.
        edge_index, edge_weight = self._coalesce(edge_index, edge_weight, n, reduce="sum")
        # to_undirected(reduce='max'): symmetrize, per undirected pair keep the max
        # weight across the two directions.
        edge_index, edge_weight = self._to_undirected_max(edge_index, edge_weight, n)
        # to_normalized('sym'): D^-1/2 A D^-1/2.
        edge_index, edge_weight = self._to_normalized_sym(edge_index, edge_weight, n)

        m_adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(n, n)
        ).coalesce()
        return m_adj.to_sparse_csr()

    @staticmethod
    def _coalesce(edge_index, edge_weight, n, reduce):
        r"""Merge duplicate edges by linearized id, reducing weights with
        ``scatter_reduce`` (``sum``/``amax``). Returns sorted unique edges."""
        keys = edge_index[0] * n + edge_index[1]
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        out = torch.zeros(unique_keys.shape[0], dtype=edge_weight.dtype, device=edge_weight.device)
        out.scatter_reduce_(0, inverse, edge_weight, reduce=reduce, include_self=False)
        rows = unique_keys // n
        cols = unique_keys % n
        return torch.stack([rows, cols], dim=0), out

    def _to_undirected_max(self, edge_index, edge_weight, n):
        r"""Symmetrize: add the reversed edges, then coalesce with ``amax`` so each
        undirected pair keeps the maximum of its two directional weights."""
        rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        both_index = torch.cat([edge_index, rev], dim=1)
        both_weight = torch.cat([edge_weight, edge_weight], dim=0)
        return self._coalesce(both_index, both_weight, n, reduce="amax")

    @staticmethod
    def _to_normalized_sym(edge_index, edge_weight, n):
        r"""D^-1/2 A D^-1/2 over the (already symmetric) edge set."""
        deg = torch.zeros(n, dtype=edge_weight.dtype, device=edge_weight.device)
        deg.scatter_add_(0, edge_index[0], edge_weight)
        d_inv_sqrt = torch.pow(deg, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        norm_weight = d_inv_sqrt[edge_index[0]] * edge_weight * d_inv_sqrt[edge_index[1]]
        return edge_index, norm_weight

    def _row_normalized_R(self) -> torch.Tensor:
        r"""``to_normalized('left')`` of the u2i graph: the per-user-mean
        (row-stochastic D^-1) user->item matrix ``R``, as a sparse CSR tensor
        ``[n_users, n_items]``. Used to seed the user embeddings from item feats.
        """
        coo = self.interaction_matrix.tocoo()
        rows = torch.from_numpy(coo.row.astype(np.int64))
        cols = torch.from_numpy(coo.col.astype(np.int64))
        vals = torch.ones(rows.shape[0], dtype=torch.float)
        deg = torch.zeros(self.n_users, dtype=torch.float)
        deg.scatter_add_(0, rows, vals)
        d_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        norm_vals = d_inv[rows]
        R = torch.sparse_coo_tensor(
            torch.stack([rows, cols], dim=0),
            norm_vals,
            size=(self.n_users, self.n_items),
        ).coalesce()
        return R.to_sparse_csr().to(self.device)

    def prepare(self, config):
        r"""Whitened init of item/user embeddings + build (or load) ``mAdj``.

        Order matches ``self.mfiles`` in the official config
        (``textual_modality.pkl, visual_modality.pkl``) so ``num_neighbors``
        (text 5, visual 1) lines up with the features: mfeats = [t_feat, v_feat].
        Both the whitened item init and ``mAdj`` are cached to the dataset dir
        (they depend only on the raw features + kNN degrees).
        """
        mfeats = [self.t_feat, self.v_feat]

        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        os.makedirs(dataset_path, exist_ok=True)
        tag = "-".join(str(k) for k in self.num_neighbors)
        madj_file = os.path.join(dataset_path, f"stair_madj_{tag}.pt")
        winit_file = os.path.join(
            dataset_path, f"stair_winit_{self.embedding_dim}_{tag}.pt"
        )

        # --- multimodal item kNN graph mAdj (BSC operand) ---
        if os.path.exists(madj_file):
            m_adj = torch.load(madj_file, map_location="cpu", weights_only=False)
        else:
            m_adj = self._build_m_adj([f.detach().cpu() for f in mfeats])
            torch.save(m_adj, madj_file)
        self.register_buffer("m_adj", m_adj.to(self.device))

        # --- whitened item init: sum_m whitening(m) * k_m / sum(k) ---
        if os.path.exists(winit_file):
            item_init = torch.load(winit_file, map_location="cpu", weights_only=False)
        else:
            whitened = [
                self.whitening(f.detach().cpu()) * k
                for f, k in zip(mfeats, self.num_neighbors)
            ]
            item_init = sum(whitened).div(sum(self.num_neighbors))
            torch.save(item_init, winit_file)
        item_init = item_init.to(self.device)
        self.item_embedding.weight.data.copy_(item_init)

        # --- whitened user init: R @ item_init (R row-stochastic u2i) ---
        R = self._row_normalized_R()
        self.user_embedding.weight.data.copy_(R @ item_init)

    # ------------------------------------------------------------------
    # FSC + forward / loss / eval
    # ------------------------------------------------------------------
    def encode(self):
        r"""FSC: per-embedding-dimension Neumann/geometric LightGCN smoothing over
        the sym-normalized joint UI adjacency ``Adj``.

        ``beta = 1 - beta3``; ``L`` hops accumulate ``Adj @ features * beta``,
        then normalize by ``(1 - beta) / (1 - beta^(L+1))``. Returns
        ``(userEmbds, itemEmbds)``.
        """
        all_embds = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )
        features = all_embds
        smoothed = all_embds

        beta = 1 - self.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = torch.sparse.mm(self.Adj, features) * beta
            smoothed = smoothed + features
        avg_embds = smoothed.mul(1 - beta).div(norm_correction)
        user_embds, item_embds = torch.split(
            avg_embds, (self.n_users, self.n_items)
        )
        return user_embds, item_embds

    def forward(self):
        return self.encode()

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        if len(interaction) >= 3:
            neg_items = interaction[2]
        else:
            neg_items = torch.randint(
                0, self.n_items, pos_items.shape, device=pos_items.device
            )

        user_embds, item_embds = self.encode()
        u = user_embds[users]           # (B, D)
        i_pos = item_embds[pos_items]   # (B, D)
        i_neg = item_embds[neg_items]   # (B, D)

        pos_scores = torch.sum(u * i_pos, dim=-1)
        neg_scores = torch.sum(u * i_neg, dim=-1)
        return self.criterion(pos_scores, neg_scores)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embds, item_embds = self.encode()
        u = user_embds[user]
        return torch.matmul(u, item_embds.transpose(0, 1))

    # ------------------------------------------------------------------
    # Optimizer: the additive factory hook -> AdamWSEvo (user + item groups)
    # ------------------------------------------------------------------
    def build_optimizer(self, config):
        r"""Return the configured ``AdamWSEvo`` over the STAIR param groups.

        The user group is plain AdamW; the item group carries a
        ``Smoother(m_adj, beta=beta3, L=num_layers, aggr='neumann')`` -- the BSC.
        Built here (not in ``__init__``) so ``m_adj`` and ``beta3`` are already on
        the training device when the Smoother captures them (the trainer builds
        the optimizer after ``model.to(device)``).
        """
        from core.config import coerce_runtime_scalar

        lr = coerce_runtime_scalar(config["learning_rate"])
        weight_decay = coerce_runtime_scalar(config["weight_decay"])

        param_groups = [
            {
                "params": list(self.user_embedding.parameters()),
                "smoother": None,
            },
            {
                "params": list(self.item_embedding.parameters()),
                "smoother": Smoother(
                    self.m_adj, beta=self.beta3, L=self.num_layers, aggr="neumann"
                ),
            },
        ]
        return AdamWSEvo(param_groups, lr=lr, weight_decay=weight_decay)
