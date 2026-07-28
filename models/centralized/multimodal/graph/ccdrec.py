# coding: utf-8
r"""
CCDRec -- Curriculum Conditional Diffusion for Multimodal Recommendation
########################################################################
Reference:
    Paper: "Curriculum Conditional Diffusion Recommendation", AAAI'2025 (oral).
    Official code: CCDRec_code/CCDRec/src/models/ccdrec.py (backbone) +
    src/models/diffusion_ver15.py (the real ``selfAttention`` denoiser).

CCDRec is a STANDALONE model that embeds a FREEDOM multimodal-graph backbone as
an INTERNAL module (mirroring the DA-MRS / BeFA / BGCC standalone-wrapper
precedent -- it does NOT depend on the sibling ``freedom.py``) and augments it
with a conditional diffusion process:

  * **FREEDOM backbone** -- xavier user/item-id embeddings; a mixed image/text
    kNN item-item graph (``mm_adj``; ``knn_k`` neighbours, ``mm_image_weight``
    blend), cached to the dataset dir; per-epoch degree-sensitive UI edge pruning
    (``dropout_rate``) via ``torch.multinomial`` into ``masked_adj``.
  * **Diffusion on item-ID embeddings** -- ``x_start = item_id_embedding[items]``
    with ``items = cat(pos, neg)``; conditioning = the projected text/image
    features (``feat_embed_dim`` must equal ``embedding_size``); antithetic
    timesteps; ``q_sample`` noising; a single-head self-attention denoiser over
    ``[x_noisy, text_feat, image_feat, timestep_emb]`` predicts ``x0`` directly;
    ``diff_loss = mse(x_start, predicted_x)``. (The official UNet and
    ``denoise_model_uncon`` are instantiated-but-dead; they and their
    ``os.environ['CUDA_VISIBLE_DEVICES']`` side effect are NOT ported.)
  * **DMA** (Diffusion-Modulated Aggregation) -- ``h`` = ``mm_adj`` convolutions
    of the id embeddings; ``h_diff[items] = w*predicted_x + (1-w)*h[items]``;
    ``cat(user, h_diff)`` propagated through ``n_ui_layers`` LightGCN-mean; the
    final item representation is ``i_g + h``.
  * **NDI** (Negative-Diffusion Interaction, epoch > 0) -- a second BPR whose
    negatives are sampled from a diffusion-derived candidate table.
  * **CNS** (Curriculum Negative Schedule) -- the source of those negatives
    advances easy->hard with the internal epoch counter: the least-denoised
    snapshot (quarter) -> half -> three-quarter -> the fully-denoised sample ->
    finally the live item embeddings.

  total_loss = (1-ndi_weight)*BPR(u,pos,neg) + ndi_weight*BPR(u,pos,neg_diff)
             + reg_weight*(FREEDOM modal BPRs) + diff_weight*diff_loss

Deviations from the official code (all documented in-code):
  * ``pre_epoch_processing`` is the framework's NO-arg hook; the model counts
    epochs itself (``self._epoch_idx``, incremented AFTER the dropout early
    return so the first counted call is epoch 0 -- matching the official
    ``epoch_idx > 0`` NDI gate AND the official quirk that ``dropout <= 0``
    freezes the counter, disabling NDI/CNS). CNS reads that counter.
  * The official trainer calls ``model.sample()`` (a full T-step stochastic
    reverse chain over ALL items) once per epoch after training, and eval then
    consumes the resulting ``sample_x``. Here that becomes a lazily-refreshed,
    SEEDED draw behind a dirty flag: ``pre_epoch_processing`` marks the sample
    dirty; ``full_sort_predict`` resamples once with a
    ``torch.Generator(eval_sample_seed + refresh index)`` when dirty (so
    ``full_sort_predict`` calls within one refresh cycle are bit-equal --
    deterministic eval) and stores the snapshots into buffers that also feed the
    next epoch's CNS (matching the official ordering). Folding the refresh index
    into the seed decorrelates successive snapshots, mirroring the official
    fresh-per-epoch global-RNG noise (iid across epochs).
  * The dense ``n_users x n_items`` interaction mask used by negative sampling
    is replaced with a per-batch CSR slice (batch rows x candidate columns,
    duplicated candidates included) -- column-position mask semantics identical
    to the official ``interaction_matrix_dense[users][:, random_indices]``.
  * The diffusion-sample buffers start as ZEROS instead of the official
    ``None``: a config where an NDI epoch precedes the first eval-time
    ``sample()`` (e.g. ``eval_step > 1``) crashes officially but degrades
    gracefully here (NDI scores against a zero table until the first refresh).
  * β-schedule tensors are registered as BUFFERS (the official code kept them as
    CPU attrs and did ``.gather(-1, t.cpu())``).
  * Generic YAML ``w`` / ``weight`` are renamed ``ccdrec_blend_w`` /
    ``ccdrec_ndi_weight``; official ``dropout`` -> canonical ``dropout_rate``;
    ``n_mm_layers`` / ``n_ui_layers`` follow the brief. Dead keys
    (``lambda_coeff``, ``weight_size``, ``c``, ``degree_ratio``, ``aug_weight``,
    ``negsample_step``, all UNet keys) are not ported; the literal candidate
    fraction (0.1) is config-ified as ``candidate_fraction``.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase
from core.utils import build_norm_adj_matrix


def _linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def _cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def _exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    return 1 - torch.exp(-beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))


class _CCDRecDiffusion(nn.Module):
    """Conditional diffusion over item-ID embeddings (ref diffusion_ver15.py).

    The denoiser is the paper's ``selfAttention``: LayerNorm + single-head QKV
    self-attention over the four conditioning tokens [x_noisy, text, image,
    timestep], mean-pooled to predict ``x0`` directly. All β-schedule
    derivations are registered as buffers so they move with ``.to(device)`` and
    ``extract`` can gather with the timesteps on any device (the official code
    forced ``.gather(-1, t.cpu())``).
    """

    def __init__(self, timesteps, beta_start, beta_end, beta_sche, embedding_dim):
        super().__init__()
        self.timesteps = timesteps
        self.embedding_dim = embedding_dim

        if beta_sche == "linear":
            betas = _linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_sche == "exp":
            betas = _exp_beta_schedule(timesteps)
        elif beta_sche == "cosine":
            betas = _cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"CCDRec: unknown beta_sche '{beta_sche}' (linear|exp|cosine)")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register every schedule derivation the forward/reverse passes read.
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance",
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        # Multimodal-fusion self-attention denoiser (ref selfAttention).
        self.w_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        for lin in (self.w_q, self.w_k, self.w_v):
            nn.init.xavier_normal_(lin.weight)
        self.ln = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    @staticmethod
    def _extract(a, t, x_shape):
        out = a.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def _timestep_embedding(self, timesteps):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

    def _self_attention(self, features):
        # features: [batch, n_tokens, embedding_dim]
        features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)
        attn = q.mul(self.embedding_dim ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        features = attn @ v
        return features.mean(dim=-2)  # mean-pool the tokens -> [batch, embedding_dim]

    def q_sample(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_t * noise

    def p_losses(self, x_start, text_feat, image_feat, t):
        """Predict x0 from noised id embeddings + modality conditioning.

        ``text_feat`` / ``image_feat`` are the per-item projected conditioning
        rows (the official call order is p_losses(model, id, text, image, t)).
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        t_emb = self._timestep_embedding(t)
        tokens = torch.cat(
            [x_noisy.unsqueeze(1), text_feat.unsqueeze(1), image_feat.unsqueeze(1), t_emb.unsqueeze(1)],
            dim=1,
        )
        predicted_x = self._self_attention(tokens)
        loss = F.mse_loss(x_start, predicted_x)
        return loss, predicted_x

    @torch.no_grad()
    def p_sample(self, x_t, text_feat, image_feat, t, t_index):
        t_emb = self._timestep_embedding(t)
        tokens = torch.cat(
            [x_t.unsqueeze(1), text_feat.unsqueeze(1), image_feat.unsqueeze(1), t_emb.unsqueeze(1)],
            dim=1,
        )
        x_start = self._self_attention(tokens)
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t_index == 0:
            return model_mean
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn(x_t.shape, generator=self._generator, device=x_t.device, dtype=x_t.dtype) \
            if self._generator is not None else torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, x_start, text_feat, image_feat, generator=None):
        """Full T-step reverse chain over all items; snapshot at 25/50/75/100%.

        Returns (predicted_x, quarter, half, three_quarter) where ``quarter`` is
        the LEAST-denoised snapshot (captured earliest in the reverse chain) and
        ``predicted_x`` is fully denoised -- the easy->hard curriculum ordering.
        When ``generator`` is provided every random draw (the initial q_sample
        noise and each posterior step) uses it, so the whole chain is
        deterministic given a fixed seed.
        """
        self._generator = generator
        device = x_start.device
        if generator is not None:
            noise = torch.randn(x_start.shape, generator=generator, device=device, dtype=x_start.dtype)
        else:
            noise = torch.randn_like(x_start)

        t_full = torch.full((x_start.shape[0],), self.timesteps - 1, dtype=torch.long, device=device)
        x_t = self.q_sample(x_start=x_start, t=t_full, noise=noise)
        x_quarter = x_half = x_three_quarter = x_t

        for n in reversed(range(0, self.timesteps)):
            t = torch.full((x_t.shape[0],), n, dtype=torch.long, device=device)
            x_t = self.p_sample(x_t, text_feat, image_feat, t, n)
            if n == int((self.timesteps - 1) * 0.75):
                x_quarter = x_t
            if n == int((self.timesteps - 1) * 0.5):
                x_half = x_t
            if n == int((self.timesteps - 1) * 0.25):
                x_three_quarter = x_t

        self._generator = None
        return x_t, x_quarter, x_half, x_three_quarter

    # Placeholder so p_sample can reference it even outside a sample() call.
    _generator = None


class CCDRec(RecommenderBase):
    def __init__(self, config, dataloader):
        super(CCDRec, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        # --- FREEDOM backbone dims ---
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        # Diffusion conditions the id embeddings on projected modality features
        # in a shared space, and the modal BPR mixes CF-space users with those
        # projections, so the two dims must match.
        if self.embedding_dim != self.feat_embed_dim:
            raise ValueError(
                "CCDRec requires embedding_size == feat_embed_dim "
                f"(got {self.embedding_dim} vs {self.feat_embed_dim})."
            )
        self.knn_k = config["knn_k"]
        self.n_mm_layers = config["n_mm_layers"]
        self.n_ui_layers = config["num_ui_layers"]
        self.reg_weight = float(config["reg_weight"])
        self.mm_image_weight = config["mm_image_weight"]
        self.n_nodes = self.n_users + self.n_items

        # --- DMA / NDI knobs (renamed from generic w / weight) ---
        self.blend_w = float(config["ccdrec_blend_w"])
        self.ndi_weight = float(config["ccdrec_ndi_weight"])

        # --- CNS knobs ---
        self.sample_k = float(config["sample_k"])
        self.candidate_fraction = float(config["candidate_fraction"])
        self.curriculum_start_epoch = config["curriculum_start_epoch"]
        self.curriculum_step = config["curriculum_step"]
        self.curriculum_end_epoch = config["curriculum_end_epoch"]

        # --- diffusion knobs ---
        self.diff_weight = float(config["diff_weight"])
        self.timesteps = config["timesteps"]
        self.eval_sample_seed = int(config["eval_sample_seed"])

        # Internal epoch counter (the framework hook is no-arg). -1 so the first
        # pre_epoch_processing call lands on epoch 0 (official NDI gate epoch>0).
        # Frozen -- never incremented -- when dropout_rate <= 0: official quirk,
        # see pre_epoch_processing.
        self._epoch_idx = -1
        # Dirty flag for the lazily-refreshed, seeded eval diffusion draw, plus
        # a refresh counter folded into the seed so successive refreshes draw
        # DIFFERENT noise (official pulls fresh global-RNG noise per epoch ->
        # iid snapshots) while each refresh stays deterministic. The counter is
        # a plain attribute (NOT in state_dict): a checkpoint resume restarts
        # the seed sequence at eval_sample_seed + 0 -- no faithfulness impact
        # (official is unseeded; within-cycle determinism and cross-refresh
        # decorrelation both survive a resume).
        self._sample_dirty = True
        self._refresh_index = 0

        # Interaction matrices: COO for the graph, CSR for the negative-sampling
        # mask (sliced per batch by rows AND candidate columns; replaces the
        # official dense n_users x n_items ``interaction_matrix_dense``).
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)
        self._inter_csr = dataloader.inter_matrix(form="csr").astype(np.float32)

        self.norm_adj = build_norm_adj_matrix(
            self.interaction_matrix, self.n_users, self.n_items, self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)

        # --- embeddings ---
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # --- mm_adj (mixed image/text kNN item graph), cached to the dataset dir ---
        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        mm_adj_file = os.path.join(
            dataset_path,
            "mm_adj_freedomdsp_{}_{}.pt".format(self.knn_k, int(10 * self.mm_image_weight)),
        )
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location="cpu", weights_only=False)
        if self.mm_adj is None:
            image_adj = text_adj = None
            if self.v_feat is not None:
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if image_adj is not None and text_adj is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj, image_adj
            self.mm_adj = self.mm_adj.coalesce()
            os.makedirs(dataset_path, exist_ok=True)
            torch.save(self.mm_adj, mm_adj_file)
        self.mm_adj = self.mm_adj.to(self.device)

        # --- internal diffusion module (real selfAttention denoiser) ---
        self.diff = _CCDRecDiffusion(
            timesteps=self.timesteps,
            beta_start=config["beta_start"],
            beta_end=config["beta_end"],
            beta_sche=config["beta_sche"],
            embedding_dim=self.embedding_dim,
        )

        # Diffusion-sample buffers (feed eval scoring AND the next epoch's CNS).
        # Registered so they ride ``.to(device)`` and survive state_dict
        # round-trips. ZERO-init is a deviation: officially these start as None
        # and are first filled by the trainer's eval-time ``sample()``, so any
        # config where an NDI epoch precedes the first eval (e.g. eval_step > 1)
        # CRASHES official sample_neg_items on ``None.shape``; here the same
        # config degrades gracefully -- NDI scores against a zero table until
        # the first refresh.
        zeros = torch.zeros(self.n_items, self.embedding_dim)
        for name in ("sample_x", "sample_quarter_x", "sample_half_x", "sample_three_quarter_x"):
            self.register_buffer(name, zeros.clone())

    # ------------------------------------------------------------------ graph
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self._compute_normalized_laplacian(indices, adj_size)

    def _compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size, dtype=torch.float32)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size, dtype=torch.float32)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size, dtype=torch.float32)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        return rows_inv_sqrt * cols_inv_sqrt

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).to(dtype=torch.long)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def pre_epoch_processing(self):
        """Framework no-arg hook: mark the diffusion sample dirty so the next
        ``full_sort_predict`` refreshes it (seeded; mirrors the trainer-side
        per-epoch ``model.sample()``, which official runs regardless of
        dropout), then rebuild the degree-pruned ``masked_adj`` and advance the
        internal epoch counter.

        Official quirk replicated (ref pre_epoch_processing ll.165-169): the
        ``dropout <= 0`` branch returns BEFORE ``self.epoch_idx`` is updated, so
        with dropout off the counter FREEZES (at 0 officially, at -1 here) and
        the ``epoch_idx > 0`` NDI gate -- hence CNS too -- never activates.
        Unreachable at the official Baby ``dropout_rate`` 0.8."""
        self._sample_dirty = True

        if self.dropout_rate <= 0.0:
            self.masked_adj = self.norm_adj
            return
        self._epoch_idx += 1
        # degree-sensitive edge pruning (FREEDOM)
        degree_len = int(self.edge_values.size(0) * (1.0 - self.dropout_rate))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        keep_indices = self.edge_indices[:, degree_idx]
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse_coo_tensor(
            all_indices, all_values, self.norm_adj.shape, dtype=torch.float32).to(self.device)

    # ---------------------------------------------------------------- forward
    def forward(self, adj, predicted_x, items):
        """DMA: blend the diffusion prediction into the mm-propagated id graph,
        then LightGCN-mean over the UI graph; final item = i_g + h."""
        h = self.item_id_embedding.weight
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        h_diff = h.clone()
        blended = self.blend_w * predicted_x + (1.0 - self.blend_w) * h[items, :]
        h_diff[items, :] = blended

        ego_embeddings = torch.cat((self.user_embedding.weight, h_diff), dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def calculate_loss(self, interaction):
        users, pos_items = interaction[0], interaction[1]
        if len(interaction) > 2:
            neg_items = interaction[2]
        else:
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=pos_items.device)
        items = torch.cat((pos_items, neg_items), dim=0)

        text_feats = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None
        image_feats = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None

        # antithetic timesteps t and T-1-t (ref calculate_loss)
        half = items.shape[0] // 2 + 1
        t = torch.randint(low=0, high=self.timesteps, size=(half,), device=items.device)
        t = torch.cat([t, self.timesteps - t - 1], dim=0)[: items.shape[0]]

        diff_items = self.item_id_embedding.weight[items, :]
        diff_t = text_feats[items, :]
        diff_v = image_feats[items, :]
        # official call order: p_losses(model, id, text, image, t)
        diff_loss, predicted_x = self.diff.p_losses(diff_items, diff_t, diff_v, t)

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj, predicted_x, items)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        neg_i_g_embeddings_diff = neg_i_g_embeddings

        # NDI: after epoch 0, draw a second negative from the curriculum table.
        if self._epoch_idx > 0:
            neg_diff_items = self.sample_neg_items(pos_items, users, ia_embeddings)
            neg_i_g_embeddings_diff = ia_embeddings[neg_diff_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        batch_mf_loss_diff = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings_diff)

        mf_v_loss = mf_t_loss = 0.0
        if self.t_feat is not None:
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])

        return (
            batch_mf_loss * (1 - self.ndi_weight)
            + self.ndi_weight * batch_mf_loss_diff
            + self.reg_weight * (mf_t_loss + mf_v_loss)
            + self.diff_weight * diff_loss
        )

    # -------------------------------------------------------- curriculum NS
    def get_curriculum_neg_sample(self, ia_embeddings):
        """Easy->hard negative source table for the current epoch (ref CNS)."""
        epoch_idx = self._epoch_idx
        start, step, end = self.curriculum_start_epoch, self.curriculum_step, self.curriculum_end_epoch
        if epoch_idx <= start:
            return self.sample_quarter_x
        if epoch_idx <= start + step:
            return self.sample_half_x
        if epoch_idx <= start + 2 * step:
            return self.sample_three_quarter_x
        if epoch_idx <= end:
            return self.sample_x
        return ia_embeddings

    def sample_neg_items(self, pos_items, users, ia_embeddings):
        """Diffusion-derived hard negatives (NDI, ref sample_neg_items).

        Draw a ``candidate_fraction`` slice of items WITH replacement, score
        each user's positive row against the candidates in the curriculum
        embedding space, set every candidate COLUMN whose item the user has
        interacted with to -inf (duplicated candidates included -- official
        ``interaction_matrix_dense[users][:, random_indices]`` masks by column
        position), then pick one from the top ``sample_k`` fraction at random.
        """
        source = self.get_curriculum_neg_sample(ia_embeddings)

        num_samples = int(self.candidate_fraction * source.shape[0])
        num_samples = max(num_samples, 1)
        candidate_ids = torch.randint(0, source.shape[0], (num_samples,), device=source.device)

        pos_embeddings = source[pos_items]
        candidate_embeddings = source[candidate_ids]
        dot_products = torch.matmul(pos_embeddings, candidate_embeddings.t())

        # Mask items each user already interacted with. Slicing the train CSR
        # by batch rows THEN candidate columns yields the same [B, C] 0/1 block
        # as the official dense slice (ref ll.328-329) -- every column position
        # of a duplicated seen item is masked -- without materialising the full
        # n_users x n_items matrix.
        seen_block = torch.from_numpy(
            self._inter_csr[users.cpu().numpy()][:, candidate_ids.cpu().numpy()].toarray()
        ).to(dot_products.device)
        dot_products[seen_block == 1] = float("-inf")

        k = max(int(self.sample_k * dot_products.shape[1]), 1)
        _, top_indices = torch.topk(dot_products, k=k, dim=1)
        random_ids = torch.randint(0, top_indices.shape[1], (len(pos_items),), device=source.device)
        most_similar_ids = top_indices[torch.arange(len(pos_items), device=source.device), random_ids]
        return candidate_ids[most_similar_ids]

    # -------------------------------------------------------- eval / sampling
    def _refresh_sample(self):
        """Seeded full reverse-chain draw -> refresh the diffusion buffers.

        Seeded with ``eval_sample_seed + refresh index``: every
        ``full_sort_predict`` within one refresh cycle is bit-equal
        (deterministic eval), and successive refreshes draw DIFFERENT noise --
        snapshots decorrelate across epochs, matching the official per-epoch
        ``model.sample()`` which pulls fresh global-RNG noise each time (iid
        across epochs). The resulting buffers also feed the next epoch's CNS.
        """
        text_feats = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None
        image_feats = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None
        generator = torch.Generator(device=self.item_id_embedding.weight.device)
        generator.manual_seed(self.eval_sample_seed + self._refresh_index)
        with torch.no_grad():
            predicted_x, quarter, half, three_quarter = self.diff.sample(
                self.item_id_embedding.weight, text_feats, image_feats, generator=generator)
        self.sample_x.copy_(predicted_x)
        self.sample_quarter_x.copy_(quarter)
        self.sample_half_x.copy_(half)
        self.sample_three_quarter_x.copy_(three_quarter)
        self._refresh_index += 1
        self._sample_dirty = False

    def full_sort_predict(self, interaction):
        user = interaction[0]
        if self._sample_dirty:
            self._refresh_sample()
        items = torch.arange(self.n_items, device=self.device)
        restore_user_e, restore_item_e = self.forward(self.norm_adj, self.sample_x, items)
        u_embeddings = restore_user_e[user]
        return torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
