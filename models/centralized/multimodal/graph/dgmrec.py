# coding: utf-8
r"""
DGMRec
################################################
Reference:
    https://github.com/HimTo-Kim/DGMRec (official refactored tree)
    SIGIR'2025: [Disentangling and Generating Modalities for Recommendation in
                 Missing Modality Scenarios]

Faithful port of the official ``src/models/dgmrec.py`` (+ ``CLUBSample`` from
``src/utils/mi_estimator.py``) run in **complete-modality mode**, the standard
baseline protocol.

Mechanism (from the reference implementation)
-----------------------------------------------
* ``cge()``  -- LightGCN mean over ``num_ui_layers`` on the symmetric-normalized
  UI graph (collaborative-filtering branch).
* ``mge()``  -- disentangles each modality into a **general** view
  ``g = sigmoid(shared_encoder(tanh(image_encoder(feats))))`` (the last shared
  layer is common across modalities) and a modality-**specific** view
  ``s = sigmoid(image_encoder_s(feats))``.
* Modality-preference filters ``spmm(adj.T, tanh(pref(user_emb))) / deg`` gate the
  general and specific views; then ``n_mm_layers`` propagation on the per-modality
  cosine-kNN item graphs (build_sim + build_knn_neighbourhood + normalized
  Laplacian). User-side modal embeddings are mean-aggregated through the UI graph.
* Fusion: ``user_emb = cf + ((u_img_g + u_txt_g)/2 + u_img_s + u_txt_s)/3`` (items
  identically); BPR on the fused embeddings.

Losses
------
* ``loss_disentangle = lambda_1 * (loss_club + loss_InfoNCE)`` -- ``CLUBSample`` MI
  upper bound between the specific and general views per modality (item-level) +
  InfoNCE aligning image_g<->text_g (and the user-level general views). The CLUB
  estimators are trained SEPARATELY in ``pre_epoch_processing`` (their own Adam,
  ``club_steps`` steps on ``club_sample_size`` random items maximizing the
  log-likelihood).
* ``loss_generation = loss_gen + loss_recon`` -- cross-modal general generators
  (image2text/text2image) + specific generators from the preference filter, scored
  by MSE against the actual view (``mse_loss_weight``); decoders reconstruct the
  RAW features from ``perturb(cat[g, s].detach())`` (``recon_weight``). Perturb and
  the generators run at TRAIN time only.
* ``loss_align = lambda_2 * (loss_alignUI + loss_alignBM)`` -- CF u<->pos-i,
  summed-general u<->i, per-modality specific u<->pos-i, and backbone<->modality
  alignment (item CF<->item general-sum, user CF<->user general-sum).
* ``loss_reg`` -- CF-embedding L2 (``reg_weight_cf``) + final modal-embedding L2
  (``reg_weight_modal``).

Complete-modality deviations from the shipped code (documented)
--------------------------------------------------------------
* The shipped code CRASHES at ``missing_modal=0`` because it references
  ``missing_items``/``missing_items_t``/``missing_items_v`` unconditionally. Here
  the missing sets are EMPTY numpy arrays, so the ``t/v/tv`` index paths
  (``np.setdiff1d(all_items, empty)``) degenerate to "all batch items", and the
  missing-modal generation (``generate_missing_modal``) and periodic adjacency
  refresh (``update_adj``) are SKIPPED (they only run under missing modality).
* The CLUB estimators are created LAZILY on the first ``pre_epoch_processing``
  call and held OUTSIDE the ``nn.Module`` registry (a plain list attribute), so
  they never appear in ``model.parameters()``. This reproduces the official
  ordering where ``init_mi_estimator()`` runs AFTER the main optimizer is built,
  excluding the CLUB params from it; they are optimized only by their own Adam.
  They are attached via ``object.__setattr__`` (bypassing ``nn.Module``'s
  registration hook entirely), so they are also excluded from ``state_dict()``
  -- resuming from a checkpoint RE-INITIALIZES the CLUB estimators from scratch
  instead of restoring their trained weights (the official code would restore
  them). Eval scores are unaffected (CLUB estimators only feed
  ``loss_disentangle`` during training). This same exclusion is what protects
  the estimators from the framework's per-epoch optimizer param-rescan
  (``core/base/trainer.py`` ``_refresh_optimizer`` re-derives the trainable set
  via ``_get_optimizer_params()`` from ``get_optimizer_params()`` /
  ``model.parameters()`` after the lazy CLUB creation): because they were never
  registered as submodules, no later rescan of ``model.parameters()`` can pick
  them up.
* The missing-modality data harness (mean-impute at init, per-epoch regeneration,
  adjacency blending) is NOT implemented -- it is a documented future option.
* The official LR schedule ``LambdaLR(0.96 ** (epoch / 50))`` is approximated by
  the framework's ``StepLR`` via ``learning_rate_scheduler: [0.96, 50]`` (a step
  decay every 50 epochs), matching the MGCN-family convention in this repo.
  Omitted side effect: within each 50-epoch window ``StepLR`` holds the LR flat
  at ``0.96**k`` while the official schedule decays smoothly toward
  ``0.96**(k+1)``, so our stepped LR runs UP TO +4.17% ABOVE the official
  smooth value at the end of a window (the two are equal at window
  boundaries, where both reduce to ``0.96**k``).
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase
from core.utils import (
    build_knn_neighbourhood,
    build_sim,
    compute_normalized_laplacian,
)


class CLUBSample(nn.Module):
    """Sampled CLUB estimator -- a variational upper bound on the mutual
    information I(x; y). Faithful copy of the official ``src/utils/mi_estimator.py``.

    ``forward`` returns the MI upper bound (used as a differentiable penalty in the
    main loss); ``learning_loss`` returns the negative log-likelihood used to train
    the estimator's own network. Note ``forward`` draws a random permutation
    regardless of train/eval mode -- DGMRec only calls it during training, so eval
    stays deterministic.
    """

    def __init__(self, x_dim, y_dim, hidden_size):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        return self.p_mu(x_samples), self.p_logvar(x_samples)

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-((mu - y_samples) ** 2) / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = -((mu - y_samples) ** 2) / logvar.exp()
        negative = -((mu - y_samples[random_index]) ** 2) / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.0

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


class DGMRec(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.embedding_dim = config["embedding_size"]
        self.n_ui_layers = config["num_ui_layers"]
        self.n_mm_layers = config["n_mm_layers"]
        self.knn_k = config["knn_k"]

        # Disentangle / alignment weights + temperatures.
        self.lambda_1 = config["lambda_1"]
        self.lambda_2 = config["lambda_2"]
        self.infoNCETemp = config["infoNCETemp"]
        self.alignBMTemp = config["alignBMTemp"]
        self.alignUITemp = config["alignUITemp"]

        # Generation / reconstruction / reg literals (all from YAML).
        self.mse_loss_weight = config["mse_loss_weight"]
        self.recon_weight = config["recon_weight"]
        self.perturb_eps = config["perturb_eps"]
        self.reg_weight_cf = config["reg_weight_cf"]
        self.reg_weight_modal = config["reg_weight_modal"]

        # CLUB MI estimator hyper-parameters.
        self.club_lr = config["club_lr"]
        self.club_steps = config["club_steps"]
        self.club_sample_size = config["club_sample_size"]
        self.mi_hidden_dim = config["mi_hidden_dim"]

        # Collaborative-filtering model.
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)
        self.n_nodes = self.n_users + self.n_items
        self.adj = self.scipy_matrix_to_sparse_tensor(
            self.interaction_matrix, torch.Size((self.n_users, self.n_items))
        )
        num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.norm_adj = self.norm_adj.to(self.device)
        # 1 / degree per node (users then items), for the mean-aggregation gates.
        self.num_inters = torch.FloatTensor(1.0 / (num_inters + 1e-7)).to(self.device)

        self.all_items = np.arange(self.n_items)

        # Complete-modality mode: EMPTY missing sets. Every setdiff against these
        # degenerates to "all batch items", and generation / adj-refresh are
        # skipped. This is the fix for the shipped code's missing_modal=0 crash.
        self.missing_modal = False
        self.missing_items_t = np.array([], dtype=np.int64)
        self.missing_items_v = np.array([], dtype=np.int64)
        self.complete_items = np.arange(self.n_items)

        # Multimodal item features + per-modality kNN item graphs.
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False).to(self.device)
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
        self.image_adj = compute_normalized_laplacian(image_adj).to_sparse_coo().to(self.device)
        del image_adj

        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False).to(self.device)
        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
        self.text_adj = compute_normalized_laplacian(text_adj).to_sparse_coo().to(self.device)
        del text_adj

        # Encoder / decoder / preference networks.
        self.image_encoder = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.text_encoder = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_uniform_(self.image_encoder.weight)
        nn.init.xavier_uniform_(self.text_encoder.weight)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        self.image_encoder_s = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.text_encoder_s = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        nn.init.xavier_uniform_(self.image_encoder_s.weight)
        nn.init.xavier_uniform_(self.text_encoder_s.weight)

        self.image_preference_ = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.text_preference_ = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.image_preference_.weight)
        nn.init.xavier_uniform_(self.text_preference_.weight)

        self.image_decoder = nn.Linear(self.embedding_dim * 2, self.v_feat.shape[1])
        self.text_decoder = nn.Linear(self.embedding_dim * 2, self.t_feat.shape[1])
        nn.init.xavier_uniform_(self.image_decoder.weight)
        nn.init.xavier_uniform_(self.text_decoder.weight)

        # Specific-feature generators.
        self.image_gen = self._make_generator()
        self.text_gen = self._make_generator()
        # General-feature cross-modal generators.
        self.image2text = self._make_generator()
        self.text2image = self._make_generator()

        self.act_g = nn.Tanh()

        # CLUB estimators are created LAZILY on the first pre_epoch_processing call
        # and stored OUTSIDE the nn.Module registry so they never enter
        # model.parameters() (and thus never enter the main optimizer). Assigning an
        # nn.Module via the normal path WOULD register it as a submodule, so we set
        # the estimator attributes with object.__setattr__ to bypass
        # nn.Module.__setattr__ entirely. They are optimized only by their own Adam
        # and moved to device / toggled train-eval explicitly.
        object.__setattr__(self, "item_image_estimator", None)
        object.__setattr__(self, "item_text_estimator", None)
        object.__setattr__(self, "_club_estimators", [])
        object.__setattr__(self, "_club_optimizer", None)

    # ------------------------------------------------------------------ helpers
    def _make_generator(self):
        gen = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        gen.apply(self._init_linear)
        return gen

    @staticmethod
    def _init_linear(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def scipy_matrix_to_sparse_tensor(self, matrix, shape):
        indices = torch.LongTensor(np.array([matrix.row, matrix.col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse_coo_tensor(indices, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(
            dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz))
        )
        for key, value in data_dict.items():
            A[key] = value
        sum_arr = (A > 0).sum(axis=1)
        diag = np.array(sum_arr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        norm_adj = torch.sparse_coo_tensor(indices, data, torch.Size((self.n_nodes, self.n_nodes)))
        return sum_arr, norm_adj

    def perturb(self, x):
        """Adversarial-style perturbation used ONLY at training time (recon / gen).

        ``x + sign(x) * normalize(rand) * perturb_eps``. The randomness makes the
        generation / reconstruction terms stochastic during training; it is never
        invoked at eval, keeping ``full_sort_predict`` deterministic.
        """
        noise = torch.rand_like(x).to(self.device)
        return x + torch.sign(x) * F.normalize(noise, dim=-1) * self.perturb_eps

    # -------------------------------------------------------------- MI estimator
    def init_mi_estimator(self):
        """Create the CLUB estimators and their own Adam optimizer, held OUTSIDE
        the nn.Module registry (so their params stay out of model.parameters()).

        Called lazily by ``pre_epoch_processing`` on the first epoch, mirroring the
        official ``init_mi_estimator()`` that runs AFTER the main optimizer is
        built -- the CLUB params are therefore never in the main optimizer.
        """
        # object.__setattr__ bypasses nn.Module.__setattr__ so these estimators are
        # NOT registered as submodules -> their params stay out of model.parameters()
        # and therefore out of the trainer's main optimizer.
        image_estimator = CLUBSample(
            self.embedding_dim, self.embedding_dim, self.mi_hidden_dim
        ).to(self.device)
        text_estimator = CLUBSample(
            self.embedding_dim, self.embedding_dim, self.mi_hidden_dim
        ).to(self.device)
        object.__setattr__(self, "item_image_estimator", image_estimator)
        object.__setattr__(self, "item_text_estimator", text_estimator)
        object.__setattr__(self, "_club_estimators", [image_estimator, text_estimator])

        params = list(image_estimator.parameters()) + list(text_estimator.parameters())
        object.__setattr__(self, "_club_optimizer", torch.optim.Adam(params, lr=self.club_lr))

    def get_optimizer_params(self):
        """Trainable params for the MAIN optimizer -- explicitly the nn.Module
        params (CLUB estimators are not registered here, but we filter defensively
        so a future refactor cannot silently leak them into the main optimizer)."""
        club_ids = {id(p) for est in self._club_estimators for p in est.parameters()}
        return [p for p in self.parameters() if id(p) not in club_ids]

    def pre_epoch_processing(self):
        """Train the CLUB estimators for ``club_steps`` steps on detached inputs.

        Complete-modality mode skips ``generate_missing_modal`` / ``update_adj``
        (they only run under missing modality). The estimators are created here on
        the first call.
        """
        if self.item_image_estimator is None:
            self.init_mi_estimator()

        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()
        # Detach: the estimators are trained on fixed disentangled views, never
        # backprop into the encoders here (their own optimizer, own graph).
        item_image_g = item_image_g.detach()
        item_text_g = item_text_g.detach()
        item_image_s = item_image_s.detach()
        item_text_s = item_text_s.detach()

        for est in self._club_estimators:
            est.train()

        n_sample = min(self.club_sample_size, self.n_items)
        for _ in range(self.club_steps):
            item_rand_idx = torch.randperm(self.n_items)[:n_sample]

            loss_mi = 0.0
            loss_mi += self.item_image_estimator.learning_loss(
                item_image_s[item_rand_idx], item_image_g[item_rand_idx]
            )
            loss_mi += self.item_text_estimator.learning_loss(
                item_text_s[item_rand_idx], item_text_g[item_rand_idx]
            )

            self._club_optimizer.zero_grad()
            loss_mi.backward()
            self._club_optimizer.step()

        for est in self._club_estimators:
            est.eval()

    # ----------------------------------------------------------------- encoders
    def cge(self, user_emb, item_emb, adj):
        """Collaborative filtering: LightGCN mean over ``n_ui_layers``."""
        ego_embeddings = torch.cat((user_emb, item_emb), dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        user_embeddings, item_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )
        return user_embeddings, item_embeddings

    def mge(self):
        """Modality embedding: general (shared last layer) + specific views."""
        item_image_g = F.sigmoid(self.shared_encoder(self.act_g(self.image_encoder(self.image_embedding.weight))))
        item_text_g = F.sigmoid(self.shared_encoder(self.act_g(self.text_encoder(self.text_embedding.weight))))
        item_image_s = F.sigmoid(self.image_encoder_s(self.image_embedding.weight))
        item_text_s = F.sigmoid(self.text_encoder_s(self.text_embedding.weight))
        return item_image_g, item_text_g, item_image_s, item_text_s

    def _preference_filters(self):
        """Per-user modality preference gates, aggregated to items via adj.T / deg."""
        item_image_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.image_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.text_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]
        return item_image_filter, item_text_filter

    # -------------------------------------------------------------------- losses
    def calculate_loss(self, interaction):
        users, pos_items = interaction[0], interaction[1]
        if len(interaction) >= 3:
            neg_items = interaction[2]
        else:
            neg_items = torch.randint(
                0, self.n_items, pos_items.shape, device=pos_items.device
            )

        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj
        )
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        all_items, _ = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)
        all_items_np = all_items.detach().cpu().numpy()

        # Complete-modality: missing sets are empty, so these degenerate to
        # "all batch items".
        t_index = np.setdiff1d(all_items_np, self.missing_items_t)
        v_index = np.setdiff1d(all_items_np, self.missing_items_v)
        tv_index = np.setdiff1d(all_items_np, np.union1d(self.missing_items_t, self.missing_items_v))

        loss_InfoNCE = self.InfoNCE(item_image_g[tv_index], item_text_g[tv_index], temperature=self.infoNCETemp)

        item_image_filter, item_text_filter = self._preference_filters()

        # Filtering (general).
        item_image_g = torch.einsum("ij, ij -> ij", item_image_filter, item_image_g)
        item_text_g = torch.einsum("ij, ij -> ij", item_text_filter, item_text_g)

        # Item-item graph GCN (general).
        for _ in range(self.n_mm_layers):
            item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
            item_text_g = torch.sparse.mm(self.text_adj, item_text_g)
        user_image_g = torch.sparse.mm(self.adj, item_image_g) * self.num_inters[:self.n_users]
        user_text_g = torch.sparse.mm(self.adj, item_text_g) * self.num_inters[:self.n_users]

        # loss_gen: cross-modal general generators + specific generators (train-only
        # perturb). MSE against the actual view on non-missing items.
        item_image_g_gen = self.text2image(self.perturb(item_text_g))
        item_text_g_gen = self.image2text(self.perturb(item_image_g))
        item_image_s_gen = self.image_gen(self.perturb(item_image_filter))
        item_text_s_gen = self.text_gen(self.perturb(item_text_filter))

        loss_gen = 0.0
        loss_gen += self._mse(item_image_s[v_index], item_image_s_gen[v_index], self.mse_loss_weight)
        loss_gen += self._mse(item_text_s[t_index], item_text_s_gen[t_index], self.mse_loss_weight)
        loss_gen += self._mse(item_text_g[tv_index], item_text_g_gen[tv_index], self.mse_loss_weight)
        loss_gen += self._mse(item_image_g[tv_index], item_image_g_gen[tv_index], self.mse_loss_weight)

        # Filtering (specific).
        item_image_s = torch.einsum("ij, ij -> ij", item_image_filter, item_image_s)
        item_text_s = torch.einsum("ij, ij -> ij", item_text_filter, item_text_s)

        # Item-item graph GCN (specific).
        for _ in range(self.n_mm_layers):
            item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
            item_text_s = torch.sparse.mm(self.text_adj, item_text_s)
        user_image_s = torch.sparse.mm(self.adj, item_image_s) * self.num_inters[:self.n_users]
        user_text_s = torch.sparse.mm(self.adj, item_text_s) * self.num_inters[:self.n_users]

        image_embs = torch.concat([user_image_g + user_image_s, item_image_g + item_image_s], dim=0)
        text_embs = torch.concat([user_text_g + user_text_s, item_text_g + item_text_s], dim=0)
        _, item_image_final = torch.split(image_embs, [self.n_users, self.n_items], dim=0)
        _, item_text_final = torch.split(text_embs, [self.n_users, self.n_items], dim=0)

        # MI sampler loss (CLUB upper bound, item-level, per modality).
        loss_club = 0.0
        loss_club += self.item_image_estimator(item_image_s, item_image_g)
        loss_club += self.item_text_estimator(item_text_s, item_text_g)

        loss_InfoNCE += self.InfoNCE(user_image_g[users], user_text_g[users], temperature=self.infoNCETemp)

        loss_alignUI = self.InfoNCE(user_embeddings[users], item_embedding[pos_items], temperature=self.alignUITemp)
        loss_alignUI += self.InfoNCE(
            user_image_g[users] + user_text_g[users],
            item_image_g[pos_items] + item_text_g[pos_items],
            temperature=self.infoNCETemp,
        )
        loss_alignUI += self.InfoNCE(user_image_s[users], item_image_s[pos_items], temperature=self.alignUITemp)
        loss_alignUI += self.InfoNCE(user_text_s[users], item_text_s[pos_items], temperature=self.alignUITemp)

        loss_alignBM = self.InfoNCE(
            item_embedding[pos_items],
            item_image_g[pos_items] + item_text_g[pos_items],
            temperature=self.alignBMTemp,
        )
        loss_alignBM += self.InfoNCE(
            user_embeddings[users],
            user_image_g[users] + user_text_g[users],
            temperature=self.alignBMTemp,
        )

        # Fusion + BPR on the fused embeddings.
        user_emb = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
        item_emb = item_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3
        user_emb, pos_item_emb, neg_item_emb = user_emb[users], item_emb[pos_items], item_emb[neg_items]

        loss_main_bpr = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)

        loss_reg = self.calculate_reg_loss(
            user_embeddings[users], item_embedding[pos_items], item_embedding[neg_items],
            item_image_final[pos_items], item_text_final[pos_items],
        )

        image_final = torch.concat([item_image_g, item_image_s], dim=1)
        text_final = torch.concat([item_text_g, item_text_s], dim=1)
        loss_recon = self.calculate_recon_loss(image_final, text_final)

        loss_disentangle = self.lambda_1 * (loss_club + loss_InfoNCE)
        loss_generation = loss_gen + loss_recon
        loss_align = self.lambda_2 * (loss_alignUI + loss_alignBM)

        return loss_main_bpr + loss_disentangle + loss_generation + loss_align + loss_reg

    def full_sort_predict(self, interaction):
        users = interaction[0]

        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj
        )
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        item_image_filter, item_text_filter = self._preference_filters()

        item_image_g = torch.einsum("ij, ij -> ij", item_image_filter, item_image_g)
        item_text_g = torch.einsum("ij, ij -> ij", item_text_filter, item_text_g)
        for _ in range(self.n_mm_layers):
            item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
            item_text_g = torch.sparse.mm(self.text_adj, item_text_g)
        user_image_g = torch.sparse.mm(self.adj, item_image_g) * self.num_inters[:self.n_users]
        user_text_g = torch.sparse.mm(self.adj, item_text_g) * self.num_inters[:self.n_users]

        item_image_s = torch.einsum("ij, ij -> ij", item_image_filter, item_image_s)
        item_text_s = torch.einsum("ij, ij -> ij", item_text_filter, item_text_s)
        for _ in range(self.n_mm_layers):
            item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
            item_text_s = torch.sparse.mm(self.text_adj, item_text_s)
        user_image_s = torch.sparse.mm(self.adj, item_image_s) * self.num_inters[:self.n_users]
        user_text_s = torch.sparse.mm(self.adj, item_text_s) * self.num_inters[:self.n_users]

        user_emb = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
        item_emb = item_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3

        return user_emb[users] @ item_emb.T

    # ------------------------------------------------------------ loss utilities
    def calculate_reg_loss(self, user_emb, pos_items_emb, neg_item_emb, image_emb, text_emb):
        loss_reg = self.reg_loss(user_emb, pos_items_emb, neg_item_emb) * self.reg_weight_cf
        loss_reg += self.reg_loss(image_emb) * self.reg_weight_modal
        loss_reg += self.reg_loss(text_emb) * self.reg_weight_modal
        return loss_reg

    def calculate_recon_loss(self, image, text):
        """Decoders reconstruct RAW features from perturbed DETACHED [g, s]."""
        item_image_recon = self.image_decoder(self.perturb(image.detach()))
        item_text_recon = self.text_decoder(self.perturb(text.detach()))
        loss = 0.0
        loss += F.mse_loss(item_image_recon, self.image_embedding.weight) * self.recon_weight
        loss += F.mse_loss(item_text_recon, self.text_embedding.weight) * self.recon_weight
        return loss

    @staticmethod
    def _mse(actual, generated, weight):
        return F.mse_loss(actual, generated) * weight

    @staticmethod
    def reg_loss(*embs):
        loss = 0.0
        for emb in embs:
            loss += torch.norm(emb, p=2)
        loss /= embs[-1].shape[0]
        return loss

    @staticmethod
    def bpr_loss(users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    @staticmethod
    def InfoNCE(view1, view2, temperature=0.4):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = torch.exp((view1 * view2).sum(dim=-1) / temperature)
        ttl_score = torch.exp(torch.matmul(view1, view2.transpose(0, 1)) / temperature).sum(dim=1)
        return torch.mean(-torch.log(pos_score / ttl_score))

    def forward(self):
        raise NotImplementedError("DGMRec uses calculate_loss / full_sort_predict, not forward().")
