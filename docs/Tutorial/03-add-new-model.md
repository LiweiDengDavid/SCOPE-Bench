# Adding a New Model

This document covers the complete workflow for adding a model to NexusRec across all four paradigms.

---

## Checklist (all paradigms)

1. Create the model file in the canonical package selected by the paradigm flags:
   `models/centralized/id/{family}/{modelname_lowercase}.py`, `models/centralized/multimodal/{family}/{modelname_lowercase}.py`, `models/federated/id/{modelname_lowercase}.py`, `models/federated/multimodal/{modelname_lowercase}.py`, `models/sequential/id/{modelname_lowercase}.py`, or `models/sequential/multimodal/{modelname_lowercase}.py`
2. Implement `forward()` and `calculate_loss()` (required for all paradigms)
3. Override `full_sort_predict()` for efficient evaluation (strongly recommended)
4. Create `configs/models/{ModelName}.yaml` with the correct paradigm flags
5. For centralized multimodal models: call `self.setup_multimodal_features(config)` in `__init__`
6. For federated models: implement `get_shared_parameters()` and `get_personal_parameters()` covering ALL parameters. `get_shared_parameters()` must resolve to at least one real `state_dict()` key; federated aggregation is split-aware by design and requires a non-empty shared set.
7. Use only canonical config field names — never invent synonyms
8. Every name in `hyper_parameters:` must be read via `config["name"]` in the model code

---

## Paradigm 1: Centralized Non-Multimodal (ID-only)

Simplest paradigm. Use when the model only uses collaborative filtering embeddings.

### File: `models/centralized/id/factorization/mymodel.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import RecommenderBase


class MyModel(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        # self.n_users, self.n_items, self.embed_size, self.device are set by super()
        # self.dropout_rate, self.learning_rate, self.weight_decay are set by super()

        self.user_embedding = nn.Embedding(self.n_users, self.embed_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embed_size)
        # Add your architecture here

    def forward(self, users, items):
        user_e = self.user_embedding(users)
        item_e = self.item_embedding(items)
        # Return scores (shape depends on your loss function)
        return (user_e * item_e).sum(dim=-1)

    def calculate_loss(self, interaction):
        # interaction is a tuple: (user_ids, pos_item_ids, neg_item_ids)
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return loss

    def full_sort_predict(self, interaction, *args, **kwargs):
        # Called at evaluation time. Must return [batch_size, n_items].
        users = interaction[0]
        user_e = self.user_embedding(users)           # [B, D]
        all_item_e = self.item_embedding.weight        # [n_items, D]
        return torch.matmul(user_e, all_item_e.t())    # [B, n_items]
```

### File: `configs/models/MyModel.yaml`

```yaml
is_federated: false
is_multimodal_model: false
is_sequential: false

embedding_size: 64
learning_rate: 0.001
weight_decay: 1e-5
dropout_rate: 0.1
loss_type: bpr

hyper_parameters: ["learning_rate", "weight_decay", "embedding_size"]

parameter_space:
  learning_rate:
    type: "loguniform"
    low: 1.0e-4
    high: 1.0e-1
  weight_decay:
    type: "loguniform"
    low: 1.0e-8
    high: 1.0e-3
  embedding_size:
    type: "choice"
    values: [32, 64, 128, 256]
```

---

## Paradigm 2: Centralized Multimodal

Adds visual and text feature tensors. Same as above, with two differences:

1. Call `self.setup_multimodal_features(config)` in `__init__` to populate `self.v_feat` and `self.t_feat`
2. Project features from `visual_dim`/`text_dim` → `embed_size` (these are different dimensions)

### Critical dimension rule

| Variable | Dimension | Source |
| --- | --- | --- |
| `embedding_size` | 64 (default) | Collaborative latent dimension (CF embeddings, output) |
| `visual_dim` / `text_dim` | 512 (default) | Pre-extracted feature dimension (input to projection) |

Cross-space operations (add, MSELoss) between `embedding_size`-dim and `visual_dim`-dim tensors are a bug.

### File: `models/centralized/multimodal/factorization/mymultimodalmodel.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import RecommenderBase


class MyMultimodalModel(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)
        # After this call:
        #   self.v_feat: Tensor [n_items, visual_dim] or None
        #   self.t_feat: Tensor [n_items, text_dim] or None

        self.user_embedding = nn.Embedding(self.n_users, self.embed_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embed_size)

        # Project pre-extracted features into CF embedding space
        if self.v_feat is not None:
            self.v_proj = nn.Linear(self.v_feat.shape[1], self.embed_size)
        if self.t_feat is not None:
            self.t_proj = nn.Linear(self.t_feat.shape[1], self.embed_size)

    def _get_item_rep(self):
        item_e = self.item_embedding.weight          # [n_items, D]
        if self.v_feat is not None:
            item_e = item_e + self.v_proj(self.v_feat)
        if self.t_feat is not None:
            item_e = item_e + self.t_proj(self.t_feat)
        return item_e

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        user_e = self.user_embedding(users)
        item_e = self._get_item_rep()
        pos_e, neg_e = item_e[pos_items], item_e[neg_items]
        loss = -torch.log(torch.sigmoid((user_e * pos_e).sum(1) - (user_e * neg_e).sum(1))).mean()
        return loss

    def full_sort_predict(self, interaction, *args, **kwargs):
        users = interaction[0]
        user_e = self.user_embedding(users)
        item_e = self._get_item_rep()
        return torch.matmul(user_e, item_e.t())
```

### File: `configs/models/MyMultimodalModel.yaml`

```yaml
is_federated: false
is_multimodal_model: true
is_sequential: false

embedding_size: 64
visual_dim: 512
text_dim: 512
learning_rate: 0.001
weight_decay: 1e-5
```

---

## Paradigm 3: Federated (ID-only or Multimodal)

Federated models split their parameters into **shared** (aggregated across clients) and **personal** (kept local per user). You must implement both methods and they must together cover ALL parameters.

Federated multimodal models do not rely on `RecommenderBase.setup_multimodal_features()` during model construction. That method intentionally no-ops when `is_federated` is true. `FederatedTrainer` loads features with `setup_federated_features()` and injects them into the reusable client model, so federated multimodal code should use `self.v_feat` / `self.t_feat` during `forward()`, `calculate_loss()`, or `full_sort_predict()`, not during early initialization before the trainer exists.

### Parameter split rules

- `get_shared_parameters()` — return a `Dict[str, Tensor]` where keys match `model.state_dict()` key names. These are averaged across clients each round.
- `get_personal_parameters()` — return a `Dict[str, Tensor]` for per-user parameters kept locally.
- `shared_params.keys() ∪ personal_params.keys()` must equal `model.state_dict().keys()` exactly.
- Any parameter in neither set is **orphaned**: it would receive gradient updates but lose them after each round (frozen at initialization). `FederatedTrainer.__init__` raises `ValueError` for orphaned parameters at startup when the shared set resolves correctly. If the shared set is empty or all names are filtered out, aggregation fails fast because `FederatedAggregator` requires a non-empty shared parameter set.

### File: `models/federated/id/myfedmodel.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import RecommenderBase


class MyFedModel(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        # Shared: item-side parameters (all clients share item knowledge)
        self.item_embedding = nn.Embedding(self.n_items, self.embed_size)

        # Personal: user-side parameters (private per user)
        self.user_embedding = nn.Embedding(self.n_users, self.embed_size)
        self.output_layer = nn.Linear(self.embed_size * 2, 1)

    def forward(self, users, items):
        user_e = self.user_embedding(users)
        item_e = self.item_embedding(items)
        return self.output_layer(torch.cat([user_e, item_e], dim=-1)).squeeze(-1)

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        pos_scores = torch.sigmoid(self.forward(users, pos_items))
        neg_scores = torch.sigmoid(self.forward(users, neg_items))
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        return F.binary_cross_entropy(scores, labels)

    def get_shared_parameters(self):
        return {
            "item_embedding.weight": self.item_embedding.weight,
        }

    def get_personal_parameters(self):
        return {
            "user_embedding.weight": self.user_embedding.weight,
            "output_layer.weight": self.output_layer.weight,
            "output_layer.bias": self.output_layer.bias,
        }

    def full_sort_predict(self, interaction, *args, **kwargs):
        users = interaction[0]
        all_items = torch.arange(self.n_items, device=self.device)
        scores = []
        for u in users:
            u_expanded = u.unsqueeze(0).expand(self.n_items)
            scores.append(self.forward(u_expanded, all_items))
        return torch.stack(scores)
```

### File: `configs/models/MyFedModel.yaml`

```yaml
is_federated: true
is_multimodal_model: false
is_sequential: false

embedding_size: 64
optimizer: "sgd"        # SGD is conventional for federated training
learning_rate: 0.01
weight_decay: 1e-5
federated:
  local_epochs: 5
  clients_sample_ratio: 1.0

hyper_parameters: ["learning_rate", "embedding_size"]

parameter_space:
  learning_rate:
    type: "loguniform"
    low: 1.0e-3
    high: 1.0e-1
  embedding_size:
    type: "choice"
    values: [32, 64, 128, 256]
```

### Federated Multimodal Variant

Federated multimodal models (`is_federated: true`, `is_multimodal_model: true`) have a critical difference from centralized multimodal models: **`setup_multimodal_features()` no-ops for federated models**. It explicitly sets `self.v_feat = None` and returns. Features are instead loaded by `FederatedTrainer.__init__()` (via `setup_federated_features()`) and injected into the model after construction.

Consequence: **`self.v_feat` and `self.t_feat` are `None` during `__init__`**. Any code that uses `self.v_feat.shape[1]` to size a projection layer in `__init__` will raise `AttributeError`.

**Correct pattern for federated multimodal `__init__`:**

```python
class MyFedMMModel(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        # Do NOT call setup_multimodal_features() — it no-ops for federated models.
        # Do NOT use self.v_feat.shape[1] here — self.v_feat is None until the trainer injects it.
        # Use config keys for projection layer dimensions:
        self.text_input_dim = config["features"]["text_dim"]
        self.visual_input_dim = config["features"]["visual_dim"]
        self.latent_dim = config["feature_embedding_size"]

        self.v_proj = nn.Linear(self.visual_input_dim, self.latent_dim)
        self.t_proj = nn.Linear(self.text_input_dim, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)

    def forward(self, users, items):
        # self.v_feat and self.t_feat are available here (injected by trainer before forward)
        item_e = self.item_embedding(items)
        if self.v_feat is not None:
            item_e = item_e + self.v_proj(self.v_feat[items])
        if self.t_feat is not None:
            item_e = item_e + self.t_proj(self.t_feat[items])
        user_e = self.user_embedding(users)
        return (user_e * item_e).sum(dim=-1)
```

**Model YAML for federated multimodal** — current federated multimodal models use `features.text_dim` / `features.visual_dim` for pre-extracted feature dimensions and `feature_embedding_size` for the collaborative latent dimension. Some legacy YAMLs still carry a flat `embedding_size` for compatibility with model-specific code, but new code should read the nested feature contract explicitly:

```yaml
is_federated: true
is_multimodal_model: true
is_sequential: false

embedding_size: 512          # legacy/model-specific top-level dimension if the model reads it
feature_embedding_size: 64   # latent collaborative dimension
learning_rate: 0.01
weight_decay: 1e-5
server_learning_rate: 0.01   # required for delta aggregation path
optimizer: "sgd"
federated:
  local_epochs: 5
```

---

## Paradigm 4: Sequential

Sequential models inherit from `SequentialRecommender` instead of `RecommenderBase`.

### What `SequentialRecommender` provides

**Config-derived attributes:**
- `self.max_seq_len`, `self.hidden_size`, `self.num_layers`, `self.dropout_rate` — from config
- `self.embed_size = config["embedding_size"]` — inherited from `RecommenderBase` (note: `embed_size`, not `embedding_size`)

`SequentialRecommender.__init__` deliberately does not create embedding, projection, norm, or dropout modules. Each concrete sequential model owns its item embedding and decoder shape, which keeps checkpoints free of unused base tables.

**Methods:**
- `gather_indexes(output, gather_index)` — extract hidden states at specific positions
- `get_attention_mask(item_seq, seq_lens)` — boolean mask for valid positions

### Required override: `encode_sequence()`

```python
def encode_sequence(
    self,
    item_seq: torch.Tensor,       # [batch_size, seq_len] — item IDs (0 = PAD)
    seq_lens: torch.Tensor,       # [batch_size] — actual (non-padding) lengths
    v_feat_seq=None,              # [batch_size, seq_len, visual_dim] or None
    t_feat_seq=None,              # [batch_size, seq_len, text_dim] or None
) -> torch.Tensor:                # must return [batch_size, seq_len, hidden_size]
```

### Interaction dict keys (sequential training)

| Key | Shape | Description |
| --- | --- | --- |
| `"user_ids"` | `[B]` | User IDs for the batch |
| `"item_seqs"` | `[B, L]` | Input sequence (padded with 0) |
| `"targets"` | `[B]` | Target item ID |
| `"seq_lens"` | `[B]` | Actual sequence length (excludes padding) |
| `"seq_masks"` | `[B, L]` | Boolean mask for non-padding sequence positions |
| `"neg_items"` | `[B]` (optional) | Negative item for BPR loss |
| `"targets_list"` | `[B, T]` (eval optional) | Padded multi-target list for sequential evaluation |
| `"num_targets"` | `[B]` (eval optional) | Number of valid targets per row in `targets_list` |

### File: `models/sequential/id/myseqmodel.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.sequential.recommender import SequentialRecommender


class MySeqModel(SequentialRecommender):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        # Attributes set by SequentialRecommender.__init__ (via super()):
        #   self.n_items, self.max_seq_len, self.hidden_size, self.num_layers,
        #   self.dropout_rate
        #
        # self.embed_size (= config["embedding_size"]) is set by RecommenderBase.
        # If your model uses a separate embedding_size (different from hidden_size),
        # declare it yourself — the base class does NOT set self.embedding_size:
        self.embedding_size = config["embedding_size"]

        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0
        )
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

    def encode_sequence(self, item_seq, seq_lens, v_feat_seq=None, t_feat_seq=None):
        item_emb = self.item_embedding(item_seq)    # [B, L, emb_size]
        packed = nn.utils.rnn.pack_padded_sequence(
            item_emb, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        seq_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=item_seq.size(1)
        )
        return seq_output    # [B, L, hidden_size]

    def decode_next(self, seq_representation):
        projected = self.dense(seq_representation)          # [B, emb_size]
        all_item_emb = self.item_embedding.weight           # [n_items+1, emb_size]
        return torch.matmul(projected, all_item_emb.t())    # [B, n_items+1]

    def forward(self, item_seq, seq_lens):
        seq_output = self.encode_sequence(item_seq, seq_lens)
        user_repr = self.gather_indexes(seq_output, seq_lens - 1)
        return self.decode_next(user_repr)

    def calculate_loss(self, interaction):
        scores = self.forward(interaction["item_seqs"], interaction["seq_lens"])
        return F.cross_entropy(scores, interaction["targets"])

    def full_sort_predict(self, interaction, *args, **kwargs):
        return self.forward(interaction["item_seqs"], interaction["seq_lens"])
```

### File: `configs/models/MySeqModel.yaml`

```yaml
is_federated: false
is_multimodal_model: false
is_sequential: true

embedding_size: 64
hidden_size: 128
num_layers: 2
dropout_rate: 0.2
max_seq_len: 50
loss_type: ce

hyper_parameters: ["learning_rate", "hidden_size", "num_layers", "dropout_rate"]

parameter_space:
  hidden_size:
    type: "choice"
    values: [64, 128, 256]
  num_layers:
    type: "choice"
    values: [1, 2, 3]
  dropout_rate:
    type: "uniform"
    low: 0.0
    high: 0.5
  learning_rate:
    type: "loguniform"
    low: 1.0e-4
    high: 1.0e-2
```

---

## `full_sort_predict()` Contract

This method is called at evaluation time and must satisfy:

- **Input:** `interaction` — same dict/tuple format as `calculate_loss()`
- **Output:** centralized/federated models return `Tensor [batch_size, n_items]` where index `i` corresponds to item ID `i` and item `0` is a normal item. Sequential models may return `[batch_size, n_items+1]` when index `0` is PAD; only `SequentialEvaluator` suppresses PAD `0`.
- **Must not** modify model state
- **Should** run in `torch.no_grad()` context (caller wraps this, but verify if overriding)

If you do not override it: `RecommenderBase` subclasses (centralized/federated) get a slow per-user loop default — override with a batched matrix multiplication for any non-trivial dataset. Sequential models should implement `full_sort_predict()` explicitly because the base class only provides sequence utilities and an eval-time dispatcher.

Recommendation export uses the same final-test top-k indices produced from `full_sort_predict()`. New models do not implement export-specific code, but their item-id semantics must match the contract above: centralized/federated index `i` maps to item id `i`, while sequential PAD index `0` is removed by `SequentialEvaluator` and exported items are shifted back to the shared zero-based NexusRec internal item index.

---

## Agent Checklist

For AI agents implementing a new model, verify all of the following before considering the task complete:

- [ ] The model file exists in the canonical package implied by the three paradigm flags and declares `class Name(...)`
- [ ] `configs/models/{Name}.yaml` exists with correct paradigm flags (`is_federated`, `is_multimodal_model`, `is_sequential`)
- [ ] `calculate_loss(interaction)` returns a scalar loss tensor
- [ ] `full_sort_predict(interaction)` returns the correct shape: `[B, n_items]` for centralized/federated models; `[B, n_items+1]` for sequential models using the item-embedding dot-product path (PAD token at index 0, score ignored by evaluator); `[B, n_items]` when using `output_projection`
- [ ] Final-test top-k item indices can be mapped back to NexusRec internal item ids; this is required for `output.export`
- [ ] Centralized multimodal model: `self.setup_multimodal_features(config)` called in `__init__`, and projection layers use `self.v_feat.shape[1]` (not a hardcoded dimension)
- [ ] Federated multimodal model: do NOT use `self.v_feat.shape[1]` in `__init__` — use `config["features"]["visual_dim"]`, `config["features"]["text_dim"]`, and latent config keys such as `config["feature_embedding_size"]`; `self.v_feat` is injected by the trainer after construction
- [ ] Federated model: `get_shared_parameters()` and `get_personal_parameters()` together cover `set(model.state_dict().keys())` exactly
- [ ] Every name in `hyper_parameters:` list is accessed as `config["name"]` in model code
- [ ] All config reads use `config["key"]` — never `config.get("key", default)`
- [ ] No numeric literals in Python code that affect training/evaluation outcomes; those belong in YAML

---

## Common Mistakes and Anti-Patterns

This section documents the most frequent implementation errors, organized by paradigm. Each entry shows the wrong pattern alongside the correct replacement.

---

### All paradigms

#### Using `config.get()` with a default

This is the single most common mistake. It silently hides two classes of bugs: a key that was renamed and the old name was never cleaned up, and an intentionally-zero value that the default overrides.

**Wrong:**
```python
self.dropout_rate = config.get("dropout_rate", 0.1)
self.temperature = config.get("temperature", 0.07)
```

**Correct:**
```python
self.dropout_rate = config["dropout_rate"]
self.temperature = config["temperature"]
```

If the key might not be present in all configs, add it to `configs/overall.yaml` with its default. That way the default is explicit, version-controlled, and visible.

---

#### Hardcoded numeric literals that affect experiment outcomes

Numeric values that control training, architecture, or evaluation must live in YAML. If a researcher must edit `.py` to change a parameter, it is hardcoded config.

**Wrong:**
```python
self.temperature = 0.07
self.n_layers = 2
self.hidden_dim = 128
loss = loss * 0.1  # fixed contrastive weight
```

**Correct:**
```python
self.temperature = config["temperature"]
self.n_layers = config["num_layers"]
self.hidden_dim = config["hidden_size"]
self.ssl_weight = config["ssl_weight"]
loss = loss * self.ssl_weight
```

Then in the model YAML:
```yaml
temperature: 0.07
num_layers: 2
hidden_size: 128
ssl_weight: 0.1
```

---

#### Non-canonical config key names

Using a non-canonical name means the HPO infrastructure, configuration system, and all documentation refer to different names for the same concept. A parameter called `tau` in the code but `temperature` in YAML will cause a `KeyError` during HPO.

**Wrong:**
```python
self.tau = config["tau"]
self.n_layers = config["n_layers"]
self.dropout = config["dropout"]
```

**Correct:**
```python
self.temperature = config["temperature"]
self.num_layers = config["num_layers"]
self.dropout_rate = config["dropout_rate"]
```

The full canonical name table is in [02-configuration.md](02-configuration.md). When porting a model from another codebase, rename to canonical before wiring any config reads.

---

#### Dead HPO parameters

A parameter in `hyper_parameters:` that is never read by `config["name"]` wastes HPO budget searching a dimension that has no effect on the model.

**Wrong (YAML):**
```yaml
hyper_parameters: ["learning_rate", "gamma", "embedding_size"]

parameter_space:
  gamma:
    type: "uniform"
    low: 0.1
    high: 1.0
```

**Wrong (Python) — `gamma` declared in YAML but never read:**
```python
def __init__(self, config, dataloader):
    super().__init__(config, dataloader)
    self.embed = nn.Embedding(self.n_items, self.embed_size)
    # gamma is never used
```

**Correct:** Either remove `gamma` from `hyper_parameters:` and `parameter_space:`, or wire it into the model:

```python
self.gamma = config["gamma"]
```

---

### Centralized multimodal models

#### Forgetting `setup_multimodal_features()`

If `setup_multimodal_features(config)` is not called in `__init__`, `self.v_feat` and `self.t_feat` remain `None`. Any code that accesses `self.v_feat.shape[1]` raises `AttributeError: 'NoneType' object has no attribute 'shape'`, but if the code guards with `if self.v_feat is not None:`, features are silently skipped and the model trains as if it were an ID-only model.

**Wrong:**
```python
class MyMM(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        # self.v_feat is None here — setup was never called
        self.v_proj = nn.Linear(512, self.embed_size)  # hardcoded 512!
```

**Correct:**
```python
class MyMM(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)
        # Now self.v_feat is Tensor[n_items, visual_dim] or None
        if self.v_feat is not None:
            self.v_proj = nn.Linear(self.v_feat.shape[1], self.embed_size)
        if self.t_feat is not None:
            self.t_proj = nn.Linear(self.t_feat.shape[1], self.embed_size)
```

---

#### Dimension mismatch: visual_dim vs. embed_size

Pre-extracted features have a large dimension (typically 512). The CF embedding space has a small dimension (typically 64). These two spaces cannot be directly added, multiplied element-wise, or compared with MSELoss without a projection.

**Wrong:**
```python
# self.v_feat is [n_items, 512], self.item_embedding.weight is [n_items, 64]
item_rep = self.item_embedding.weight + self.v_feat  # shape error or silent wrong result
```

**Correct:**
```python
# Project 512 → 64 first
v_rep = self.v_proj(self.v_feat)          # [n_items, 64]
item_rep = self.item_embedding.weight + v_rep  # [n_items, 64] — correct
```

Always derive projection dimensions from the loaded tensor, not from a hardcoded value:

```python
self.v_proj = nn.Linear(self.v_feat.shape[1], self.embed_size)  # correct
self.v_proj = nn.Linear(512, self.embed_size)                    # wrong — hardcoded
```

---

#### Not moving tensors to device

`RecommenderBase.to(device)` moves `self.v_feat` and `self.t_feat` automatically for any tensors that are already assigned at the moment `.to(device)` is called. In the normal training flow, the trainer calls `.to(device)` once after constructing the model, before any training begins. If you assign feature tensors in a `pre_epoch_processing()` hook — which runs each epoch, after the initial `.to(device)` call — those tensors are on CPU and will not be moved automatically. You must call `.to(self.device)` explicitly.

**Wrong:**
```python
def pre_epoch_processing(self):
    # Reloads features from disk each epoch — wrong pattern, but shows the device bug
    self.v_feat = torch.from_numpy(np.load("...")).float()
    # Tensor is on CPU; model is on GPU → RuntimeError on next forward pass
```

**Correct:**
```python
def pre_epoch_processing(self):
    self.v_feat = torch.from_numpy(np.load("...")).float().to(self.device)
```

In normal operation, call `setup_multimodal_features(config)` once in `__init__` and let `RecommenderBase.to(device)` handle placement.

---

### Federated models

#### Orphaned parameters

An orphaned parameter is one that appears in `model.state_dict()` but not in either `get_shared_parameters()` or `get_personal_parameters()`. It would receive gradient updates during local training but those updates would be discarded at aggregation, leaving it frozen at its initialization value. `FederatedTrainer.__init__` raises `ValueError` on the first orphaned parameter, so a broken split is caught before training starts.

**Wrong:**
```python
class MyFed(RecommenderBase):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.item_embedding = nn.Embedding(self.n_items, self.embed_size)
        self.user_embedding = nn.Embedding(self.n_users, self.embed_size)
        self.output_layer = nn.Linear(self.embed_size * 2, 1)

    def get_shared_parameters(self):
        return {"item_embedding.weight": self.item_embedding.weight}

    def get_personal_parameters(self):
        return {"user_embedding.weight": self.user_embedding.weight}
    # output_layer.weight and output_layer.bias are orphaned
```

**Correct:**
```python
def get_personal_parameters(self):
    return {
        "user_embedding.weight": self.user_embedding.weight,
        "output_layer.weight": self.output_layer.weight,
        "output_layer.bias": self.output_layer.bias,
    }
```

To catch orphans before training, run this check after instantiation:

```python
model = MyFed(config, dataloader)
all_keys = set(model.state_dict().keys())
shared = set(model.get_shared_parameters().keys())
personal = set(model.get_personal_parameters().keys())
assert all_keys == shared | personal, f"Orphaned: {all_keys - shared - personal}"
```

---

#### Wrong parameter names in `get_shared_parameters()`

Parameter names must exactly match the keys returned by `model.state_dict()`. A single-character typo causes the parameter to be filtered out before split validation. If this leaves no valid shared parameter, federated aggregation fails fast because it cannot run without a shared set.

**Wrong:**
```python
def get_shared_parameters(self):
    return {
        "item_emb.weight": self.item_embedding.weight,  # actual key: "item_embedding.weight"
    }
```

**Correct:** Get the exact names first:

```bash
python -c "
from models.federated.id.myfedmodel import MyFedModel
import torch
# minimal config needed to instantiate
for k in MyFedModel(config, dataloader).state_dict().keys():
    print(k)
"
```

Then copy the exact strings into `get_shared_parameters()` and `get_personal_parameters()`.

---

#### Using `deepcopy` per client instead of `load_state_dict`

Some federated learning codebases (e.g., older RecBole-FedRec implementations) create one deep copy of the model per client. In NexusRec this is the wrong pattern: the framework uses a single reusable `_client_model` instance with in-place state swaps. Do not implement your own training loop inside a model that calls `copy.deepcopy(self)`.

**Wrong (in model code):**
```python
def train_local(self, user_id, dataloader):
    local_model = copy.deepcopy(self)  # wrong — N copies × GPU memory
    # ... train local_model ...
    return local_model.state_dict()
```

**Correct:** Implement only `calculate_loss()`, `get_shared_parameters()`, and `get_personal_parameters()`. The `FederatedTrainer` handles all local training orchestration.

---

### Sequential models

#### Inheriting from `RecommenderBase` instead of `SequentialRecommender`

Sequential models must inherit from `SequentialRecommender` (in `core/sequential/recommender.py`), not `RecommenderBase`. Using the wrong base class means `encode_sequence()`, `get_attention_mask()`, `gather_indexes()`, and `aggregate_sequence()` are not available, and the file must be placed under `models/sequential/id/` or `models/sequential/multimodal/` to be discovered correctly.

**Wrong:**
```python
# models/sequential/id/myseqmodel.py
from core.base.recommender import RecommenderBase

class MySeqModel(RecommenderBase):  # wrong base class
    ...
```

**Correct:**
```python
# models/sequential/id/myseqmodel.py
from core.sequential.recommender import SequentialRecommender

class MySeqModel(SequentialRecommender):  # correct
    ...
```

Also ensure the model YAML has `is_sequential: true` and the file is at `models/sequential/id/{modelname_lowercase}.py` or `models/sequential/multimodal/{modelname_lowercase}.py`.

---

#### Using wrong interaction dict keys

Sequential models receive interactions as a dict. The keys are defined in the sequential dataloader. Using the wrong key raises a `KeyError` that does not point to the real problem.

**Wrong:**
```python
def calculate_loss(self, interaction):
    seqs = interaction["sequences"]    # KeyError: "sequences"
    target = interaction["target"]     # KeyError: "target"
```

**Correct:**
```python
def calculate_loss(self, interaction):
    seqs = interaction["item_seqs"]    # [B, L]
    target = interaction["targets"]    # [B]
    seq_lens = interaction["seq_lens"] # [B]
```

The full key reference is in the Paradigm 4 section above: `"item_seqs"`, `"targets"`, `"seq_lens"`, and optionally `"neg_items"`.

---

#### Returning `[B, n_items]` instead of `[B, n_items+1]` from `decode_next()`

Sequential item embeddings include a PAD token at index 0, so the embedding table has shape `[n_items+1, D]`. When using the item-embedding dot-product scoring path, `decode_next()` should return `[B, n_items+1]` (index 0 = PAD, its score is ignored by the evaluator). Using `output_projection` (which maps to `n_items` without the PAD column) is also valid; the evaluator handles both shapes.

The mismatch only becomes a bug if you mix the two: returning a `[B, n_items]` tensor but building the item embedding as `nn.Embedding(n_items+1, ...)` — the index correspondence breaks.

**Wrong (mixed):**
```python
self.item_embedding = nn.Embedding(self.n_items + 1, self.embed_size, padding_idx=0)
self.output_layer = nn.Linear(self.hidden_size, self.n_items)  # outputs [B, n_items]

def decode_next(self, seq_repr):
    # item_embedding has n_items+1 rows but output_layer produces n_items columns
    # → item 1 maps to output column 0, off-by-one error in evaluation
    return self.output_layer(seq_repr)
```

**Correct (dot-product path):**
```python
self.item_embedding = nn.Embedding(self.n_items + 1, self.embed_size, padding_idx=0)

def decode_next(self, seq_repr):
    # seq_repr: [B, embed_size]
    # item_embedding.weight: [n_items+1, embed_size]
    return torch.matmul(seq_repr, self.item_embedding.weight.t())  # [B, n_items+1]
```

**Correct (projection path):**
```python
self.output_projection = nn.Linear(self.hidden_size, self.n_items)

def decode_next(self, seq_repr):
    return self.output_projection(seq_repr)  # [B, n_items] — evaluator handles this too
```
