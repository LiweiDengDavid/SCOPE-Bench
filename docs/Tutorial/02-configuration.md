# Configuration System

NexusRec uses a four-layer config hierarchy. Each layer deep-merges into the previous, with later layers taking precedence.

```
configs/overall.yaml          ← global defaults for every model and dataset
       ↓  (deep merge)
configs/models/{MODEL}.yaml   ← model-specific overrides and HPO search space
       ↓  (deep merge)
configs/datasets/{DATASET}.yaml  ← dataset-specific overrides (optional, directory may be absent)
       ↓  (deep merge)
CLI args + --param_overrides  ← runtime overrides, highest precedence
```

Dataset configs may also include `model_overrides`. When present,
`model_overrides.{MODEL}` is deep-merged after the shared dataset keys and
before CLI overrides. Use it for dataset-specific tuned defaults that should
apply only to one selected model:

```yaml
model_overrides:
  BPR:
    learning_rate: 0.003
    embedding_size: 128
```

---

## Flattened Groups

`configs/overall.yaml` organizes keys under logical group headings (`training:`, `evaluation:`, `output:`, etc.). At runtime, `ConfigManager` promotes all keys in these groups to the top level so the entire codebase uses flat access:

```python
config["max_epochs"]     # not config["training"]["max_epochs"]
config["valid_metric"]   # not config["evaluation"]["valid_metric"]
config["log_path"]       # not config["output"]["log_path"]
config["export"]         # promoted from output.export
```

**Flattened groups:** `training`, `evaluation`, `output`, `experiment`, `sequential`, `sampling`, `resources`

Model YAML files use flat keys directly for scalar runtime parameters. Nested dictionaries inside flattened groups are preserved as dictionaries when promoted: `output.export` becomes runtime `config["export"]`, while `optimization.final_train` remains nested because `optimization` is not flattened.

---

## Key Config Sections

### Training

| Key | Default | Description |
|---|---|---|
| `max_epochs` | `500` | Maximum training epochs |
| `learning_rate` | `0.001` | Optimizer learning rate |
| `weight_decay` | `1e-5` | L2 weight decay |
| `train_batch_size` | `2048` | Training mini-batch size |
| `eval_batch_size` | `4096` | Evaluation batch size |
| `optimizer` | `"adam"` | Optimizer: `adam`, `sgd`, `rmsprop`, `adamw` |
| `loss_type` | `"bpr"` | Default loss: `bpr`, `ce`, `top1` |
| `early_stopping` | `true` | Enable early stopping |
| `stopping_step` | `10` | Patience (epochs without improvement) |
| `clip_grad_norm` | `5.0` | Gradient clipping norm |
| `eval_step` | `1` | Evaluate every N epochs |
| `nan_abort_threshold` | `3` | Consecutive NaN/Inf losses before abort |

### Evaluation

| Key | Default | Description |
|---|---|---|
| `valid_metric` | `"NDCG@10"` | Metric for early stopping and HPO target |
| `metrics` | `["Recall", "NDCG", "Precision"]` | Metrics to compute |
| `topk` | `[10, 20, 50]` | Cutoffs for all metrics |
| `filter_seen` | `true` | Filter training items from prediction |
| `test_history_mask` | `"train_valid"` | Test masking protocol: `train_valid` or `train_only` |
| `eval_test_during_training` | `false` | Evaluate test split during training epochs |
| `eval_final_test` | `true` | Evaluate test split once with validation-best state at the end |
| `eval_test_frequency` | `1` | Test evaluation interval when test-during-training is enabled |
| `valid_metric_bigger` | `true` | Whether higher `valid_metric` is better |
| `item_bucket_metrics` | `false` | Add head/mid/tail metrics based on training item popularity buckets |
| `item_bucket_tail_quantile` | `0.2` | Tail cutoff quantile for bucket metrics |
| `item_bucket_head_quantile` | `0.8` | Head cutoff quantile for bucket metrics |

Supported metric names are `Recall`, `Precision`, `NDCG`, `MAP`, `MRR`, `Hit`, `Diversity`, `Novelty`, and `Coverage`.

### CDS/LCDS evaluation

`lcds.enabled` is false globally and enabled by the ShortVideoSampled and ShortVideoFull dataset configs. When enabled, the evaluator automatically appends `A-LCDS@K` and `E-LCDS@K` using every cutoff in `evaluation.topk`; these names do not need to be added to `evaluation.metrics`.

```yaml
lcds:
  enabled: true
  cds_jsonl: "scoring/results/Qwen3_7_Max_full_t0p3_seed42_scores.jsonl"
  dataset_dir: ""  # empty resolves to data_path/dataset
  gain_divisor: 6.0
```

The CDS JSONL, `id_mappings.json`, and `items.json` are required when enabled. Numeric scores map to `score / gain_divisor`; null or missing labels use zero gain.

### Output and recommendation export

| Key | Default | Description |
|---|---|---|
| `save_model` | `true` | Save `best_model.pth` after validation-best selection |
| `save_recommended_topk` | `false` | Legacy diagnostic TSV of internal item ids in `checkpoint_dir` |
| `export.enabled` | `false` | Enable recommendation-list export during final test evaluation |
| `export.formats` | `["json"]` | Non-empty unique subset of `json`, `jsonl`, `csv`, `tsv` |
| `export.include_scores` | `true` | Add post-mask raw ranking scores to each recommendation |
| `export.split` | `"test"` | Only `test` is currently supported |
| `export.topk` | `null` | `null` exports the full evaluated top-k width; otherwise a positive integer |
| `export.path` | `""` | Empty means `config["paths"]["save"]`; non-empty writes to that directory |

Write this in YAML under `output.export`, even though code reads it as `config["export"]` after flattening:

```yaml
output:
  export:
    enabled: true
    formats: ["json"]
    include_scores: true
    split: "test"
    topk: 50
```

`output.export` is mutually exclusive with `save_recommended_topk`. Use `output.export` for offline pipelines and on-device reranking because it writes stable recommendation-list artifacts; keep `save_recommended_topk` only for legacy internal-id diagnostics.

### HPO final training

`optimization.final_train` controls the optional second stage after HPO:

```yaml
optimization:
  final_train:
    enabled: false
    overrides:
      type: "final_train"
      comment: "best_config"
      save_model: true
      resume_training: false
      eval_final_test: true
      eval_test_during_training: false
```

When enabled, HPO first selects `best_configuration`, then NexusRec starts one ordinary training run using `input_config + best_configuration + final_train.overrides`. The final-train run disables `smart_hpo`, disables parallel HPO, disables `optimization.final_train` recursively, and records `hpo_lineage` in the result/export metadata. HPO trials themselves always disable `output.export`.

### Features (Multimodal)

Feature config lives under the `features:` group in `overall.yaml`, which is **not** in `_FLATTEN_GROUPS`. These keys are not promoted to top-level at runtime.

The feature loader (`core/data/features.py`) reads them as `config["features"]["vision_feature_file"]` etc. Centralized multimodal model code can use `self.v_feat.shape[1]` to get the actual dimension at runtime; federated multimodal model code must size layers from `config["features"]` because features are injected after model construction.

| YAML key (nested under `features:`) | Default | Description |
|---|---|---|
| `vision_feature_file` | `"image_features.npy"` | Filename under `datasets/{DATASET}/` |
| `text_feature_file` | `"text_features.npy"` | Filename under `datasets/{DATASET}/` |
| `visual_dim` | `512` | Declared visual feature dimension; validated against `.npy` shape |
| `text_dim` | `512` | Declared text feature dimension; validated against `.npy` shape |

Centralized multimodal models should size projection layers from loaded tensors (`self.v_feat.shape[1]`, `self.t_feat.shape[1]`). Federated multimodal models construct layers before trainer feature injection, so they should read `config["features"]["visual_dim"]`, `config["features"]["text_dim"]`, and their latent dimension keys explicitly.

### Sequential

| Key | Default | Description |
|---|---|---|
| `max_seq_len` | `50` | Maximum input sequence length |
| `min_seq_len` | `3` | Minimum sequence length to keep |
| `split_method` | `"leave_one_out"` | Sequential split protocol |
| `hybrid_threshold` | `10` | Hybrid split threshold |
| `data_augmentation` | `false` | Enable sequential augmentation |
| `augmentation_strategies` | `["crop", "mask", "reorder"]` | Augmentation operators |
| `augmentation_mask_ratio` | `0.2` | Fraction of sequence items to mask |
| `augmentation_max_mask` | `3` | Maximum items masked per sequence |
| `augmentation_min_seq_len` | `3` | Minimum sequence length for crop/reorder augmentation |
| `neg_sampling` | `false` | Sequential BPR negative sampling switch |
| `train_ratio` | `0.8` | Ratio-split training fraction |
| `valid_ratio` | `0.1` | Ratio-split validation fraction |

### Federated

Federated settings stay nested under `federated:`. `extract_federated_params()` reads `config["federated"]`; these keys are not flattened to top level.

| Key | Default | Description |
|---|---|---|
| `clients_sample_ratio` | `1.0` | Fraction of clients to sample per round |
| `clients_sample_strategy` | `"random"` | Sampling strategy: `random` |
| `local_epochs` | `5` | Local training epochs per round |
| `aggregation_method` | `"fedavg"` | Aggregation: `fedavg` |

---

## Model YAML Structure

Every model requires a YAML under `configs/models/{ModelName}.yaml`.

### Minimal example (centralized non-multimodal)

```yaml
# configs/models/MyModel.yaml
is_federated: false
is_multimodal_model: false
is_sequential: false

embedding_size: 64
learning_rate: 0.001
weight_decay: 1e-5
dropout_rate: 0.1
```

### With HPO search space

```yaml
is_federated: false
is_multimodal_model: true
is_sequential: false

embedding_size: 64
learning_rate: 0.001
weight_decay: 1e-5

hyper_parameters: ["learning_rate", "weight_decay", "embedding_size"]

parameter_space:
  learning_rate:
    type: "loguniform"
    low: 1.0e-5
    high: 1.0e-1
  weight_decay:
    type: "loguniform"
    low: 1.0e-8
    high: 1.0e-3
  embedding_size:
    type: "choice"
    values: [32, 64, 128, 256]
```

### Parameter space types

| `type` | Required fields | Description |
|---|---|---|
| `"choice"` | `values: [...]` | Discrete list of options |
| `"uniform"` | `low`, `high` | Continuous uniform range |
| `"loguniform"` | `low`, `high` | Log-scale continuous range |
| `"logscale"` | `low`, `high` | Alias supported by the HPO samplers |
| `"int"` | `low`, `high` | Integer range (inclusive) |
| `"logint"` | `low`, `high` | Integer range sampled on a log scale |

Grid search only varies explicit discrete values. Use `grid_values` or `choice.values` for grid-search dimensions; range-only types fall back to the resolved default value in grid mode.

**Rule:** Every name in `hyper_parameters` must be read somewhere in model code as `config["name"]`. Names that are never read waste HPO budget searching a meaningless dimension.

---

## Runtime Overrides

### Via `--param_overrides`

Any flat key or nested structure can be overridden:

```bash
# Single value
python main.py --model VBPR --dataset Beauty \
    --param_overrides '{"learning_rate": 0.01}'

# Nested override (flattened groups are accessed flat, nested dicts use dict syntax)
python main.py --model VBPR --dataset Beauty \
    --param_overrides '{"multimodal_ablation": {"visual": "remove"}}'

# Multiple values
python main.py --model SASRec --dataset Beauty \
    --param_overrides '{"max_seq_len": 30, "num_layers": 3, "dropout_rate": 0.3}'

# Enable automatic formal training after HPO
python main.py --model VBPR --dataset Beauty --smart_hpo \
    --param_overrides '{"optimization": {"final_train": {"enabled": true}}}'

# Enable recommendation-list export
python main.py --model VBPR --dataset Beauty \
    --param_overrides '{"output": {"export": {"enabled": true, "formats": ["json"]}}}'
```

### Multimodal ablation

```bash
# Remove visual features entirely
--param_overrides '{"multimodal_ablation": {"visual": "remove"}}'

# Replace text features with Gaussian noise
--param_overrides '{"multimodal_ablation": {"text": "noise", "text_noise_scale": 1.0}}'

# Remove ID embeddings
--param_overrides '{"multimodal_ablation": {"id": "remove"}}'
```

---

## Canonical Field Names

Using non-canonical names causes config-code divergence. Always use these exact keys:

| Concept | Canonical key | Do NOT use |
|---|---|---|
| Contrastive temperature | `temperature` | `tau`, `ssl_temperature` |
| Transformer/GNN layer count | `num_layers` | `n_layers`, `num_blocks` |
| Dropout probability | `dropout_rate` | `dropout`, `dropout_prob` |
| Attention heads | `num_attention_heads` | `n_heads` |
| MM graph conv layers | `num_mm_layers` | `n_mm_layers` |
| Latent/bottleneck dim | `latent_dim` | `dim_latent`, `latent_dimension` |
| Embedding L2 regularization | `embedding_weight_decay` | `l2_emb` |
| Alignment loss weight | `lambda_align` | `lamda` |
| SSL/contrastive loss weight | `ssl_weight` | `cl_weight`, `ssl_loss_weight` |

---

## How Output Filenames Are Generated

Output paths and filenames are generated from config values by `set_paths()` in `core/config.py`. Understanding this prevents confusion about where results land:

```
log_file:    [MODEL]-[DATASET]-[type.comment]-[TIMESTAMP].txt
result_file: [MODEL]-[DATASET]-[type.comment].csv
```

- `type` comes from `config["type"]`: CLI default is `"test"` (from `--type` flag); YAML default is `"experiment"`. When invoked from the command line without `--type`, output files use `"test"` as the type tag. Set this explicitly to avoid overwriting results across runs.
- `comment` comes from `config["comment"]`: CLI default is `"test"` (from `--comment` flag); YAML default is `"default"`.
- Set these via `--type` and `--comment` CLI flags to organize multiple runs — e.g., `--type ablation --comment no_visual`.

### Recommendation export and optional diagnostics

The output group controls two families of artifacts:

1. `output.export`: final-test recommendation-list export for offline and device-side consumers that use NexusRec's internal id vocabulary.
2. Legacy diagnostics: internal-id or tensor payloads useful while debugging model behavior.

JSON writes one array of evaluated-user records:

```json
[{"user_id": 0, "items": [{"rank": 1, "item_id": 3, "score": 0.25955888628959656}, {"rank": 2, "item_id": 5, "score": 0.06581674516201019}, {"rank": 3, "item_id": 4, "score": -0.26425743103027344}]}]
```

Supported formats are `json`, `jsonl`, `csv`, and `tsv`; `json` is the default single-file grouped format, `jsonl` writes one grouped user record per line, and CSV/TSV are long tables with `user_id`, `rank`, `item_id`, and optional `score`. Each data file gets a sibling `.metadata.json` containing NexusRec internal id-space metadata, user/item counts, row grain, metrics, item/score semantics, provenance, and HPO lineage when present. It runs only during final test evaluation. If `eval_test_during_training: true`, normal training does not call the final-test export path.

Legacy diagnostics:

| Key | Artifact |
|---|---|
| `save_recommended_topk` | TSV of internal top-k item ids in `checkpoint_dir`; cannot be enabled together with `output.export.enabled` |
| `save_recommendation_scores` | Masked full-sort score payload in `checkpoint_dir` |
| `save_recommendation_score_components` | Model-defined score component payload; requires `full_sort_predict_components()` |
| `save_recommendation_artifact_by_epoch` | Adds epoch tags to score artifacts |
| `save_training_triplets` | Federated training triplet CSV |
| `save_training_triplet_scores` | Adds model-specific triplet scores; requires compatible `forward()` output |

Normal result CSVs are single-row artifacts. HPO CSVs are trial histories.

---

## Agent Notes

- `config.get("key", default)` is **forbidden**. Use `config["key"]`. If the key might be absent, add it to `configs/overall.yaml` with its default.
- All numeric literals that affect training, sampling, or evaluation outcomes belong in YAML. Python source files must not contain hardcoded experiment parameters.
- The `_FLATTEN_GROUPS` list in `core/config.py` controls which YAML sections are flattened. If you add a new top-level YAML group and expect flat access, add its name to `_FLATTEN_GROUPS`.
- `output.export` is defined under `output:` in YAML but read as `config["export"]` at runtime. Do not introduce parallel names for the same contract.
