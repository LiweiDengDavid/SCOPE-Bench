# NexusRec Interface Reference

This reference documents the stable surfaces that researchers and model authors are expected to call or implement. Anything not listed here should be treated as internal unless a tutorial explicitly uses it.

---

## CLI Surface

The command-line entrypoint is `main.py`. It parses CLI arguments, builds a `config_dict`, and always calls `core.training.quick_start()`.

```bash
python main.py --model VBPR --dataset Beauty
python main.py --model VBPR --dataset Beauty --param_overrides '{"learning_rate": 0.01}'
python main.py --model VBPR --dataset Beauty --smart_hpo --strategy bayesian --hpo_budget 50
```

Important flags:

| Flag | Runtime effect |
|---|---|
| `--model`, `--dataset` | Select model config and dataset directory. Model name must match class/config name exactly. |
| `--gpu_id` | Select CUDA device index after visibility is resolved. If CUDA is unavailable, CPU is used. |
| `--type`, `--comment` | Set output path tags. CLI defaults are both `test`, overriding `overall.yaml` defaults. |
| `--max_epochs` | Writes top-level runtime `max_epochs`. |
| `--early_stopping` | Writes top-level runtime `early_stopping` as a boolean override. |
| `--hyper_parameters` | JSON string overriding the HPO hyperparameter list; pass `'[]'` to disable configured HPO dimensions. |
| `--param_overrides` | JSON object deep-merged last. Can override flat keys or nested groups. |
| `--smart_hpo` | Routes `quick_start()` into HPO instead of standard training. |
| `--strategy` | HPO strategy: `random`, `grid`, `bayesian`, or `tpe`. |
| `--hpo_budget` | Writes `optimization.budget`. |
| `--hpo_parallel` | Enables single-node trial-level HPO sharding. |
| `--hpo_gpus` | Comma-separated physical GPU ids for parallel HPO. |
| `--hpo_parallel_dry_run` | Prints shard commands and returns the shard plan without launching child processes. |
| `--no-resume` | Starts HPO fresh instead of resuming from prior Optuna DB or CSV history. |
| `--resume-training` | Enables ordinary training resume from the full-state checkpoint `resume_state.pth`. This is separate from HPO `--no-resume`. |

Runtime routing is controlled by top-level `smart_hpo`. HPO settings such as strategy, budget, parallelism, and final training live under `optimization`.

---

## Programmatic Training APIs

### `quick_start()`

Defined in `core/training/interface.py`.

```python
from core.training import quick_start

result = quick_start(
    model="VBPR",
    dataset="Beauty",
    config_dict={"training": {"max_epochs": 20}},
    save_model=False,
    resume=True,
    verbose=False,
)
```

Standard training returns:

```python
{
    "test_result": {"Recall@10": 0.0, "NDCG@10": 0.0, ...},
    "valid_result": {"Recall@10": 0.0, "NDCG@10": 0.0, ...},
    "config": config,
}
```

HPO returns the result dictionary from `run_unified_hpo()` or `run_parallel_hpo()`. When `optimization.final_train.enabled` is true, `quick_start()` appends one additional key:

```python
{
    ...,
    "final_train": {
        "test_result": {...},
        "valid_result": {...},
        "config": config,
    },
}
```

The nested `final_train` value is the result of a second normal training run built from the HPO `best_configuration`.

### `run_training()`

Defined in `core/training/interface.py`.

```python
from core.training.interface import run_training

result = run_training(
    model="BPR",
    dataset="Beauty",
    config_dict={"training": {"max_epochs": 5}},
    save_model=False,
)
```

Use this only when you explicitly want normal training and want to bypass the `smart_hpo` routing decision.

---

## HPO APIs

### `run_unified_hpo()`

Defined in `core/hpo/engine.py`.

```python
from core.hpo.engine import run_unified_hpo

result = run_unified_hpo(
    model_name="VBPR",
    dataset_name="Beauty",
    base_config={"type": "hpo", "comment": "demo"},
    strategy="bayesian",
    target_metric="NDCG@10",
    max_trials=50,
    resume=True,
)
```

Typical return keys:

```python
{
    "best_configuration": {...},
    "best_score": 0.0,
    "best_metrics": {"valid_metrics": {...}, "test_metrics": {...}},
    "best_trial_num": 1,
    "total_trials": 50,
    "trial_history": [...],
    "csv_file": "outputs/hyper_search/...",
}
```

Optuna-backed strategies also include `target_metric` and `strategy` in the returned dictionary. Serial enumeration can return early when every grid/random combination is already complete; that resume path returns the best fields, `target_metric`, `strategy`, `csv_file`, and `total_trials: 0`, without new `trial_history`. In Optuna paths, `total_trials` is the number of finished study trials; in enumeration paths that execute new trials, it is the number executed in the current call.

Trial history CSV locations differ by strategy: Bayesian/TPE/random write under `outputs/hyper_search/{MODEL}/{DATASET}/[MODEL]-[DATASET]-[strategy.TYPE.COMMENT].csv`, while serial grid search writes under `outputs/results/{MODEL}/{DATASET}/{TYPE}/[MODEL]-[DATASET]-[experiment.TYPE.COMMENT].csv`.

Trial CSVs include `target_metric` for the optimized metric key and `target_score` for that trial's value. Metric columns such as `valid_metrics` and `test_metrics` are dict-valued columns serialized by pandas; `test_metrics` is commonly `{}` unless HPO final test evaluation is enabled.

CLI default strategy is `bayesian`; the lower-level function default is `grid`, so pass `strategy` explicitly in scripts.

During HPO trials, `output.export.enabled` is forced to false so trial search does not emit final recommendation artifacts. Use `optimization.final_train.enabled` when the best configuration should produce a formal model, checkpoint, result CSV, and optional recommendation export.

### Parallel HPO Return Shape

`quick_start()` calls `core.hpo.parallel.run_parallel_hpo()` when both `smart_hpo` and `optimization.parallel` are true.

Dry-run returns:

```python
{
    "parallel": True,
    "dry_run": True,
    "total_trials": 60,
    "shards": [
        {
            "index": 0,
            "gpu_id": "0",
            "budget": 20,
            "comment": "demo_shard00",
            "log_file": "...stdout.log",
            "hpo_dir": ".../parallel/type/demo_shard00",
            "checkpoint_dir": ".../parallel/type/demo_shard00",
        }
    ],
}
```

Completed parallel HPO returns the merged CSV path, `best_configuration`, best score/trial metadata, successful trial count, and the same shard descriptors. The merged CSV adds `shard_index`, `shard_comment`, and `shard_csv` columns to the child trial rows.

Parallel HPO is single-node trial-level parallelism. It supports `bayesian` and `tpe`; `grid` and `random` use the serial HPO path. Child shards disable `optimization.final_train`, and the parent may run final training once after the merged best row is selected. Dry-run returns only the shard plan and never runs final training.

---

## Configuration API

`ConfigManager(model, dataset, config_dict)` is the runtime config object. Merge order is:

```text
configs/overall.yaml
  -> configs/models/{MODEL}.yaml
  -> configs/datasets/{DATASET}.yaml, if present
  -> configs/datasets/{DATASET}.yaml model_overrides.{MODEL}, if present
  -> config_dict / CLI overrides
```

Groups flattened to top-level runtime keys:

```text
training, evaluation, output, experiment, sequential, sampling, resources
```

Groups intentionally kept nested include `optimization`, `features`, `federated`, `bayesian`, `tpe`, and `multimodal_ablation`.

Nested dictionaries inside a flattened group are promoted as dictionaries. For example, YAML `output.export` becomes runtime `config["export"]`, while YAML `optimization.final_train` remains `config["optimization"]["final_train"]`.

Use direct indexing:

```python
learning_rate = config["learning_rate"]
strategy = config["optimization"]["strategy"]
vision_file = config["features"]["vision_feature_file"]
export_enabled = config["export"]["enabled"]
final_train_enabled = config["optimization"]["final_train"]["enabled"]
```

Do not use `.get()`, `.update()`, `.pop()`, or `.setdefault()` on `ConfigManager`; the class intentionally does not expose those dict helpers.

---

## Model Registry API

Defined in `core/model_registry.py`.

| Function | Contract |
|---|---|
| `load_model_profile(model_name)` | Reads required flags from `configs/models/{Model}.yaml`. |
| `infer_paradigm(profile)` | Maps flags to one canonical paradigm root. |
| `get_model_source_path(model_name)` | Recursively resolves exactly one lowercase source file under the inferred root. |
| `get_model(model_name)` | Imports the source module and returns the class whose name exactly matches `model_name`. |
| `get_trainer(model_name, is_federated, is_sequential)` | Returns `SequentialTrainer`, a model-specific `{Model}Trainer`, `FederatedTrainer`, or `TrainerBase`. |

Paradigm routing:

| Flags | Root |
|---|---|
| `is_sequential: true`, `is_multimodal_model: false` | `models/sequential/id/` |
| `is_sequential: true`, `is_multimodal_model: true` | `models/sequential/multimodal/` |
| `is_federated: true`, `is_multimodal_model: false` | `models/federated/id/` |
| `is_federated: true`, `is_multimodal_model: true` | `models/federated/multimodal/` |
| neither sequential nor federated, multimodal false | `models/centralized/id/` |
| neither sequential nor federated, multimodal true | `models/centralized/multimodal/` |

---

## Model Implementation Contract

All non-sequential models inherit `RecommenderBase`; sequential models inherit `SequentialRecommender`.

Required methods and practical evaluation hooks:

| Method | Expected behavior |
|---|---|
| `forward(...)` | Core scoring/encoding pass. Signature is model-specific. |
| `calculate_loss(interaction)` | Returns a scalar loss tensor. Centralized training can sum tuple losses, but federated and sequential trainers expect a scalar tensor. |
| `full_sort_predict(interaction, *args, **kwargs)` | Provided by base classes, but should be overridden for performance when needed. Centralized/federated scores are `[batch_size, n_items]` and item `0` is a normal item. Sequential scores may be `[batch_size, n_items+1]` when index `0` is PAD; only `SequentialEvaluator` suppresses PAD `0`. |

Optional hooks:

| Hook | Purpose |
|---|---|
| `pre_epoch_processing()` / `post_epoch_processing()` | Per-epoch model maintenance. |
| `get_optimizer_params()` | Custom optimizer parameter groups. |
| `get_regularization_loss()` | Additional model regularization. |
| `full_sort_predict_components()` | Required only when `save_recommendation_score_components` is true. |
| `finalize_training()` | Called after training when present. |

Centralized multimodal models should call `self.setup_multimodal_features(config)` in `__init__`. Federated multimodal models get features from `FederatedTrainer`; calling the base setup method on a federated model instance is a no-op.

Federated model hooks:

| Hook | Contract |
|---|---|
| `get_shared_parameters()` | Dict of state-dict names to tensors aggregated across clients. |
| `get_personal_parameters()` | Dict of state-dict names to tensors retained per client. |
| `get_server_grad_param_names()` | Optional list of shared names using server-side delta aggregation. Requires explicit `server_learning_rate`. |

Shared plus personal parameters must cover all trainable parameters. When at least one valid shared parameter name is resolved, `FederatedTrainer.__init__` raises `ValueError` if any parameter is orphaned, since an orphan would be frozen at initialization. If the shared list is empty or all names are filtered out, aggregation fails fast because the server aggregator requires a non-empty shared parameter set.

---

## Data And Loader Interfaces

`RecDataset` expects `datasets/{DATASET}/inter.csv` by default, with columns from `USER_ID_FIELD`, `ITEM_ID_FIELD`, and `inter_splitting_label`. By convention:

| `split_label` | Meaning |
|---:|---|
| `0` | train |
| `1` | validation |
| `2` | test |

For centralized and federated models, `create_loaders(config, train_dataset, valid_dataset, test_dataset)` returns train/valid/test loader objects. Sequential models bypass `core/data/pipeline.py` and are built through `core/sequential/integration.auto_setup()`.

Common loader contracts:

| Loader | Iteration output |
|---|---|
| `TrainDataLoader` | Tensor shaped `[2, B]` when negative sampling is disabled; `[2 + K, B]` when enabled, where `K=config["num_negatives"]`. `K > 1` requires the model to declare `supports_multi_negatives=True`. |
| `EvalDataLoader` | `[batch_users, batch_mask_matrix]`; exposes `get_eval_items()`, `get_eval_len_list()`, `get_eval_users()`. |
| `FederatedDataLoader` | Iterates `(user_id, per_user_loader)` and exposes `loaders`, `user_set`, plus eval helper methods. |
| `SequentialDataLoader` | Dict batches with `user_ids`, `item_seqs`, `targets`, `seq_lens`, `seq_masks`, optional `neg_items`, and optional `targets_list` / `num_targets`. |

`test_history_mask` controls test-time masking:

| Value | Masked items |
|---|---|
| `train_only` | Training history only. |
| `train_valid` | Training plus validation history. |

The current evaluation loader does not mask the current target item when the same
item appears across splits for the same user; this preserves reachable
cross-split repurchase targets. Test-time `Novelty` and item-bucket popularity
use pure train-split popularity, independent of the test history-mask mode.
`EvalDataLoader` is not a shuffled loader; shuffled evaluation fails fast.

Multimodal feature artifacts are dense `.npy` files under the dataset directory.
When a model is multimodal and not `end2end`, each loaded feature file must have
exactly one row per item in the catalog and its second dimension must match
`config["features"]["visual_dim"]` or `config["features"]["text_dim"]`.

Recommendation export uses the same NexusRec zero-based integer ids as
evaluation. `user_id` is an evaluated user index; `item_id` is a shared internal
item index after any model-local offset has been removed. Sequential models use
PAD index `0` internally, but export maps real items back to the shared
`0..item_count-1` item space.

---

## Evaluation Interface

`TopKEvaluator(config).evaluate(batch_matrix_list, eval_data, is_test=False, idx=0)` returns a metric dictionary such as:

```python
{"Recall@10": 0.123456, "NDCG@10": 0.078901}
```

Supported metric names:

```text
Recall, Precision, NDCG, MAP, MRR, Hit, Diversity, Novelty, Coverage
```

`SequentialEvaluator` uses the same metric kernel for ranking metrics, suppresses PAD item `0`, applies seen-item filtering when configured, and supports multi-target sequential evaluation. `SequentialTrainer` passes train-split item popularity frequencies, so `Novelty` uses the same train-popularity basis as the common evaluator.

---

## Result Artifacts

Normal training writes one canonical result row:

```text
outputs/results/{MODEL}/{DATASET}/{type}/[{MODEL}]-[{DATASET}]-[{type}.{comment}].csv
```

The file is overwritten in place as a single-row artifact. Empty or multi-row result CSVs are rejected by `Result` from `core.utils.result`; metric columns must parse to finite floats.

Checkpoints are saved as `best_model.pth` under `checkpoint_dir` when `save_model` is true. Federated checkpoints also include per-client personal state.

### Recommendation Export

`output.export` is the stable recommendation-list export contract. It is disabled by default and writes only during final test evaluation (`evaluate_final_test(..., write_export=True)`), not during validation, training-time test evaluation, or HPO trials.

YAML shape:

```yaml
output:
  export:
    enabled: true
    formats: ["json"]         # any non-empty unique subset of json/jsonl/csv/tsv
    include_scores: true
    split: "test"             # only test is supported
    topk: null                # null means the full evaluated top-k width
    path: ""                  # empty means config["paths"]["save"]
```

JSON writes one array of user records:

```json
[{"user_id": 0, "items": [{"rank": 1, "item_id": 3, "score": 0.25955888628959656}, {"rank": 2, "item_id": 5, "score": 0.06581674516201019}, {"rank": 3, "item_id": 4, "score": -0.26425743103027344}]}]
```

For CSV and TSV, each recommendation is one row. Columns are `user_id`, `rank`,
`item_id`, and optional `score`. The table format is intentionally long-form so
offline jobs can join, filter, and aggregate without parsing list cells.

Each data file has a sibling `.<format>.metadata.json` containing
`artifact_type`, model/dataset/run tags, split, top-k,
`id_space=nexusrec_internal`, user/item counts, row grain, user row
count, recommendation count, list ordering, score ordering/comparability,
`rank_base`, metrics, provenance, `model_item_id_offset`, and any HPO lineage.
Sequential export applies `model_item_id_offset=1` so model-side PAD index `0`
is never exported and real item ids map back to NexusRec's shared zero-based
internal item ids.

### Loading Recommendation Export

Use `Recommendation` from `core.utils.recommendation` when offline jobs or
device-side reranking preparation need to reload an export artifact. The loader
validates the sibling metadata and returns the same user-record structure for
all supported formats:

```python
from core.utils.recommendation import Recommendation

records, metadata = Recommendation.load(path)
rows = Recommendation.to_rows(records, include_scores=metadata["include_scores"])
```

`records` is a list of `{"user_id": ..., "items": [...]}` dictionaries. For
CSV/TSV inputs, `Recommendation.load()` reconstructs this shape from the long
table before validation.

`output.export.enabled=true` is mutually exclusive with legacy `save_recommended_topk=true`. The export contract fails fast on unsupported/duplicate formats, non-test split, invalid `topk`, duplicate user rows, invalid user/item id ranges, duplicate items within one user list, score-shape mismatches, and NaN/Inf scores or metrics.

### Legacy diagnostics

These artifacts remain useful for debugging model internals, but they are not the recommendation-list export contract:

| Config key | Output |
|---|---|
| `save_recommended_topk` | TSV of internal top-k item ids in `checkpoint_dir`; mutually exclusive with `output.export.enabled`. |
| `save_recommendation_scores` | Masked full-sort score tensor payload. |
| `save_recommendation_score_components` | Model-defined score component tensor payload. |
| `save_training_triplets` | Federated training triplet CSV. |

---

## Benchmark And Significance Interfaces

Structured benchmark manifests use `configs/examples/benchmark.yaml` style fields:

```yaml
experiments:
  - name: "seed-sweep"
    models: ["BPR", "LightGCN"]
    datasets: ["Beauty"]
    seeds: [2024, 2025]
    mode: "train"
    type: "benchmark"
    comment: "seed_sweep"
    overrides:
      training:
        max_epochs: 20
```

Commands:

```bash
python scripts/run_benchmark.py --spec configs/examples/benchmark.yaml --dry-run
python scripts/run_benchmark.py --spec configs/examples/benchmark.yaml
python scripts/run_benchmark.py --spec configs/examples/benchmark.yaml --summarize
```

Ledgers are written under `outputs/benchmarks/{manifest_name}/{manifest_hash12}/` as `ledger.jsonl`, `ledger.csv`, and `plan.json`.

Structured benchmark `mode: hpo` accepts `hpo.strategy`, `hpo.budget`,
`hpo.resume`, and `hpo.verbose`. Summary support reads HPO histories from
`outputs/hyper_search`, so benchmark HPO manifests support `bayesian`, `tpe`,
and `random`; `grid` is rejected at plan time.

With `--summarize`, the CLI executes or resumes incomplete planned runs first, then writes `summary_runs.csv`, `summary_groups.csv`, `summary.md`, and, when a baseline is configured, `summary_significance.csv` in the same benchmark output directory. `summary_runs.csv` keeps failed or incomplete runs visible with empty metric cells.

Standalone paired significance:

```bash
python scripts/significance_test.py \
  --baseline "outputs/results/BPR/Beauty/stability/*.csv" \
  --candidate "outputs/results/NCF/Beauty/stability/*.csv" \
  --metrics NDCG@10 Recall@10 \
  --pair-field comment \
  --test wilcoxon
```

Standalone `--pair-field` choices are `comment` and `stem`. Benchmark summary significance supports `seed` and `comment`; prefer `seed` for multi-seed manifests, and use `comment` only when it uniquely identifies paired runs. (`output_comment`/`run_id` encode the model, so they can never pair a baseline against a candidate and are not offered.)

Benchmark significance compares candidate models against the baseline within the same `(experiment_name, dataset, mode, type)` group. Groups without the baseline are skipped unless the baseline is absent everywhere, which is treated as an error.
