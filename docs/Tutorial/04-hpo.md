# Hyperparameter Optimization (HPO)

NexusRec provides a unified HPO interface supporting four strategies. All strategies use the same `parameter_space` format defined in the model YAML. HPO can optionally launch a second, formal training run with the selected best configuration through `optimization.final_train`.

User-facing runs always enter through `main.py -> quick_start()`. When `--smart_hpo` is enabled, `quick_start()` routes into `_run_hpo_flow()` or `run_parallel_hpo()`, then optionally calls final training.

---

## Quick Start

```bash
# Bayesian optimization (recommended, default strategy)
python main.py --model VBPR --dataset Beauty --smart_hpo --strategy bayesian

# TPE (Optuna default; similar to Bayesian)
python main.py --model VBPR --dataset Beauty --smart_hpo --strategy tpe

# Random search with budget limit
python main.py --model VBPR --dataset Beauty --smart_hpo --strategy random --hpo_budget 50

# Grid search (exhaustive, no budget needed)
python main.py --model VBPR --dataset Beauty --smart_hpo --strategy grid

# Verbose output (shows per-trial training progress)
python main.py --model VBPR --dataset Beauty --smart_hpo --strategy bayesian --verbose

# Fresh start (do not resume prior HPO state)
python main.py --model VBPR --dataset Beauty --smart_hpo --strategy bayesian --no-resume

# HPO followed by one formal final_train run using the best configuration
python main.py --model VBPR --dataset Beauty --smart_hpo --strategy bayesian \
  --param_overrides '{"optimization": {"final_train": {"enabled": true}}}'
```

---

## Single-Node Parallel HPO

HPO trials are independent, so on a machine with multiple GPUs you can split
the total budget across GPUs. NexusRec keeps this simple: each GPU runs a
normal HPO shard, and the shard CSVs are merged when all shards finish.

This is **not** distributed model training. It is trial-level parallelism on one
node, which is usually the best fit for research experiments.

Parallel HPO is intentionally limited to Optuna-backed `bayesian` and `tpe`
searches. Each shard shares one Optuna study, uses a shard-specific sampler
seed, and enables Optuna's constant-liar protection for TPE so concurrent
workers do not waste budget on identical or near-identical suggestions.

```bash
python main.py \
  --model VBPR \
  --dataset Beauty \
  --smart_hpo \
  --strategy bayesian \
  --hpo_budget 60 \
  --hpo_parallel \
  --hpo_gpus 0,1,2 \
  --type VBPR_Beauty_HPO_parallel \
  --comment local_test
```

This launches three shards:

| GPU | Trials |
|---:|---:|
| 0 | 20 |
| 1 | 20 |
| 2 | 20 |

If the budget does not divide evenly, earlier shards get one extra trial. For
example, 5 trials on GPUs `0,2` becomes 3 trials on GPU 0 and 2 trials on GPU 2.

### Recommended first check

Before launching a long run, inspect the shard plan:

```bash
python main.py \
  --model VBPR \
  --dataset Beauty \
  --smart_hpo \
  --strategy bayesian \
  --hpo_budget 60 \
  --hpo_parallel \
  --hpo_gpus 0,1,2 \
  --hpo_parallel_dry_run \
  --type VBPR_Beauty_HPO_parallel \
  --comment local_test
```

Dry-run output is intentionally short:

```text
Parallel HPO: model=VBPR dataset=Beauty strategy=bayesian target=NDCG@10 total_trials=60 shards=3
[shard 00] gpu=0 trials=20 comment=local_test_shard00
  log: outputs/logs/VBPR/Beauty/parallel/VBPR_Beauty_HPO_parallel/local_test_shard00.stdout.log
  hpo: outputs/hyper_search/VBPR/Beauty/parallel/VBPR_Beauty_HPO_parallel/local_test_shard00
```

Programmatic dry-run returns the same shard plan as a dictionary:

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
            "comment": "local_test_shard00",
            "log_file": "...stdout.log",
            "hpo_dir": ".../parallel/VBPR_Beauty_HPO_parallel/local_test_shard00",
            "checkpoint_dir": ".../parallel/VBPR_Beauty_HPO_parallel/local_test_shard00",
        }
    ],
}
```

### GPU selection

Use `--hpo_gpus` to list the physical GPU IDs available to this process:

```bash
--hpo_gpus 0,1,2
```

If `--hpo_gpus` is omitted, NexusRec uses all CUDA GPUs visible to the current
process. This is useful when an external system has already set
`CUDA_VISIBLE_DEVICES`.

### Output layout

Each shard writes its own child trial-history CSV and checkpoint directory:

```text
outputs/hyper_search/{MODEL}/{DATASET}/parallel/{TYPE}/{COMMENT}_shard00/
outputs/checkpoints/{MODEL}/{DATASET}/parallel/{TYPE}/{COMMENT}_shard00/
```

The merged trial history is written to the normal HPO directory:

```text
outputs/hyper_search/{MODEL}/{DATASET}/
    [MODEL]-[DATASET]-[strategy.TYPE.COMMENT.parallel].csv
```

The merged CSV preserves the normal HPO columns and adds `shard_index`,
`shard_comment`, and `shard_csv` so the selected trial can be traced back to
the exact child process.

Per-shard CSV and checkpoint directories stop workers from overwriting each
other's outputs. The Optuna study itself is shared: all shards write one
`optuna_journal.log`, and `JournalStorage` file locking keeps those concurrent
writes safe.

### When to use it

Use parallel HPO when:

- you have multiple GPUs on the same node
- each trial fits on one GPU
- the search space is small or medium
- faster wall-clock time matters more than one fully sequential Bayesian study

Avoid it when:

- you only have one GPU
- the model itself needs multi-GPU training
- you need exact exhaustive grid-search ordering

Parallel HPO currently supports `bayesian` and `tpe`. `random` and `grid`
searches stay on the serial path because their CSV-based bookkeeping and exact
trial ordering are easier to reason about there.

---

## Strategy Selection

| Strategy | Use when | Resume mechanism |
|---|---|---|
| `bayesian` | ≥ 20 trials, continuous parameters, exploration-exploitation balance | Optuna journal storage (automatic) |
| `tpe` | Same as Bayesian; TPE is Optuna's default sampler | Optuna journal storage (automatic) |
| `random` | Cheap baseline comparison | CSV of completed trials |
| `grid` | Small discrete parameter spaces, need exhaustive coverage | CSV of completed trials |

**CLI default** (when `--strategy` is omitted): `bayesian`

**Lower-level API default** (`run_unified_hpo()` function signature): `grid`. Always pass `strategy` explicitly when calling the API directly.

---

## Configuring the Search Space

The search space is defined in the model's YAML under `parameter_space`. The `hyper_parameters` key lists which parameters to search (subset of `parameter_space` keys is allowed).

### Full example

```yaml
# configs/models/VBPR.yaml

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
    values: [32, 64, 128, 256, 512]
  dropout_rate:
    type: "uniform"
    low: 0.0
    high: 0.5
  num_layers:
    type: "int"
    low: 1
    high: 5
```

### Parameter type reference

| `type` | Fields | Notes |
|---|---|---|
| `"choice"` | `values: [v1, v2, ...]` | Discrete options; works with all strategies |
| `"uniform"` | `low`, `high` | Continuous; random/Bayesian/TPE sample the range |
| `"loguniform"` | `low`, `high` | Log-scale; recommended for learning_rate, weight_decay |
| `"logscale"` | `low`, `high` | Alias for log-scale continuous sampling |
| `"int"` | `low`, `high` | Integer range inclusive; Bayesian/TPE use integer sampling |
| `"logint"` | `low`, `high` | Integer range sampled on a log scale |

Grid search is discrete by design. It uses `grid_values` when present, otherwise `choice.values`, otherwise the resolved default value for that parameter. It does not discretize `uniform`, `loguniform`, `logscale`, `int`, or `logint` ranges automatically.

### Setting the budget

The budget is read from `optimization.budget`:

```yaml
# configs/overall.yaml — global default (already present, default 1000):
optimization:
  budget: 1000

# configs/models/MyModel.yaml — model-specific override (takes priority):
optimization:
  budget: 50
```

Or via CLI (highest priority): `--hpo_budget 50`

---

## Resume Behavior

| Strategy | Default behavior with `--smart_hpo` | `--no-resume` effect |
|---|---|---|
| `bayesian` / `tpe` | Resume from the Optuna journal study if it exists | Serial HPO uses a fresh study identity; parallel HPO removes the shared journal before launching shards |
| `grid` | Skip already-completed combinations (from CSV) | Re-run all combinations |
| `random` | Reads existing CSV rows and runs only the remaining requested budget | Re-run the requested budget from scratch |

Optuna studies are stored in `outputs/hyper_search/{MODEL}/{DATASET}/optuna_journal.log`. Do not delete this file if you want to resume.
The study identity includes the target metric, so changing `valid_metric` starts a separate Bayesian/TPE study instead of resuming trials optimized for the old metric.

For parallel HPO, all shards share a single Optuna journal study, so trials are
deduplicated across processes. Re-running the same command resumes every shard
unless `--no-resume` is passed.

---

## Final Training With The Best Configuration

HPO search runs are not the same thing as the final model artifact. Search trials explore candidates, record metrics, and may save trial checkpoints when configured. `optimization.final_train` starts a second ordinary training run after HPO chooses the best row:

```yaml
optimization:
  final_train:
    enabled: true
    overrides:
      type: "final_train"
      comment: "best_config"
      save_model: true
      resume_training: false
      eval_final_test: true
      eval_test_during_training: false
```

The final-train config is constructed as:

```text
input config
  + expanded HPO best_configuration
  + optimization.final_train.overrides
  + smart_hpo=false
  + optimization.parallel=false
  + optimization.final_train.enabled=false
  + hpo_lineage metadata
```

This means final training reloads config, data, logging, paths, model, and trainer as a normal run. It does not simply load the best trial checkpoint. The resulting `hpo_lineage` records the HPO source CSV, strategy, target metric, best trial number, best score, and selected configuration.

Pair final training with recommendation export when downstream offline jobs or device-side reranking need top-k recommendations:

```bash
python main.py --model VBPR --dataset Beauty \
  --smart_hpo --strategy bayesian --hpo_budget 50 \
  --param_overrides '{"optimization": {"final_train": {"enabled": true}}, "output": {"export": {"enabled": true, "formats": ["json"]}}}'
```

HPO trials always disable `output.export`, so this command writes recommendation files only for the formal final-train run.

---

## Output Files

After an HPO run, results are written to strategy-specific locations:

```text
outputs/hyper_search/{MODEL}/{DATASET}/
    [MODEL]-[DATASET]-[bayesian.TYPE.COMMENT].csv    ← Bayesian trial history
    [MODEL]-[DATASET]-[tpe.TYPE.COMMENT].csv         ← TPE trial history
    [MODEL]-[DATASET]-[random.TYPE.COMMENT].csv      ← random-search trial history
    optuna_journal.log                               ← Bayesian/TPE JournalStorage when resume=true

outputs/results/{MODEL}/{DATASET}/{TYPE}/
    [MODEL]-[DATASET]-[experiment.TYPE.COMMENT].csv  ← serial grid-search trial history

outputs/checkpoints/{MODEL}/{DATASET}/
    hpo/{strategy}/{type}/{comment}/best_model.pth    ← best-trial checkpoint when HPO save_model=true

outputs/results/{MODEL}/{DATASET}/final_train/
    [MODEL]-[DATASET]-[final_train.best_config].csv    ← automatic final_train result, when enabled
    [MODEL]-[DATASET]-[final_train.best_config.seedX.idx0.top50.recommendations].json
    [MODEL]-[DATASET]-[final_train.best_config.seedX.idx0.top50.recommendations].json.metadata.json
```

Parallel HPO writes child histories under `outputs/hyper_search/{MODEL}/{DATASET}/parallel/{TYPE}/{COMMENT}_shardNN/` and writes the merged trial history back to `outputs/hyper_search/{MODEL}/{DATASET}/[MODEL]-[DATASET]-[strategy.TYPE.COMMENT.parallel].csv`.

### Trial history CSV columns

| Column | Description |
|---|---|
| `trial_num` | Trial index (1-based) |
| `strategy` | HPO strategy used |
| `target_metric` | Metric key optimized by this HPO run, usually resolved from `valid_metric` |
| `duration` | Trial wall-clock time in seconds |
| `status` | `"completed"`, `"failed"`, or `"pruned"` |
| `target_score` | `valid_metric` value for this trial |
| `{param_name}` | One column per searched hyperparameter |
| `valid_metrics` | Dict-valued validation metrics column serialized by pandas |
| `test_metrics` | Dict-valued test metrics column serialized by pandas when `optimization.eval_final_test` is enabled |

Keep `optimization.eval_final_test: true` for benchmark HPO runs that must produce test metrics directly in the trial history.
The formal final_train result is a normal single-row result CSV, not a trial-history CSV.

---

## Return Value (programmatic use)

When using `run_unified_hpo()` directly:

```python
from core.hpo.engine import run_unified_hpo

# Strategy and max_trials kwargs take precedence over base_config values.
# base_config sets other experiment parameters (learning_rate, dataset config, etc.).
result = run_unified_hpo(
    model_name="VBPR",
    dataset_name="Beauty",
    base_config={"valid_metric": "NDCG@10"},  # other config overrides go here
    strategy="bayesian",      # overrides base_config["optimization"]["strategy"] if set
    target_metric="NDCG@10",  # overrides base_config["valid_metric"] if set
    max_trials=50,            # overrides optimization.budget from base_config if set
    resume=True,
    verbose=False,
)

result["best_configuration"]    # Dict of best hyperparameter values
result["best_score"]            # float — best valid_metric value
result["best_metrics"]          # Dict with "valid_metrics" and "test_metrics"; test_metrics may be {}
result["best_trial_num"]        # int — which trial was best
result["total_trials"]          # int — Optuna finished trials, or enumeration trials run in this call
result.get("trial_history")     # List of dicts when the path materializes trial rows
result.get("csv_file")          # str path when a CSV is written in this call
# quick_start(), not run_unified_hpo(), appends result["final_train"] when
# optimization.final_train.enabled=true.
```

Enumeration resume can return early when all combinations have already completed. In that case the result contains the best fields, `target_metric`, `strategy`, `csv_file`, and `total_trials: 0`, but no new `trial_history`.

---

## Data Caching

HPO reuses the same data loaders across all trials. Data is loaded once when the first trial starts and then shared. This avoids the overhead of re-reading and re-splitting the dataset for each trial, which would otherwise dominate runtime for small models.

---

## HPO Log File

Serial HPO uses the normal training log setup. With `--verbose`, trial progress
is written to the active run log:

```
outputs/logs/{MODEL}/{DATASET}/
    [MODEL]-[DATASET]-[hpo.COMMENT]-[TIMESTAMP].txt
```

Without `--verbose`, HPO keeps the log compact. Parallel HPO writes one stdout
log per shard under `outputs/logs/{MODEL}/{DATASET}/parallel/{TYPE}/`.

---

## Agent Notes

- `hyper_parameters` controls which parameters the HPO engine actually samples. A parameter in `parameter_space` but NOT in `hyper_parameters` is ignored (its default value is used).
- A parameter in `hyper_parameters` but NOT read by `config["name"]` in the model code means the HPO is searching a dimension that has no effect. Always verify the round-trip: YAML `hyper_parameters` entry → `config["param"]` read in model code.
- Grid search generates all Cartesian product combinations from explicit discrete values. Use `grid_values` or `choice.values` for every parameter you want grid search to vary; range-only parameters fall back to their resolved default value.
- Bayesian/TPE require `optuna` to be installed (`pip install optuna`).
- HPO trials disable `output.export`; enable `optimization.final_train` for the single formal model run that should produce result CSVs, checkpoints, and recommendation-list exports.
