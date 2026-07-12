# Quick Start

Run your first NexusRec experiment in five minutes.

---

## Prerequisites

```bash
pip install -r requirements.txt
# PyTorch requires manual installation matching your CUDA version
# See: https://pytorch.org/get-started/locally/
```

---

## Dataset Structure

Every dataset lives under `datasets/{DATASET}/`. The minimum required file is the interaction CSV.

### `datasets/{DATASET}/inter.csv`

| Column | Type | Description |
|---|---|---|
| `userID` | int | User identifier (non-negative integer; not auto-remapped internally) |
| `itemID` | int | Item identifier (non-negative integer; row `i` in feature files must correspond to item ID `i`) |
| `split_label` | int | 0 = train, 1 = validation, 2 = test |

Minimal example:
```
userID,itemID,split_label
1,101,0
1,102,0
1,103,1
1,104,2
2,201,0
...
```

### Optional Feature Files

Multimodal models (VBPR, BM3, FREEDOM, etc.) additionally require:

| File | Shape | Description |
|---|---|---|
| `datasets/{DATASET}/image_features.npy` | `[n_items, visual_dim]` | Pre-extracted visual embeddings |
| `datasets/{DATASET}/text_features.npy` | `[n_items, text_dim]` | Pre-extracted text embeddings |

Row `i` in each feature file corresponds to item ID `i`. The row count must equal the item catalog size (`max(itemID) + 1` after preprocessing), not merely the number of unique item IDs. Centralized multimodal models usually infer projection sizes from the loaded tensors; federated multimodal models read `features.text_dim` / `features.visual_dim`, so keep YAML and feature files consistent.

## Running an Experiment

### Centralized model (VBPR on Beauty)

```bash
python main.py --model VBPR --dataset Beauty
```

### Sequential model (GRU4Rec)

```bash
python main.py --model GRU4Rec --dataset Beauty
```

### Federated model (FedAvg)

```bash
python main.py --model FedAvg --dataset Beauty
```

### Override GPU

```bash
python main.py --model VBPR --dataset Beauty --gpu_id 1
```

### Override any config key at runtime

```bash
python main.py --model VBPR --dataset Beauty \
    --param_overrides '{"learning_rate": 0.01, "embedding_size": 128}'
```

The `--param_overrides` value is a JSON string. Any key that exists in `configs/overall.yaml` or the model YAML can be overridden here.

### Export final-test recommendations

Enable recommendation-list export with NexusRec internal ids:

```bash
python main.py --model VBPR --dataset Beauty \
    --param_overrides '{"output": {"export": {"enabled": true, "formats": ["json"]}}}'
```

The export is written only during final test evaluation. It does not run during validation, training-time test evaluation, or HPO trials.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `VBPR` | Model name (must match class name exactly) |
| `--dataset`, `-d` | `MovieLens` | Dataset name (must match directory under `datasets/`) |
| `--gpu_id`, `-g` | `0` | CUDA device ID |
| `--type`, `-t` | `test` | Experiment type tag (used in output filenames) |
| `--comment`, `-c` | `test` | Free-form comment tag (used in output filenames) |
| `--max_epochs` | (from YAML) | Override `max_epochs` |
| `--early_stopping` | (from YAML) | Override `early_stopping` with `true`/`false` |
| `--hyper_parameters` | (from YAML) | JSON list overriding HPO parameters, for example `'[]'` |
| `--param_overrides` | (none) | JSON string of arbitrary config key overrides |
| `--smart_hpo` | `false` | Enable HPO mode (see [04-hpo.md](04-hpo.md)) |
| `--strategy` | `bayesian` | HPO strategy: `random`, `grid`, `bayesian`, `tpe` |
| `--hpo_budget` | (from YAML) | Max HPO trials |
| `--hpo_parallel` | `false` | Split HPO trials across local GPUs on one node |
| `--hpo_gpus` | all visible CUDA GPUs | Comma-separated physical GPU IDs for parallel HPO |
| `--hpo_parallel_dry_run` | `false` | Print the parallel HPO shard plan without launching jobs |
| `--verbose` | `false` | Verbose logging during HPO |
| `--no-resume` | `false` | Start HPO from scratch (default: resume from prior run) |
| `--resume-training` | `false` | Resume ordinary training from `resume_state.pth`; separate from HPO `--no-resume` |

---

## Programmatic Quick Start

Python code should use the same user-facing entrypoint as the CLI:

```python
from core.training import quick_start

result = quick_start(
    model="VBPR",
    dataset="Beauty",
    config_dict={
        "training": {"max_epochs": 20},
        "type": "experiment",
        "comment": "api_demo",
    },
    save_model=False,
)

print(result["valid_result"])
print(result["test_result"])
```

For normal training, `quick_start()` returns a dict with `valid_result`, `test_result`, and `config`. In HPO mode, it returns the HPO result payload. If `optimization.final_train.enabled` is true, the HPO result additionally contains `result["final_train"]`, which is the result of the formal retraining run.

---

## Parallel HPO on One Node

If a node has multiple GPUs, split HPO trials across them:

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

This runs 60 trials as three 20-trial shards. Each shard gets its own stdout
log, HPO directory, and checkpoint directory, while all shards share one Optuna
study for Bayesian/TPE coordination. Run the same command with
`--hpo_parallel_dry_run` first if you want to check the plan before launching.

See [04-hpo.md](04-hpo.md) for details.

---

## Reading the Output

After training completes, look in:

```
outputs/
  logs/VBPR/Beauty/
      [VBPR]-[Beauty]-[test.test]-[TIMESTAMP].txt   ← full training log
  results/VBPR/Beauty/test/
      [VBPR]-[Beauty]-[test.test].csv               ← metrics CSV
      [VBPR]-[Beauty]-[test.test.seed2024.idx0.top50.recommendations].json
      [VBPR]-[Beauty]-[test.test.seed2024.idx0.top50.recommendations].json.metadata.json
  checkpoints/VBPR/Beauty/
      best_model.pth                                ← saved if save_model: true
```

For Bayesian/TPE HPO runs, `outputs/hyper_search/{MODEL}/{DATASET}/` contains the trial-history CSV and `optuna_journal.log`. The journal file is what lets those runs resume.

> **Note:** The `--type` flag defaults to `"test"` and `--comment` defaults to `"test"` when run from the CLI. Output goes to `results/{MODEL}/{DATASET}/test/`. To organize results across multiple runs, always set these explicitly.
>
> ```bash
> python main.py --model VBPR --dataset Beauty --type experiment --comment v1
> # → results/VBPR/Beauty/experiment/[VBPR]-[Beauty]-[experiment.v1].csv
> ```

### Metrics CSV columns

| Column | Description |
|---|---|
| `epoch` | Epoch number where best validation score was achieved |
| `valid_NDCG@10` | Validation NDCG at cutoff 10 (or whichever `valid_metric` is set) |
| `test_Recall@10`, `test_NDCG@10`, ... | Test metrics at all configured `topk` cutoffs |

### Recommendation export rows

JSON writes one array of evaluated-user records:

| Column | Description |
|---|---|
| `user_id` | NexusRec zero-based internal user index from evaluation |
| `items` | Ordered recommendation objects; each object has `rank`, `item_id`, and optional `score` |

JSON example:

```json
[{"user_id": 0, "items": [{"rank": 1, "item_id": 3, "score": 0.25955888628959656}, {"rank": 2, "item_id": 5, "score": 0.06581674516201019}, {"rank": 3, "item_id": 4, "score": -0.26425743103027344}]}]
```

JSONL remains available when explicitly requested and writes one grouped user record per line. CSV and TSV are long tables with `user_id`, `rank`, `item_id`, and optional `score`. Each exported data file has a sibling `.metadata.json` with `id_space=nexusrec_internal`, user/item counts, row grain, ordering and score semantics, metrics, provenance, and HPO lineage when the run came from `optimization.final_train`.

---

## Common Issues

**`ModuleNotFoundError` for the model:** Check that the model file exists under the canonical paradigm root selected by the YAML flags (`models/centralized/id/`, `models/centralized/multimodal/`, `models/federated/id/`, `models/federated/multimodal/`, `models/sequential/id/`, or `models/sequential/multimodal/`), typically inside a family subpackage such as `graph/` or `factorization/`, and that the class name matches `--model` exactly (case-sensitive).

**`KeyError` on a config key:** A required config key is missing from `configs/overall.yaml` or the model YAML. Add the key with its default value — never use `config.get(key, default)`.

**Feature shape mismatch:** The number of rows in `image_features.npy` and `text_features.npy` must equal `max(itemID) + 1`, and feature dimensions must match `features.visual_dim` / `features.text_dim`. Re-extract features or fix the preprocessing map.

**Recommendation export id error:** If `output.export.enabled=true`, exported `user_id` values must be within the user count, exported item ids must be within the item count, each user may appear only once, and each user's `items` list must not contain duplicate items.

**CUDA out of memory:** Reduce `train_batch_size` or `eval_batch_size` in the model YAML or via `--param_overrides`.

---

## Preparing a Custom Dataset

NexusRec expects data in a specific format. This section explains how to convert raw interaction logs into that format, how to choose a split strategy, what counts as a minimum viable dataset, and how to verify the result before running any model.

### Converting raw interaction data

For ordinary training, the only required file is `datasets/{DATASET}/inter.csv` with three columns:

| Column | Type | Notes |
|---|---|---|
| `userID` | int | User identifier. Values do not need to start at 0, but the framework uses `max(userID) + 1` to size embeddings, so large gaps waste memory. |
| `itemID` | int | Item identifier. Same sizing rule applies. |
| `split_label` | int | 0 = train, 1 = validation, 2 = test |

A typical conversion from a raw log looks like this: map raw user/item IDs to contiguous integers (optional but saves memory), then assign each interaction a `split_label` according to your split strategy.

```python
import pandas as pd

# Load your raw data
df = pd.read_csv("raw_interactions.csv")

# Map string/non-contiguous IDs to integers if needed
user_ids = {u: i for i, u in enumerate(df["user"].unique())}
item_ids = {v: i for i, v in enumerate(df["item"].unique())}
df["userID"] = df["user"].map(user_ids)
df["itemID"] = df["item"].map(item_ids)

# Assign split labels (see strategies below)
df["split_label"] = 0   # filled in per strategy

df[["userID", "itemID", "split_label"]].to_csv(
    "datasets/MyDataset/inter.csv", index=False
)
```

### Temporal split vs. ratio split

NexusRec reads the `split_label` column directly — it does not perform splitting internally. You assign labels yourself in your preprocessing script. Two common strategies:

**Leave-one-out (temporal) split**: For each user, sort interactions by timestamp. Assign the last interaction `split_label=2` (test), the second-to-last `split_label=1` (valid), and all earlier ones `split_label=0` (train). This is the standard for sequential and most collaborative filtering papers.

```python
df = df.sort_values(["userID", "timestamp"])
df["split_label"] = 0
df.loc[df.groupby("userID").tail(1).index, "split_label"] = 2
valid_idx = df.groupby("userID").tail(2).groupby("userID").head(1).index
df.loc[valid_idx, "split_label"] = 1
```

Use temporal split when: the model should not see future interactions during training, or when you want results to be comparable with papers that use the same protocol (most CF and sequential papers).

**Ratio split**: Assign a random fraction of each user's interactions to validation and test. A common split is 80% train / 10% valid / 10% test.

```python
def assign_ratio_split(group, train_ratio=0.8, valid_ratio=0.1):
    n = len(group)
    idx = group.index.tolist()
    train_end = int(n * train_ratio)
    valid_end = train_end + max(1, int(n * valid_ratio))
    labels = [0] * train_end + [1] * (valid_end - train_end) + [2] * (n - valid_end)
    return pd.Series(labels, index=idx)

df["split_label"] = df.groupby("userID").apply(assign_ratio_split).reset_index(level=0, drop=True)
```

Use ratio split when: you have very sparse users (few interactions per user) where leave-one-out would leave only one or two training examples, or when the paper you are replicating uses ratio split.

### Minimum viable dataset

The smallest dataset that will run without errors must satisfy:

- At least 2 unique users and 2 unique items.
- Every user must have at least 1 interaction with `split_label=0` (train), 1 with `split_label=1` (valid), and 1 with `split_label=2` (test). Users with no validation interactions are filtered by `filter_out_cold_start_users: true` (the default) and produce empty evaluation batches.
- `split_label` values must be exactly 0, 1, and 2 (integers). No other values are recognized.
- The `itemID` and `userID` columns must contain non-negative integers. The framework does not require contiguous IDs but uses `max(ID) + 1` to size embeddings, so large gaps waste memory.

For multimodal models, the feature files must additionally satisfy:

- Number of rows = `max(itemID) + 1` (i.e., row `i` corresponds to item `i`).
- Feature dimensions match `features.visual_dim` and `features.text_dim`.
- All rows should contain finite float values. Validate this in preprocessing because NaN or Inf values can surface later as unstable losses or invalid scores.

### Verifying the dataset

Run these checks before running any model:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("datasets/MyDataset/inter.csv")

# 1. Required columns present
assert {"userID", "itemID", "split_label"}.issubset(df.columns), "Missing columns"

# 2. split_label values are exactly {0, 1, 2}
assert set(df["split_label"].unique()).issubset({0, 1, 2}), "Invalid split_label values"

# 3. Every user has at least one row in each split
for label, name in [(0, "train"), (1, "valid"), (2, "test")]:
    users_in_split = set(df[df["split_label"] == label]["userID"])
    missing = set(df["userID"].unique()) - users_in_split
    if missing:
        print(f"WARNING: {len(missing)} users missing from {name} split")

# 4. No negative IDs
assert df["userID"].min() >= 0, "Negative userID"
assert df["itemID"].min() >= 0, "Negative itemID"

# 5. Print summary
n_users = df["userID"].nunique()
n_items = df["itemID"].nunique()
n_train = (df["split_label"] == 0).sum()
n_valid = (df["split_label"] == 1).sum()
n_test = (df["split_label"] == 2).sum()
print(f"Users: {n_users}, Items: {n_items}")
print(f"Train: {n_train}, Valid: {n_valid}, Test: {n_test}")
print(f"Density: {len(df) / (n_users * n_items) * 100:.4f}%")
```

For multimodal datasets, verify the feature files align with the interaction data:

```python
import numpy as np
import pandas as pd

df = pd.read_csv("datasets/MyDataset/inter.csv")
n_expected_rows = df["itemID"].max() + 1

v_feat = np.load("datasets/MyDataset/image_features.npy")
t_feat = np.load("datasets/MyDataset/text_features.npy")

assert v_feat.shape[0] == n_expected_rows, (
    f"image_features.npy has {v_feat.shape[0]} rows, expected {n_expected_rows}"
)
assert t_feat.shape[0] == n_expected_rows, (
    f"text_features.npy has {t_feat.shape[0]} rows, expected {n_expected_rows}"
)
assert v_feat.shape[1] == 512, "Update this check to your configured visual_dim"
assert t_feat.shape[1] == 512, "Update this check to your configured text_dim"
assert np.isfinite(v_feat).all(), "image_features.npy contains NaN or Inf"
assert np.isfinite(t_feat).all(), "text_features.npy contains NaN or Inf"
print(f"Visual dim: {v_feat.shape[1]}, Text dim: {t_feat.shape[1]}")
```

The visual and text dims printed here are the values you should set as `visual_dim` and `text_dim` in your model YAML.

---

## Research Workflow

This section describes the end-to-end process for implementing a new model from a paper and running a complete experiment. Each step includes the exact command or code change needed.

### Step 1: Choose a skeleton model

Find the existing model whose architecture is closest to the paper you are implementing. Use it as a starting point rather than writing from scratch.

| If the paper describes... | Start from... |
|---|---|
| CF with MF-style scoring | `models/centralized/multimodal/factorization/vbpr.py` (multimodal) or `models/centralized/id/graph/lightgcn.py` (graph) |
| Graph-based multimodal fusion | `models/centralized/multimodal/graph/freedom.py` or `models/centralized/multimodal/graph/dragon.py` |
| Self-supervised contrastive learning | `models/centralized/multimodal/contrastive/bm3.py` |
| Sequential recommendation | `models/sequential/id/sasrec.py` |
| Federated ID-only | `models/federated/id/fedavg.py` |
| Federated multimodal | `models/federated/multimodal/mmfedavg.py` |

Copy the skeleton file and YAML:

```bash
cp models/centralized/multimodal/factorization/vbpr.py models/centralized/multimodal/factorization/mymodel.py
cp configs/models/VBPR.yaml configs/models/MyModel.yaml
```

### Step 2: Implement the model

Edit the copied model file in its canonical package. Key changes from the skeleton:

1. Rename the class: `class MyModel(RecommenderBase):`
2. Remove architecture-specific layers from `__init__` and add yours.
3. Implement `calculate_loss()` to match the paper's training objective.
4. Implement `full_sort_predict()` as a batched matrix multiply for efficient evaluation.
5. Update `configs/models/MyModel.yaml`: set `is_federated`, `is_multimodal_model`, `is_sequential` correctly, and set architecture hyperparameters.

See [03-add-new-model.md](03-add-new-model.md) for complete code templates for each paradigm.

### Step 3: Sanity check (1-2 epochs, small batch)

Before running a full experiment, verify the model trains without errors:

```bash
python main.py --model MyModel --dataset Beauty \
    --param_overrides '{"max_epochs": 2, "train_batch_size": 64, "eval_batch_size": 128}'
```

What to verify at this stage:

- No Python errors on startup (import, config, dataset loading).
- Loss is a finite number on the first batch (not NaN or Inf).
- Loss decreases (or at least is not constant) across batches.
- Evaluation runs and produces non-zero metrics.

If loss is NaN immediately, see the troubleshooting guide ([06-troubleshooting.md](06-troubleshooting.md)).

### Step 4: Run a full experiment

Once the sanity check passes:

```bash
python main.py --model MyModel --dataset Beauty \
    --type experiment --comment v1
```

Monitor `outputs/logs/MyModel/Beauty/[MyModel]-[Beauty]-[experiment.v1]-[TIMESTAMP].txt` for per-epoch metrics. Training will stop when validation NDCG@10 does not improve for `stopping_step` consecutive evaluations (default 10). With `eval_step: 1` that equals 10 epochs; with `eval_step: 5`, it equals 50 epochs.

To run on a different dataset:

```bash
python main.py --model MyModel --dataset Clothing --type experiment --comment v1
```

### Step 5: Run HPO

Once the model trains correctly, tune hyperparameters. First define the search space in `configs/models/MyModel.yaml`:

```yaml
hyper_parameters: ["learning_rate", "embedding_size", "dropout_rate"]

parameter_space:
  learning_rate:
    type: "loguniform"
    low: 1.0e-4
    high: 1.0e-1
  embedding_size:
    type: "choice"
    values: [32, 64, 128, 256]
  dropout_rate:
    type: "uniform"
    low: 0.0
    high: 0.5
```

Then run HPO:

```bash
python main.py --model MyModel --dataset Beauty \
    --smart_hpo --strategy bayesian --hpo_budget 50
```

The best configuration is printed at the end. For Bayesian/TPE runs, the HPO state is persisted under `outputs/hyper_search/MyModel/Beauty/` through the trial-history CSV and `optuna_journal.log`.

For a formal model after HPO, prefer automatic final training:

```bash
python main.py --model MyModel --dataset Beauty \
    --smart_hpo --strategy bayesian --hpo_budget 50 \
    --param_overrides '{"optimization": {"final_train": {"enabled": true}}}'
```

This runs HPO first, then starts one ordinary training run with the selected `best_configuration`. The final-train run defaults to `type: final_train`, `comment: best_config`, `save_model: true`, `eval_final_test: true`, and `eval_test_during_training: false`.

To export final recommendations from that formal run, enable `output.export` in the same command:

```bash
python main.py --model MyModel --dataset Beauty \
    --smart_hpo --strategy bayesian --hpo_budget 50 \
    --param_overrides '{"optimization": {"final_train": {"enabled": true}}, "output": {"export": {"enabled": true, "formats": ["json"]}}}'
```

Manual `--param_overrides` with copied best values still works, but it is now mainly useful for reproducing a known configuration outside an HPO run.

### Step 6: Compare with baselines

Run the same dataset with baseline models using the same paradigm. For a multimodal model on Beauty:

```bash
python main.py --model VBPR --dataset Beauty --type experiment --comment baseline
python main.py --model BM3 --dataset Beauty --type experiment --comment baseline
python main.py --model FREEDOM --dataset Beauty --type experiment --comment baseline
```

Results are in `outputs/results/{MODEL}/Beauty/experiment/`. Collect the `test_*` columns from each CSV for comparison.

To ensure fair comparison, baselines should use their own HPO-tuned configs (or the configs from the original papers, if available as model YAMLs).

### Step 7: Save and report results

The default automatic final-train results CSV is at `outputs/results/MyModel/Beauty/final_train/[MyModel]-[Beauty]-[final_train.best_config].csv`. The columns you need for a paper table are `test_Recall@10`, `test_NDCG@10`, `test_Recall@20`, and `test_NDCG@20` (or whichever cutoffs you report).

If `output.export.enabled=true`, recommendation files are written beside that CSV by default, or under `output.export.path` when it is set.

To run across multiple datasets in one session, use a shell loop:

```bash
for dataset in Beauty Clothing Sports; do
    python main.py --model MyModel --dataset $dataset \
        --type final --comment best_config
done
```

That loop is a manual final run and writes each dataset under `outputs/results/MyModel/{DATASET}/final/`. Automatic HPO final training writes under `final_train/` by default. Checkpoint files (if `save_model: true`) go to `outputs/checkpoints/MyModel/{DATASET}/best_model.pth`.
