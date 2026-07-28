# Troubleshooting Guide

This guide covers the most common failure modes in NexusRec, organized by the phase where the error occurs. Each entry follows a consistent format: what you see, why it happens, and exactly what to change.

---

## Debug Checklist

Run through this list before investigating further. Most problems are caught by these checks.

- [ ] Does `datasets/{DATASET}/inter.csv` exist and contain the three required columns: `userID`, `itemID`, `split_label`?
- [ ] Does `configs/models/{ModelName}.yaml` exist with the exact same name (case-sensitive) as the `--model` argument?
- [ ] Does the model file exist under the canonical paradigm root implied by `is_federated`, `is_multimodal_model`, and `is_sequential`, such as `models/centralized/id/`, `models/centralized/multimodal/`, `models/federated/id/`, `models/federated/multimodal/`, `models/sequential/id/`, or `models/sequential/multimodal/`, and is it placed in a sensible family subpackage?
- [ ] Are all three paradigm flags present in the model YAML (`is_federated`, `is_multimodal_model`, `is_sequential`)?
- [ ] For non-end2end multimodal models: do `datasets/{DATASET}/image_features.npy` and `datasets/{DATASET}/text_features.npy` exist?
- [ ] For non-end2end multimodal models: does the number of rows in the feature `.npy` files equal `max(itemID) + 1`?
- [ ] For federated models: do `get_shared_parameters()` and `get_personal_parameters()` together cover every key in `model.state_dict()`?
- [ ] Does every name in `hyper_parameters:` appear as `config["name"]` somewhere in the model code?
- [ ] Is there any `config.get("key", default)` in the model code? (Not allowed — must be `config["key"]`.)
- [ ] Is there any hardcoded numeric literal that affects training or evaluation? (Must be in YAML.)
- [ ] Does the log file exist in `outputs/logs/{MODEL}/{DATASET}/`? If yes, check the last 50 lines for the root cause.

---

## Startup Errors

### `FileNotFoundError: Required config file not found: ...configs/overall.yaml`

**Symptom**: Process exits immediately with this message before any training.

**Cause**: `ConfigManager` locates `overall.yaml` using `Path(__file__).resolve().parent.parent`, which is resolved from the source file location at import time — not from the current working directory. This error appears when the `configs/overall.yaml` file is physically absent (e.g., the `configs/` directory was renamed or deleted), or when the NexusRec package itself is installed in a location where the `configs/` directory is not adjacent to `core/`.

**Fix**: Always run `python main.py` from the project root (`NexusRec/`). Verify the file exists at `configs/overall.yaml`.

---

### `ValueError: Model config not found: configs/models/{ModelName}.yaml`

**Symptom**: Error raised by `load_model_profile()` in `core/model_registry.py` during startup. The message contains the full absolute path of the missing file (e.g., `Model config not found: /home/user/NexusRec/configs/models/MyModel.yaml`).

**Cause**: The YAML file for the model does not exist. This is raised when the registry tries to read the paradigm flags (`is_federated`, `is_multimodal_model`, `is_sequential`) before importing the model class.

**Fix**: Create `configs/models/{ModelName}.yaml` with at least the three paradigm flags:

```yaml
is_federated: false
is_multimodal_model: false
is_sequential: false
```

If this happens from a generated queue, inspect the queue file first. Local queue files can go stale; structured benchmark manifests are the safer starting point for repository-maintained experiment plans. For the neural collaborative filtering baseline, use the paper-facing model name `NCF`; the former `NeuMF` command name remains available only for compatibility.

---

### `ImportError: Failed to load model {ModelName} from [...]`

**Symptom**: Error raised by `get_model()` in `core/model_registry.py` after config loading succeeds.

**Cause**: One of three sub-causes. The registry reads the paradigm flags from the model YAML and then imports exactly one canonical module path. If that import fails:

1. The file does not exist at any of those locations.
2. The file exists but has a syntax error or an import that fails.
3. The file exists and imports cleanly, but the class name does not exactly match the `--model` argument (case-sensitive). `--model GRU4Rec` requires `class GRU4Rec`, not `class Gru4Rec` or `class gru4rec`.

**Fix**:

- Verify the file exists at the correct path.
- Check the class name matches exactly: `class {ModelName}(RecommenderBase)`.
- Run the package-specific import directly to surface import errors, for example `python -c "import models.centralized.multimodal.factorization.vbpr"` or `python -c "import models.federated.id.fedavg"`.

---

### `ValueError: Missing required training config: ['max_epochs', 'learning_rate']` or similar

**Symptom**: `ConfigValidationError` raised during `_validate_training()` in `ConfigManager`.

**Cause**: A required key is absent from the merged config. This most commonly happens when a new key was added to a model's code with `config["new_key"]` but was never added to `configs/overall.yaml` or the model YAML. Because `ConfigManager` does not silently fill in defaults, the key is simply missing.

**Fix**: Add the missing key to `configs/overall.yaml` (for framework-wide keys) or to `configs/models/{ModelName}.yaml` (for model-specific keys).

---

### `ConfigValidationError: dropout_rate=1.5 must be between 0.0 and 1.0`

**Symptom**: Error raised by `validate_parameter_ranges()` at the end of `ConfigManager.__init__`.

**Cause**: A dropout-related config key (`dropout_rate`, `attention_dropout_rate`, `message_dropout_rate`) was set to a value outside `[0.0, 1.0]`. This often happens with `--param_overrides` during HPO or manual experiments.

**Fix**: Set the dropout value to a float in `[0, 1]`. For HPO search spaces, set `"type": "uniform", "low": 0.0, "high": 0.5`.

---

### `ValueError: File {path}/inter.csv not exist`

**Symptom**: Error raised by `RecDataset.__init__()` in `core/data/dataset.py`.

**Cause**: The interaction file does not exist at `datasets/{DATASET}/inter.csv`. The `data_path` config key (default `./datasets/`) is joined with `dataset` to form the expected path.

**Fix**: Ensure the file exists at `datasets/{DATASET}/inter.csv`. The dataset name in the path must match the `--dataset` argument exactly.

---

### `ValueError: Empty interaction dataframe for dataset at {path}`

**Symptom**: Error raised immediately after loading the CSV.

**Cause**: The `inter.csv` file exists but contains no rows (or only a header row).

**Fix**: The file must contain at least one interaction row per split label (0, 1, 2). See the "Preparing a Custom Dataset" section in [01-quick-start.md](01-quick-start.md) for minimum dataset requirements.

---

### `ValueError: Usecols do not match columns, columns expected but not found: [...]`

**Symptom**: Error raised by `pandas.read_csv()` inside `load_inter_graph()` in `RecDataset`. The message comes from pandas, not from NexusRec directly.

**Cause**: The CSV is missing one or more of the required columns. `load_inter_graph()` calls `pd.read_csv(usecols=cols)` where `cols` contains `userID`, `itemID`, and `split_label`. If any column is absent, pandas raises this error before NexusRec's own column check runs. These column names are controlled by `USER_ID_FIELD`, `ITEM_ID_FIELD`, and `inter_splitting_label` in `overall.yaml`.

**Fix**: Add the missing columns to the CSV. If your data uses different column names, override the field mapping keys in `configs/overall.yaml`:

```yaml
USER_ID_FIELD: "user_id"
ITEM_ID_FIELD: "item_id"
inter_splitting_label: "split"
```

---

### `KeyError: Missing required federated config key(s): [...]`

**Symptom**: `KeyError` raised by `extract_federated_params()` in `core/config.py` during `FederatedTrainer.__init__`.

**Cause**: The `federated:` section is missing one or more of the required keys: `local_epochs`, `clients_sample_ratio`, `clients_sample_strategy`, `aggregation_method`. These must be present in `configs/overall.yaml` under the `federated:` group (they are set there by default).

**Fix**: The defaults are already in `configs/overall.yaml`. This error most commonly occurs when the `federated:` section was manually deleted from `overall.yaml`, or when a model YAML replaces the entire `federated:` block instead of only specifying the keys to change. Because `ConfigManager` uses deep-merge (sibling keys under `federated:` are preserved even when you add new ones), you only need to specify the keys you want to override:

```yaml
# Correct: only override local_epochs, other keys stay from overall.yaml
federated:
  local_epochs: 10
```

---

### `ValueError: Unsupported aggregation_method: '{method}'`

**Symptom**: Error raised in `FederatedTrainer.__init__`.

**Cause**: The `aggregation_method` key in the `federated:` config is set to something other than `"fedavg"`. Only `fedavg` is currently implemented.

**Fix**: Set `aggregation_method: "fedavg"` in your model YAML or leave it at the `overall.yaml` default.

---

### `ValueError: Model {model} declares server-gradient-aggregated params [...] but 'server_learning_rate' is not set in config`

**Symptom**: Error raised during `FederatedTrainer.__init__` for models implementing `get_server_grad_param_names()`.

**Cause**: The model declares that some shared parameters should use delta-based aggregation (used by algorithms like FedVLR), but `server_learning_rate` was not added to the model YAML. The framework refuses to use an implicit default for this parameter because the correct value is algorithm-specific.

**Fix**: Add `server_learning_rate: 0.01` (or whatever value the paper specifies) to `configs/models/{ModelName}.yaml`.

---

### `ValueError: server_grad_param_names [...] are not in shared_param_names`

**Symptom**: Error raised during `FederatedTrainer.__init__` for models with delta-aggregation.

**Cause**: Parameters returned by `get_server_grad_param_names()` must also be returned by `get_shared_parameters()`. A parameter cannot be delta-aggregated on the server unless it is also classified as shared (i.e., not personal).

**Fix**: Add the missing parameter names to `get_shared_parameters()`.

---

## Training Problems

### NaN loss at epoch 1 (centralized)

**Symptom**: Log shows `NaN` or `Inf` loss on the very first batch. Training aborts after `nan_abort_threshold` (default 3) consecutive NaN losses.

**Cause**: Common root causes:

1. Learning rate is too high for the model architecture.
2. Embeddings or projections are not initialized properly (e.g., using `nn.Linear` with very large default weights and no normalization).
3. A cross-space operation: adding a tensor of shape `[B, 512]` (visual features) to a tensor of shape `[B, 64]` (CF embeddings) without projection. This would cause a shape error first, but if dimensions accidentally match due to misconfiguration, the result may be numerically unstable.
4. `torch.log` or `torch.log2` called on a value that is zero or negative.

**Fix**: Start with a small learning rate (`1e-4`) and a single epoch to verify the forward pass is stable. Check that all projections are correct (visual features → `embed_size`, not directly added). Use `--param_overrides '{"learning_rate": 1e-4}'` to test.

---

### NaN loss in federated training (single or multiple rounds)

**Symptom**: Log shows `[Round N] M/K clients produced NaN loss — discarding round, global model and personal state unchanged`. The round is discarded and training continues.

**Cause**: One or more clients have degenerate data (e.g., a user with a single interaction and no negatives, causing division-by-zero in BPR loss). The federated trainer discards the entire round and does not update the global model or any personal states.

**Fix**: If this is occasional (a few users per round), it is expected behavior. If it is pervasive (most rounds are NaN), reduce the learning rate. You can also increase `min_seq_len` or add a minimum interaction count filter when preparing the dataset to remove degenerate users.

---

### `ValueError: Training diverged: N consecutive NaN/Inf losses` (sequential models only)

**Symptom**: `ValueError` raised inside `SequentialTrainer.train_epoch()`. Training terminates with an error rather than silently discarding the batch.

**Cause**: `SequentialTrainer` counts consecutive NaN/Inf batches and raises `ValueError` when `nan_abort_threshold` (default `3`) is reached. This differs from centralized `TrainerBase`, which silently skips NaN batches and continues. The abort-on-NaN behavior is intentional for sequential models because NaN in a sequence model's RNN/Transformer typically means the hidden state is corrupted and cannot self-recover.

**Fix**: The underlying cause is the same as the centralized NaN section below (learning rate too high, dimension mismatch, log of zero). To investigate without aborting, temporarily increase `nan_abort_threshold` in the model YAML or `overall.yaml` to see whether the NaN is transient or persistent. To silence the abort during debugging only, reduce the learning rate first.

---

### Loss decreases then suddenly jumps to NaN

**Symptom**: Training looks normal for N epochs, then loss becomes NaN and stays NaN.

**Cause**: Gradient explosion. The model has learned embeddings with very large magnitudes, and at some point a gradient update pushes a value to infinity.

**Fix**: Reduce `clip_grad_norm` in the model YAML (default is `5.0`; try `1.0` or `2.0`). If the issue persists, add L2 regularization via `weight_decay` or `embedding_weight_decay`.

---

### Early stopping triggers immediately (stops at epoch 1 or 2)

**Symptom**: Training ends very quickly. The log shows "Early stopping at epoch N" with N being very small.

**Cause**: The validation metric never improves after the first evaluation, causing `cur_step` to increment on every evaluation. This can mean:

1. `valid_metric` is set to a metric that the model genuinely cannot improve (e.g., `NDCG@10` but the model only improves `Recall@50`).
2. `stopping_step` is set too low (e.g., `2`).
3. The model's `full_sort_predict()` returns a constant tensor (a bug in the implementation).

**Fix**: Verify `valid_metric` in `configs/overall.yaml` or the model YAML. Increase `stopping_step`. Add a print/log statement in `full_sort_predict()` to verify it returns varying scores across batches.

---

### Training is very slow (centralized)

**Symptom**: Each epoch takes much longer than expected.

**Cause**: The default `full_sort_predict()` in `RecommenderBase` loops per user in Python, which is O(batch_size) forward passes. Any model that does not override `full_sort_predict()` with a batched matrix multiplication will be very slow at evaluation time.

**Fix**: Override `full_sort_predict()` with a single matrix multiply:

```python
def full_sort_predict(self, interaction, *args, **kwargs):
    users = interaction[0]
    user_e = self.user_embedding(users)           # [B, D]
    item_e = self.item_embedding.weight            # [n_items, D]
    return torch.matmul(user_e, item_e.t())        # [B, n_items]
```

Also consider reducing `eval_step` in the model YAML to evaluate less frequently during early development.

---

### CUDA out of memory during training

**Symptom**: `RuntimeError: CUDA out of memory` during a forward pass or evaluation.

**Cause**: `train_batch_size` or `eval_batch_size` is too large for the available GPU memory. For multimodal models, the feature tensors add to the base memory requirement.

**Fix**: Reduce batch sizes in the model YAML or via `--param_overrides '{"train_batch_size": 512, "eval_batch_size": 1024}'`. For federated models with many clients, the `_client_model` and all `client_models` personal state dicts are kept in memory; reducing `clients_sample_ratio` reduces peak round memory.

---

### CUDA out of memory during federated evaluation

**Symptom**: OOM occurs during evaluation (not training), even with small batch sizes.

**Cause**: Federated evaluation calls `model.full_sort_predict()` for every user one at a time. If each call allocates a `[1, n_items]` tensor on GPU and it is not freed quickly, memory accumulates. This is most likely on large datasets (many items).

**Fix**: Ensure the model's `full_sort_predict()` does not retain unnecessary intermediate tensors. If the dataset has a very large item space, reduce `eval_batch_size` even though evaluation is per-user for federated models; the batch size controls how many users are processed at once when using `EvalDataLoader`.

---

### `RuntimeError: Expected all tensors to be on the same device`

**Symptom**: Error during `calculate_loss()` or `full_sort_predict()`.

**Cause**: Either `self.v_feat` or `self.t_feat` was not moved to `self.device`. The `RecommenderBase.to(device)` override handles this automatically, but models that assign feature tensors after `super().__init__()` or after the model is moved to device may miss it.

**Fix**: Call `self.setup_multimodal_features(config)` before building any layers that use the features. `RecommenderBase.to()` will then move them correctly. If you assign features manually, always follow with `.to(self.device)`:

```python
self.v_feat = some_tensor.to(self.device)
```

---

## Evaluation Problems

### `TypeError: The metrics must be a list, but get <class 'str'>`

**Symptom**: Error raised by `validate_topk_args()` in `TopKEvaluator.__init__`.

**Cause**: The `metrics` config key was set to a string (`"Recall"`) instead of a list (`["Recall"]`). This can happen if the model YAML overrides `metrics` with a bare string instead of a YAML list.

**Fix**: In the model YAML or `overall.yaml`, use list syntax:

```yaml
metrics: ["Recall", "NDCG", "Precision"]
```

---

### `ValueError: There is no metric named '{metric}'`

**Symptom**: Error raised by `validate_topk_args()` when `TopKEvaluator` initializes.

**Cause**: The `metrics` list contains a name that is not in the supported set. Supported metric names are: `Recall`, `Precision`, `NDCG`, `MAP`, `MRR`, `Hit`, `Diversity`, `Novelty`, `Coverage`. The comparison is case-insensitive, but the name must match exactly.

**Fix**: Change the metric name to one from the supported list. Common mistakes: `"recall"` (fine, case-insensitive), `"AUC"` (not supported), `"F1"` (not supported), `"HR"` (use `"Hit"` instead).

---

### `TypeError: The topk must be a list, but get <class 'int'>`

**Symptom**: Error raised by `validate_topk_args()`.

**Cause**: The `topk` config key was set to a single integer (`10`) instead of a list (`[10]`).

**Fix**:

```yaml
topk: [10, 20, 50]
```

---

### `ValueError: The topk must be a positive integer, but get {k}`

**Symptom**: Error raised by `validate_topk_args()` in `TopKEvaluator.__init__`.

**Cause**: One of the values in the `topk` list is `0`, a negative integer, or a non-integer type. All topk cutoffs must be strictly positive integers.

**Fix**: Check the `topk` list in `overall.yaml` or the model YAML:

```yaml
topk: [10, 20, 50]    # correct
topk: [0, 10, 20]     # wrong — 0 is not a valid cutoff
```

---

### `AssertionError` in evaluation: `len(pos_len_list) != len(topk_index)`

**Symptom**: AssertionError in `TopKEvaluator.evaluate()`.

**Cause**: The number of users in the evaluation data does not match the number of prediction rows accumulated in `batch_matrix_list`. This indicates a bug in `full_sort_predict()`: it returned a different number of rows than the number of users in the batch, or the evaluation loop skipped or duplicated batches.

**Fix**: Verify `full_sort_predict()` returns exactly `[batch_size, n_items]` where `batch_size` matches `len(interaction[0])`. Do not skip any user or return extra rows.

---

### All metrics are 0.0

**Symptom**: Training completes but every metric in the results CSV is `0.0`.

**Cause**: Usually one of these:

1. `full_sort_predict()` returns the same score for all items (e.g., all zeros or a constant), so `torch.topk` selects the first K items by index. The target item is almost never among item IDs 0 through K-1.
2. `filter_seen=true` is masking the target item because it was incorrectly included in the training split.
3. The `split_label` column was assigned incorrectly: all target items ended up in split 0 (train) instead of split 2 (test).

**Fix**: Add a debug print in `full_sort_predict()` to verify the score variance across items. Verify that each user has at least one row with `split_label=2` in `inter.csv`.

---

### `valid_metric` not found

**Symptom**: Training fails with a `KeyError` when the framework tries to extract the target validation metric.

**Cause**: `valid_metric` in config is set to a metric or cutoff that is not present in the evaluator's output dict. Common mistakes: `"NDCG10"` (missing `@`), `"NDCG@5"` when `5` is not in the `topk` list, or `"MAP@10"` when `MAP` is not in the `metrics` list.

**Fix**: The `valid_metric` value must follow the format `"{MetricName}@{K}"` where `MetricName` is from the supported set and `K` is an integer present in the `topk` list. Example: `valid_metric: "NDCG@10"` requires `10` to be in `topk: [10, 20, 50]`.

---

## Recommendation Export Problems

### `output.export` is enabled but no recommendation file appears

**Symptom**: Training finishes and result CSV exists, but no `.json` recommendation export is written.

**Cause**: Recommendation export only runs on the final test evaluation path. It does not run during validation, training-time test evaluation, or HPO trials. If `eval_test_during_training=true`, normal training skips the final-test path. If `smart_hpo=true` but `optimization.final_train.enabled=false`, every HPO trial disables export and no formal export is produced.

**Fix**: For a normal run, keep `eval_final_test: true` and `eval_test_during_training: false`. For HPO, enable final training:

```bash
python main.py --model VBPR --dataset Beauty --smart_hpo \
  --param_overrides '{"optimization": {"final_train": {"enabled": true}}, "output": {"export": {"enabled": true}}}'
```

---

### `output.export is the recommendation-list export. Disable legacy save_recommended_topk...`

**Symptom**: Config validation fails before training starts.

**Cause**: `output.export.enabled=true` and legacy `save_recommended_topk=true` are both enabled. They produce different artifact contracts: `output.export` writes schema-managed recommendation-list artifacts, while `save_recommended_topk` writes legacy internal item-id diagnostics.

**Fix**: Use one contract per run. For offline jobs or on-device reranking, keep only `output.export.enabled=true` and set `save_recommended_topk: false`.

---

### `Unsupported output.export format(s)` or duplicate formats

**Symptom**: Config validation fails with an unsupported or duplicate format message.

**Cause**: `output.export.formats` must be a non-empty list of unique strings chosen from `json`, `jsonl`, `csv`, and `tsv`.

**Fix**:

```yaml
output:
  export:
    enabled: true
    formats: ["json"]
```

Other export contract fields are validated at config time as well: `enabled`,
`include_scores`, and `save_recommended_topk` must be booleans; `path` must be a
string; and `topk` must be `null` or a positive integer.

---

### `output.export currently supports split='test' only`

**Symptom**: Config validation fails after changing `output.export.split`.

**Cause**: The export contract currently supports final test artifacts only.

**Fix**: Set `output.export.split: "test"`.

---

### `Recommendation export user_id/item_id is out of range`

**Symptom**: Final test evaluation reaches export, then fails with an out-of-range `user_id` or `item_id`.

**Cause**: The evaluator produced ids outside the known user/item counts. For sequential models, PAD index `0` is shifted out by `item_id_offset=1`; a real exported item id must still map back into `0..n_items-1`.

**Fix**: Check dataset preprocessing, model output width, and sequential PAD handling. `inter.csv` item ids should match the item catalog expected by the model and feature files.

---

### `Recommendation export has duplicate user_id=...`

**Symptom**: Export fails because the same evaluated user appears more than once.

**Cause**: The grouped JSON/JSONL contract writes one recommendation record per user. Duplicate user rows would make downstream consumers choose between multiple ranked lists for the same user.

**Fix**: Check the evaluation data loader or sequential evaluation setup so each exported user contributes one final recommendation list.

---

### `Recommendation export has duplicate item_id=...`

**Symptom**: Export fails for one `eval_index` with a duplicate item id.

**Cause**: A user's top-k list contains the same item more than once. The export contract writes ordered recommendation objects, so duplicates would make the recommendation list ambiguous.

**Fix**: Inspect the top-k generation path for the model or trainer that produced duplicated indices.

---

### `output.export.topk=... exceeds available evaluated top-k width`

**Symptom**: Export fails at final test.

**Cause**: `output.export.topk` is larger than the top-k width already computed by the evaluator. The available width is normally `max(topk)` from the evaluation config.

**Fix**: Lower `output.export.topk` or increase evaluation `topk`:

```yaml
topk: [10, 20, 50, 100]
output:
  export:
    topk: 100
```

---

### `output.export.include_scores=true requires top-k scores`

**Symptom**: Export fails because scores are missing.

**Cause**: Score export is enabled, but the trainer did not provide top-k score values to the exporter.

**Fix**: For the current centralized, federated, and sequential trainers this should be available on the final-test path. If you are modifying trainer code, either pass the top-k score matrix or set `output.export.include_scores: false`.

---

## HPO Problems

### `ValueError: Budget configuration missing!`

**Symptom**: Error raised by `UnifiedHPOManager._get_budget()` at the start of `run_optimization()`.

**Cause**: `optimization.budget` is missing from the config. The default in `configs/overall.yaml` sets `optimization.budget: 1000`, so this should not occur in normal use. It can occur if the `optimization:` section was accidentally removed from `overall.yaml`, or if a model YAML overrides `optimization:` as an empty dict.

**Fix**: Add `optimization: {budget: 50}` to the model YAML, or pass `--hpo_budget 50` on the CLI.

---

### `ValueError: Unsupported strategy: {strategy}. Available: grid, random, bayesian, tpe`

**Symptom**: Error raised in `UnifiedHPOManager._run_enumeration_path()`.

**Cause**: The `--strategy` argument was set to an unsupported value.

**Fix**: Use one of the four supported strategies: `grid`, `random`, `bayesian`, `tpe`.

---

### HPO `grid` search has zero combinations remaining

**Symptom**: Log shows "All combinations already completed." and HPO exits immediately.

**Cause**: All Cartesian product combinations have already been run and saved to the trial history CSV. This is the correct resume behavior for grid search.

**Fix**: If you want to re-run all combinations (e.g., after changing the search space), pass `--no-resume`. If the search space itself needs to be expanded, add more values to `parameter_space` in the model YAML.

---

### HPO Bayesian/TPE does not resume

**Symptom**: Each HPO run starts from scratch despite a prior run existing.

**Cause**: The Optuna study journal file `outputs/hyper_search/{MODEL}/{DATASET}/optuna_journal.log` may be missing or corrupted. Alternatively, `--no-resume` was passed.

**Fix**: Do not pass `--no-resume` if you want to resume. Verify `outputs/hyper_search/{MODEL}/{DATASET}/optuna_journal.log` exists. If the journal is corrupted, delete `optuna_journal.log` and start fresh.

---

### HPO trial fails with `KeyError` on a config key

**Symptom**: Individual HPO trials fail and are marked as `"failed"` in the CSV.

**Cause**: The model code reads a config key (e.g., `config["my_param"]`) that is in `hyper_parameters:` but was not added to `configs/overall.yaml` as a default. When HPO injects the trial-specific parameter values, it only sets the keys listed in `hyper_parameters`. If the model reads an additional key that has no default, it raises `KeyError`.

**Fix**: Add the missing key to `configs/overall.yaml` or the model YAML as a default, then list it in `hyper_parameters` if you want HPO to search it.

---

### A hyperparameter in `hyper_parameters:` has no effect on results

**Symptom**: HPO explores different values for a parameter but the reported metric is the same regardless of the value.

**Cause**: The parameter is listed in `hyper_parameters:` but is never read by the model code via `config["param_name"]`. The model always uses a hardcoded value or a differently-named key.

**Fix**: Search the model file for `config["param_name"]`. If it is absent, either remove the parameter from `hyper_parameters:` (saving HPO budget) or wire the config read into the model code. See the canonical field names table in [02-configuration.md](02-configuration.md) to ensure the name in `hyper_parameters` matches what the code reads.

---

### `HPO final_train requires a non-empty best_configuration`

**Symptom**: HPO completes or resumes, then final training fails before model training starts.

**Cause**: `optimization.final_train.enabled=true`, but the HPO result does not contain a usable `best_configuration`. This can happen if no trial completed successfully, or if a custom HPO caller returns an incomplete result dictionary.

**Fix**: Inspect the HPO CSV and ensure at least one row has `status="completed"` and searched parameter columns. For parallel HPO, verify the merged CSV exists and contains successful shard rows.

---

## Federated-Specific Problems

### ValueError: `Federated parameter split is INCOMPLETE — the following parameters are neither shared nor personal and would be frozen at initialization`

**Symptom**: `FederatedTrainer.__init__` raises `ValueError` and the run stops immediately.

**Cause**: The union of keys returned by `get_shared_parameters()` and `get_personal_parameters()` does not cover all keys in `model.state_dict()`. The listed parameters would receive gradient updates during local training but those updates would be discarded at the end of each round, leaving them frozen at their initialization values. The trainer refuses to start rather than train a silently broken model.

**Fix**: Audit `model.state_dict().keys()` and ensure every key appears in exactly one of the two methods:

```python
# Debug: find orphaned params
model = MyFedModel(config, dataloader)
all_keys = set(model.state_dict().keys())
shared = set(model.get_shared_parameters().keys())
personal = set(model.get_personal_parameters().keys())
orphans = all_keys - shared - personal
print("Orphaned:", orphans)
```

---

### Parameter names in `get_shared_parameters()` do not appear in `model.state_dict()`

**Symptom**: The `ValueError` above is not triggered (because `split_aware_federation` is only enabled when `get_shared_parameters()` returns a non-empty dict), but the model effectively trains without any split, aggregating all parameters.

**Cause**: The parameter names returned by `get_shared_parameters()` do not match the actual keys in `model.state_dict()`. The `FederatedTrainer._extract_parameter_names()` method silently filters out names not present in `state_dict()`. If all names are filtered out, `shared_param_names` becomes empty and the framework falls back to simple FedAvg on all parameters.

**Fix**: Get the exact names from the model:

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

Use these exact strings in `get_shared_parameters()` and `get_personal_parameters()`. Common mistake: using `"user_emb.weight"` when the actual key is `"user_embedding.weight"`.

---

### Checkpoint loading fails with shape mismatch

**Symptom**: `RuntimeError: size mismatch for {layer}.weight: copying a param with shape torch.Size([A, B]) from checkpoint, the shape in current model is torch.Size([C, D])`.

**Cause**: The checkpoint was saved from a run with different hyperparameters (different `embedding_size`, different `n_users`, or different `n_items`). Checkpoints are not portable across datasets or embedding size changes.

**Fix**: Checkpoints can only be loaded into a model initialized with the same architecture. To resume training on the same dataset, use the same model YAML values. To load a checkpoint for inference, recreate the model with identical config values to those used during training (visible in the log file that was created alongside the checkpoint).

---

### Federated model evaluation produces lower scores than expected

**Symptom**: Final test metrics are lower than the best validation metrics, or lower than a centralized baseline by a suspicious margin.

**Cause**: Per-user personal parameters may not be restored correctly during evaluation, so each user is scored with a stale or generic embedding instead of their trained personal one.

**Fix**: Verify that `personal_param_names` includes per-user parameters (typically `user_embedding.weight`) so that each user's personal embedding is restored during evaluation via `_restore_personal_state()`.

---

## Common Configuration Mistakes

### Using `config.get("key", default)` in model code

**Symptom**: No immediate error. Silent wrong behavior: if the key was renamed or the YAML has a typo, the default value is used without any warning.

**Fix**: Always use `config["key"]`. If the key might not be present in older configs, add it to `configs/overall.yaml` with its default value.

---

### `valid_metric` uses a metric name not in `metrics`

**Symptom**: Training reaches evaluation and then raises a `KeyError` for the missing target metric.

**Example**: `valid_metric: "MAP@10"` but `metrics: ["Recall", "NDCG", "Precision"]`. MAP is never computed, so the key is never in the result dict.

**Fix**: Ensure `valid_metric` references a metric that is in the `metrics` list and a K that is in the `topk` list:

```yaml
metrics: ["Recall", "NDCG", "MAP"]
topk: [10, 20, 50]
valid_metric: "MAP@10"
```

---

### Modifying `_FLATTEN_GROUPS` behavior by adding a new YAML section

**Symptom**: A new top-level section added to `overall.yaml` is not accessible as a flat key in Python code.

**Cause**: Only sections listed in `_FLATTEN_GROUPS` in `core/config.py` are promoted to top-level. A new section like `mygroup:` will remain nested as `config["mygroup"]["key"]` unless added to `_FLATTEN_GROUPS`.

**Fix**: Either access the key as `config["mygroup"]["key"]`, or add `"mygroup"` to `_FLATTEN_GROUPS` in `core/config.py`. The latter is appropriate if the section contains values that should be accessed flat by many parts of the codebase (like `training` or `evaluation`). For model-specific sections, the former is cleaner.
