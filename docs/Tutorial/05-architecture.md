# Architecture Reference

This document describes the NexusRec module structure, data flow, and the reasoning behind key design decisions.

---

## Module Map

```
NexusRec/
├── main.py                        ← CLI entry point
├── configs/
│   ├── overall.yaml               ← global defaults (all models)
│   └── models/{ModelName}.yaml    ← per-model overrides and HPO space
├── models/
│   ├── centralized/id/{family}/{modelname}.py          ← centralized ID models (autoencoder/, diffusion/, factorization/, flow/, graph/)
│   ├── centralized/multimodal/{family}/{modelname}.py ← centralized multimodal models (contrastive/, diffusion/, factorization/, graph/)
│   ├── federated/id/{modelname}.py                    ← federated ID models
│   ├── federated/multimodal/{modelname}.py            ← federated multimodal models
│   ├── sequential/id/{modelname}.py                   ← sequential ID models
│   └── sequential/multimodal/{modelname}.py           ← sequential multimodal models
├── core/
│   ├── config.py                  ← ConfigManager, flattening, set_paths()
│   ├── model_registry.py          ← model/trainer discovery
│   ├── base/
│   │   ├── recommender.py         ← RecommenderBase (all paradigms)
│   │   └── trainer.py             ← TrainerBase (centralized)
│   ├── sequential/
│   │   ├── recommender.py         ← SequentialRecommender
│   │   └── trainer.py             ← SequentialTrainer
│   ├── federated/
│   │   ├── trainer.py             ← FederatedTrainer
│   │   └── dataloader.py          ← FederatedDataLoader
│   ├── training/
│   │   ├── interface.py           ← quick_start()
│   │   ├── core.py                ← train_single()
│   │   ├── environment.py         ← prepare_env(), setup_hpo_environment()
│   │   └── factory.py             ← optimizer, scheduler, loss constructors
│   ├── hpo/
│   │   ├── engine.py              ← UnifiedHPOManager, run_unified_hpo()
│   │   ├── optuna_backend.py      ← Bayesian/TPE via Optuna
│   │   ├── parallel.py            ← single-node multi-GPU trial sharding
│   │   └── parameters.py          ← grid/random parameter generation
│   ├── data/
│   │   ├── pipeline.py            ← create_loaders(), dataset construction
│   │   ├── dataset.py             ← RecDataset
│   │   └── features.py            ← setup_centralized_features(), setup_federated_features()
│   ├── evaluation/
│   │   ├── evaluator.py           ← TopKEvaluator (Recall, NDCG, Precision, MAP, MRR, Hit, Novelty, Diversity, Coverage)
│   │   ├── ranking.py             ← metric kernels
│   │   ├── topk_kernel.py         ← top-k hit matrix construction
│   │   ├── export.py              ← recommendation-list export
│   │   └── export_contract.py     ← output.export config contract
│   ├── utils/
│   │   └── metrics.py             ← extract_target_metric()
│   └── runtime/
│       └── logger.py              ← init_logger(), TrainLogger
└── datasets/
    └── {DATASET}/
        ├── inter.csv
        ├── image_features.npy     ← optional, multimodal
        └── text_features.npy      ← optional, multimodal
```

---

## Data Flow

### 1. Startup

```
main.py:load_config()
    → parse CLI args
    → assemble base config dict
    → call quick_start()
        → if smart_hpo: run_unified_hpo() or run_parallel_hpo()
             → optional optimization.final_train
        → else: run_training()
```

### 2. Environment Preparation

```
prepare_env(model_name, dataset_name, config_dict)
    → ConfigManager(model_name, dataset_name, config_dict)
        → load overall.yaml
        → deep-merge models/{MODEL}.yaml
        → deep-merge datasets/{DATASET}.yaml (if exists)
        → deep-merge datasets/{DATASET}.yaml model_overrides.{MODEL} (if exists)
        → deep-merge CLI overrides
        → flatten groups (training, evaluation, output, ...)
        → set_paths(): generate log_dir, checkpoint_dir, result paths
        → normalize hyperparameter scalars
        → validate output.export
    → init_seed(config["seed"])
    → init_logger(config)
    → if sequential: core/sequential/integration.auto_setup(config)
      else: create_loaders(config)
        → RecDataset.load(inter.csv)
        → split by split_label (0=train, 1=valid, 2=test)
        → wrap in TrainDataLoader / EvalDataLoader
           (or FederatedDataLoader if is_federated)
    → return (config, train_data, valid_data, test_data)
```

### 3. Model Construction

```
get_model(model_name)
    → read paradigm flags from configs/models/{MODEL}.yaml
    → route to one canonical package
    → return class

model = ModelClass(config, train_data)
    → RecommenderBase.__init__(config, dataloader)
        → sets self.device, self.n_users, self.n_items, self.embed_size, ...
        → self.v_feat = None, self.t_feat = None (defaults)
    → model subclass __init__ continues
        → if centralized multimodal: self.setup_multimodal_features(config)
            → load image_features.npy → self.v_feat [n_items, visual_dim]
            → load text_features.npy  → self.t_feat [n_items, text_dim]
        → build nn.Embedding, nn.Linear, etc.
```

Federated multimodal models are different: `setup_multimodal_features()` no-ops on the model instance when `is_federated=true`. `FederatedTrainer` loads feature tensors with `setup_federated_features()` and shares them with the reusable client model before local client training.

### 4. Training Loop

```
trainer.fit(train_data, valid_data, test_data)
    if resume_training=true:
        load resume_state.pth (model, optimizer, scheduler, RNG, dataloader, best tracking)
    for epoch in range(max_epochs):
        model.pre_epoch_processing()    ← optional hook
        for batch in train_data:
            loss = model.calculate_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
        model.post_epoch_processing()   ← optional hook

        if epoch % eval_step == 0:
            valid_scores = evaluator.evaluate(model, valid_data)
                → model.full_sort_predict(batch) → [B, n_items]
                → filter seen items
                → compute Recall/NDCG/Precision@K
            if valid_scores[valid_metric] > best_valid_score:
                best_valid_score = ...
                best_model_state = deepcopy(model.state_dict())
                cur_step = 0
            else:
                cur_step += 1
                if cur_step >= stopping_step:
                    break   ← early stopping

    model.load_state_dict(best_model_state)
    if eval_final_test=true and eval_test_during_training=false:
        test_scores = evaluator.evaluate(model, test_data, write_export=true)
        if output.export.enabled:
            write_recommendations(...)
    return best_valid_score, best_valid_result, test_scores
```

### 5. Federated Training (parallel to step 4)

```
FederatedTrainer.fit(train_data, valid_data, test_data)
    for round in range(max_epochs):
        sampled_clients = sample(all_clients, clients_sample_ratio)

        for client_id in sampled_clients:
            _client_model.load_state_dict(global_state_dict)
            if split-aware:
                restore client_id's personal params from client_models[client_id]
            
            for local_epoch in range(local_epochs):
                for batch in client_data[client_id]:
                    loss = _client_model.calculate_loss(batch)
                    loss.backward(); optimizer.step()
            
            accumulate client contribution (online on GPU):
                shared params: weighted sum → will be divided later
                personal params: save to client_models[client_id]

        finalize_aggregation():
            shared params: divide by total weight → new global state
            delta params (server_grad path): apply server_lr × delta
        
        commit to global model
        evaluate every eval_step rounds
```

---

## Key Design Decisions

### Why flat config access (no nested dicts in Python)?

Nested YAML groups (`training:`, `evaluation:`) are for human readability in the config files only. They are flattened to top-level by `ConfigManager` so all Python code uses `config["max_epochs"]` rather than `config["training"]["max_epochs"]`. Nested dictionaries inside flattened groups are promoted as dictionaries, so YAML `output.export` becomes runtime `config["export"]`. This eliminates a layer of indirection and makes grep-based auditing (who reads what key) straightforward.

### Why single `_client_model` instead of per-client deepcopy?

Memory. With N users, N separate model copies would multiply memory by N. Instead, `FederatedTrainer` maintains a single reusable model instance and swaps its state via `load_state_dict()` before each client's local training. The only per-user storage is the personal parameter state dict, which is typically just the user embedding row.

### Why config-first routing instead of a decorator registry?

Decorators require importing model modules at startup to register them, which causes import side-effects for all models (CUDA allocations, etc.) even when only one model is used. NexusRec now reads the paradigm flags from `configs/models/{Model}.yaml`, routes to exactly one canonical paradigm root, and only then resolves the concrete module file under that root. This keeps startup explicit and avoids global import side-effects.

### Why no `try-catch` anywhere in the codebase?

This is an academic research framework. Errors must surface immediately and visibly so researchers can diagnose experiments. Swallowed exceptions turn correctness bugs into silent degraded-performance bugs that are much harder to trace. All errors propagate to the top.

### Why are `config.get(key, default)` calls forbidden?

`config.get("key", 0.0)` hides two classes of bugs: (1) a key was renamed and the old name was never removed from code, and (2) an intentionally-zero value (e.g., `weight_decay: 0`) is silently overridden by the default. Using `config["key"]` directly makes both bugs immediately visible as `KeyError`.

### Why is HPO final training a second normal run?

The search phase should identify `best_configuration`; the formal artifact phase should train once with that configuration under ordinary training semantics. This keeps trial bookkeeping, final result CSVs, checkpoints, and recommendation exports separate. It also avoids treating a trial checkpoint as the final model when the trial existed primarily to compare hyperparameters.

---

## Inheritance Hierarchy

```
nn.Module
└── RecommenderBase         (core/base/recommender.py)
    ├── VBPR                 (models/centralized/multimodal/factorization/vbpr.py)     — centralized multimodal
    ├── FedAvg               (models/federated/id/fedavg.py)             — federated ID
    ├── MMFedAvg             (models/federated/multimodal/mmfedavg.py)   — federated multimodal
    └── SequentialRecommender (core/sequential/recommender.py)
        ├── GRU4Rec          (models/sequential/id/gru4rec.py)
        ├── SASRec           (models/sequential/id/sasrec.py)
        └── BERT4Rec         (models/sequential/id/bert4rec.py)

ABC
└── TrainerBase              (core/base/trainer.py)
    ├── FederatedTrainer     (core/federated/trainer.py)
    └── SequentialTrainer    (core/sequential/trainer.py)
```

**Rule: Maximum two layers.** `MyModel` may inherit from `RecommenderBase` or `SequentialRecommender`. It must not inherit from another model class.

---

## Configuration System Internals

```
ConfigManager.__init__(model, dataset, config_dict)
    1. Load configs/overall.yaml             → self._config
    2. Deep-merge configs/models/{MODEL}.yaml  (if file exists)
    3. Deep-merge configs/datasets/{DATASET}.yaml (if file exists)
    4. Deep-merge model_overrides.{MODEL} from the dataset YAML (if present)
    5. Deep-merge config_dict (CLI overrides)
    6. reject deprecated / forbidden keys    → fail fast on legacy config
    7. _flatten_groups()                     → promote training/evaluation/etc. to top-level
       Rule: explicit top-level model keys override flattened group defaults
    8. normalize_hparams()                   → collapse list HPO defaults to scalars (non-HPO mode)
    9. set_paths()                           → generate all output path strings
    10. extract_federated_params()           → pull federated sub-dict to top-level
```

### Flattened group resolution order

When a key appears in both a flattened group (e.g., `training: {learning_rate: 0.001}`) and at top-level in the model YAML (`learning_rate: 0.01`), the explicit top-level model value wins. The flattening pass only backfills keys that are not already present.

---

## Evaluation

`TopKEvaluator` in `core/evaluation/evaluator.py`:

1. Calls `model.full_sort_predict(batch)` → scores `[B, n_items]`
2. If `filter_seen=true`: mask training items per user with `-inf`
3. `torch.topk(scores, k=max(topk))` → top-K item indices
4. Compare against ground-truth targets per user
5. Aggregate Recall@K, NDCG@K, Precision@K across the batch
6. Return dict: `{"Recall@10": float, "NDCG@10": float, ...}`

Supported metric names in `metrics:`: `Recall`, `Precision`, `NDCG`, `MAP`, `MRR`, `Hit`, `Diversity`, `Novelty`, `Coverage`. Any other name raises a `ValueError` at evaluation time. Sequential training now passes train-split item popularity frequencies to its evaluator, so `Novelty` follows the same train-popularity semantics as the common ranking stack.

`output.export` is layered after evaluation. Trainers pass the final top-k indices, optional top-k scores, metrics, and eval users to `core/evaluation/export.py`. The exporter validates NexusRec internal user/item ids, writes grouped JSON by default, and can optionally write JSONL or CSV/TSV long tables. Each data file gets one metadata JSON. HPO trials disable export; HPO users should enable `optimization.final_train` when the best configuration needs an export artifact.
