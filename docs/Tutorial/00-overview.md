# NexusRec Tutorial — Overview

NexusRec is a research recommendation framework built for centralized, multimodal, federated, and sequential recommendation. Every experiment, model, and HPO run enters through `main.py` and `quick_start()`, then follows the same config, data, training, evaluation, and artifact contracts. Understanding this pipeline is the prerequisite for all other tutorials.

---

## The Runtime Paradigms

A model belongs to exactly one paradigm, declared by boolean flags in its YAML config.

| Paradigm | `is_federated` | `is_multimodal_model` | `is_sequential` | Examples |
|---|---|---|---|---|
| Centralized ID | `false` | `false` | `false` | BPR, LightGCN, NCF, DiffRec |
| Centralized Multimodal | `false` | `true` | `false` | VBPR, BM3, FREEDOM, DRAGON |
| ID-based Federated | `true` | `false` | `false` | FedAvg, FedNCF, PFedRec |
| Federated Multimodal | `true` | `true` | `false` | MMFedAvg, MMFedNCF, MMFedRAP, MMFCF, MMPFedRec |
| Sequential | `false` | `false` or `true` | `true` | GRU4Rec, SASRec, BERT4Rec, HM4SR |

The framework reads these flags at startup and selects the correct trainer automatically:

```
is_sequential=true  →  SequentialTrainer  +  SequentialRecommender
is_federated=true   →  FederatedTrainer   +  RecommenderBase
(neither)           →  TrainerBase        +  RecommenderBase
```

---

## Execution Pipeline

```
python main.py --model VBPR --dataset Beauty
       │
       ▼
  load_config()           ← merge CLI args into base config
       │
       ▼
  ConfigManager           ← overall.yaml → model.yaml → dataset.yaml → CLI / param_overrides
       │
       ▼
  quick_start()           ← single user-facing runtime entry
       │
       ├─ _run_hpo_flow()  ← enabled by --smart_hpo
       │      ├─ run_unified_hpo() / run_parallel_hpo()
       │      └─ optional optimization.final_train
       └─ run_training()   ← standard training path
       │
       ▼
  prepare_env()           ← load dataset, init logger, set seed
       │
       ▼
  get_model(model_name)   ← config-first routing: paradigm flags → canonical model package
       │
       ▼
  get_trainer(...)        ← paradigm-selected trainer class
       │
       ▼
  trainer.fit(train, valid, test)
       │
       ▼
  outputs/                ← logs, results CSV, checkpoints, optional exports
```

---

## Output Directory Structure

```
outputs/
  logs/{MODEL}/{DATASET}/
      [Model]-[Dataset]-[type.comment]-[timestamp].txt
  results/{MODEL}/{DATASET}/{type}/
      [Model]-[Dataset]-[type.comment].csv
      [Model]-[Dataset]-[experiment.type.comment].csv    ← serial grid-HPO trial history
      [Model]-[Dataset]-[type.comment.seedX.idx0.top50.recommendations].json
      [Model]-[Dataset]-[type.comment.seedX.idx0.top50.recommendations].json.metadata.json
  checkpoints/{MODEL}/{DATASET}/
      best_model.pth                            ← saved when save_model=true
  hyper_search/{MODEL}/{DATASET}/
      [Model]-[Dataset]-[strategy.type.comment].csv      ← Bayesian/TPE/random trial history
      optuna_journal.log                        ← Bayesian/TPE JournalStorage when resume=true
      parallel/{TYPE}/{COMMENT}_shard00/        ← isolated parallel-HPO shard state
```

Normal training result CSVs are single-row artifacts. HPO CSVs contain trial histories. Recommendation export files are optional final-test artifacts controlled by `output.export`; JSON contains ordered `items` records grouped by evaluated user, JSONL/CSV/TSV are optional explicit formats, and all export files are written only during final test evaluation.

---

## Tutorial Map

| Tutorial | You need this when... |
|---|---|
| [01 — Quick Start](01-quick-start.md) | Running your first experiment, preparing a custom dataset, research workflow |
| [02 — Configuration](02-configuration.md) | Understanding YAML hierarchy and config overrides |
| [03 — Adding a New Model](03-add-new-model.md) | Implementing a new model for any paradigm |
| [04 — HPO](04-hpo.md) | Running hyperparameter optimization |
| [05 — Architecture](05-architecture.md) | Understanding module structure and data flow |
| [06 — Troubleshooting](06-troubleshooting.md) | Diagnosing startup errors, NaN loss, evaluation problems, federated issues |
| [07 — Benchmarking](07-benchmarking.md) | Running repeated sweeps, summaries, and paired significance tests |
| [Reference — Interfaces](../reference/interfaces.md) | Calling public APIs and implementing model/data/evaluation contracts |

---

## File Conventions

- Centralized ID model file: `models/centralized/id/{family}/{modelname_lowercase}.py`
- Centralized multimodal model file: `models/centralized/multimodal/{family}/{modelname_lowercase}.py`
- Federated ID model file: `models/federated/id/{modelname_lowercase}.py`
- Federated multimodal model file: `models/federated/multimodal/{modelname_lowercase}.py`
- Sequential ID model file: `models/sequential/id/{modelname_lowercase}.py`
- Sequential multimodal model file: `models/sequential/multimodal/{modelname_lowercase}.py`
- Config file: `configs/models/{ModelName}.yaml`
- Dataset directory: `datasets/{DATASET}/`
- Interaction file: `datasets/{DATASET}/inter.csv`

---

## Agent Quick-Reference

This section is written for AI agents. It captures the invariants that hold across all paradigms.

**Invariants:**
- `config["key"]` — never `config.get("key", default)`. If a key might be missing, add it to `configs/overall.yaml` first.
- `self.n_items` is always set by `RecommenderBase.__init__` before model `__init__` continues.
- `self.device` is always a `torch.device` by the time `__init__` finishes.
- `self.v_feat` and `self.t_feat` are `None` by default. Centralized multimodal models must call `self.setup_multimodal_features(config)` in `__init__` to populate them. Federated multimodal models must NOT — `setup_multimodal_features()` no-ops for federated models; features are injected by `FederatedTrainer` after model construction and are only available in `forward()`/`calculate_loss()`/`full_sort_predict()`, not in `__init__`.
- `full_sort_predict()` return shape depends on paradigm: centralized/federated models return `[batch_size, n_items]`, where item `0` is a normal item. Sequential models may return `[batch_size, n_items+1]` when index `0` is the PAD token; only `SequentialEvaluator` suppresses PAD `0`.
- Parameter names in `get_shared_parameters()` and `get_personal_parameters()` must exactly match keys in `model.state_dict()`.
- Shared and sequential negative sampling now use explicit negative pools. If a user has no available negatives after excluding interacted items, the dataloader raises immediately instead of silently falling back to retries.
- `output.export` is the recommendation-list export. JSON is the default single-file grouped format, JSONL/CSV/TSV are opt-in alternatives, and the export should be paired with `optimization.final_train` when exporting the best HPO configuration.
