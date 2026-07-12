# NexusRec Architecture

NexusRec 是一个面向学术实验的统一推荐研究框架。其核心思想是将四种推荐范式（集中式CF、集中式多模态、联邦学习、序列推荐）纳入同一套配置、数据、训练、评估机制，便于跨范式对比实验。

---

## 一、设计原则

框架最小内核只需回答五个问题：

1. 这次要运行什么实验（CLI → config）
2. 最终使用哪一份已解析配置（ConfigManager）
3. 配置走哪条唯一的数据与训练主链（training/）
4. 模型需要满足什么最小接口（RecommenderBase）
5. 哪些模型被支持、证据是什么（model YAML 中的范式标志）

不直接服务这五个问题的模块不应留在内核中。

---

## 二、实际架构（基于代码）

```
main.py
  └─ quick_start()                          [core/training/interface.py]
       ├─ _run_hpo_flow()                   [core/training/interface.py]
       │   ├─ run_unified_hpo()             [core/hpo/engine.py]
       │   ├─ run_parallel_hpo()            [core/hpo/parallel.py]
       │   └─ _maybe_run_final_train()      [core/training/interface.py]
       └─ run_training()                    [core/training/interface.py]
            └─ prepare_env()                [core/training/environment.py]
                 ├─ ConfigManager           [core/config.py]
                 ├─ prepare_data()          [core/training/environment.py]
                 └─ train_single()          [core/training/core.py]
                      ├─ get_model()        [core/model_registry.py]
                      ├─ get_trainer()      [core/model_registry.py]
                      ├─ trainer.fit()
                      └─ evaluate_final_test()
                           └─ output.export → write_recommendations()
```

HPO 只负责搜索并产出 `best_configuration`。当 `optimization.final_train.enabled=true` 时，框架会用这份 best config 重新构建一次普通训练配置，再跑一次正式训练；最终结果 CSV、checkpoint、推荐导出都来自这次正式训练，而不是 HPO trial 的中间状态。

---

## 三、目录结构

```
core/
├── config.py                  配置加载、合并、校验（overall.yaml → model.yaml → dataset.yaml → CLI）
├── model_registry.py          模型发现 (get_model)、profile 解析 (load_model_profile)、
│                              训练器路由 (get_trainer)
├── package_exports.py         公开 API 再导出
├── base/
│   ├── recommender.py         RecommenderBase：所有模型的统一基类
│   └── trainer.py             TrainerBase：优化器设置、早停、最优结果追踪
├── training/
│   ├── interface.py           quick_start()、_run_hpo_flow()、run_training() 入口
│   ├── environment.py         prepare_data()、prepare_env()
│   ├── core.py                train_single()：单次完整训练
│   └── factory.py             优化器/调度器/组件工厂
├── data/
│   ├── pipeline.py            create_loaders()：统一数据加载器工厂
│   ├── dataset.py             RecDataset：交互数据加载与 train/valid/test 分割
│   ├── dataloader.py          TrainDataLoader / EvalDataLoader
│   └── features.py            setup_centralized_features() / setup_federated_features()
├── evaluation/
│   ├── evaluator.py           TopKEvaluator：Recall/NDCG/Precision/MAP/MRR/Hit/Novelty/Diversity/Coverage 等指标计算
│   ├── ranking.py             排名指标核心计算
│   ├── topk_kernel.py         top-k 命中矩阵构建
│   ├── export.py              final test 阶段的推荐列表导出
│   └── export_contract.py     output.export 配置契约校验
├── hpo/
│   ├── engine.py              UnifiedHPOManager（含结果管理）
│   ├── optuna_backend.py      Optuna 后端（bayesian/tpe 均用 TPESampler）
│   ├── parallel.py            单机多 GPU trial 分片与合并
│   └── parameters.py          HPO 参数空间生成 + 方向 helper（is_better / worst_score）
├── benchmark/
│   ├── runner.py              结构化 benchmark manifest 计划/执行/账本
│   ├── reporting.py           汇总与配对显著性报告
│   └── options.py             reporting 选项解析
├── federated/
│   ├── trainer.py             FederatedTrainer：客户端采样、本地训练、参数聚合
│   └── dataloader.py          联邦数据加载（按用户/客户端分割）
├── sequential/
│   ├── recommender.py         SequentialRecommender：位置编码、序列工具
│   ├── trainer.py             SequentialTrainer
│   ├── evaluator.py           SequentialEvaluator
│   ├── dataset.py             序列数据集
│   ├── dataloader.py          序列 DataLoader
│   └── integration.py         序列范式集成工具
├── runtime/
│   ├── logger.py              init_logger()、TrainLogger
│   └── monitor.py             训练过程监控
└── utils/
    ├── graph.py               图操作工具
    ├── metrics.py             指标提取（extract_target_metric）
    ├── multimodal.py          多模态工具
    ├── recommendation.py      推荐列表 artifact 读取、校验与 reshape
    ├── result.py              单行实验结果 CSV 与 provenance
    └── training.py            init_seed() / train_epoch() / early_stopping() / dict2str()

scripts/
├── prepare_short_video.py     统一构建 sampled/full ShortVideo bundle
├── rebuild_short_video_fixed.py 从修复后的原始文件重建两个 bundle
├── build_short_video_text_features.py 单独重建文本特征
├── export_test_recommendations.py 导出测试推荐列表
├── compute_lcds_from_recommendations.py 计算列表 CDS 指标
├── run_benchmark.py           结构化 benchmark manifest 执行与汇总
├── significance_test.py       已完成结果 CSV 的严格配对显著性检验
├── validate_short_video_bundle.py ShortVideo bundle 验证
├── validate_models.py         模型配置验证
└── data_repair/               WWW2025 id/ASR 修复工具
models/                    （论文来源与完整列表见 docs/models.md）
├── centralized/id/
│   ├── autoencoder/       MultiVAE, RecVAE
│   ├── diffusion/         DiffRec, CFDiff（含 cfdiff_components/）
│   ├── factorization/     BPR, NCF
│   ├── flow/              FlowCF
│   └── graph/             LightGCN, SGL
├── centralized/multimodal/
│   ├── contrastive/       BM3, IDFREE, SLMRec
│   ├── diffusion/         DiffMM（含 diffmm_components/）
│   ├── factorization/     VBPR, ItemKNNCBF
│   └── graph/             DRAGON, DualGNN, FREEDOM, GRCN, LATTICE, MGCN, MMGCN, MVGAE
├── federated/id/          FedAvg, FCF, FedNCF, FedRAP, PFedRec
├── federated/multimodal/  MMFedAvg, MMFedNCF, MMFedRAP, MMFCF, MMPFedRec（含 components/）
├── sequential/id/         BERT4Rec, GRU4Rec, SASRec
├── sequential/multimodal/ HM4SR
└── templates/recommender_template.py  新模型模板

configs/
├── overall.yaml               所有参数默认值（单一真值源）
├── models/{Model}.yaml        模型专有参数（37 个模型配置）
├── datasets/{Dataset}.yaml    可选数据集专有覆盖（目录可不存在）
└── examples/
    └── benchmark.yaml         结构化 benchmark manifest 示例
```

---

## 四、分层说明

### 1. 控制面
`main.py` 是唯一正式入口。它负责解析 CLI 参数并统一调用 `quick_start()`；是否进入 HPO 流程由 `quick_start()` 内部根据 `smart_hpo` 决定。

### 2. 训练编排层
`core/training/` 负责组织整个训练流程：
- `interface.py`：对外暴露的 `quick_start()` 入口，及 HPO / parallel HPO / final_train / 标准流程的分发逻辑
- `environment.py`：配置初始化、数据准备、日志设置
- `core.py`：单次训练主函数 `train_single()`
- `factory.py`：优化器、调度器、损失函数的工厂

### 3. 配置层
单一 `core/config.py` 包含完整配置逻辑：
- 加载顺序：`overall.yaml` → `models/{Model}.yaml` → `datasets/{Dataset}.yaml` → `datasets/{Dataset}.yaml` 的 `model_overrides.{Model}` → CLI 参数
- 通过 `_post_process_config()` 统一完成设备分配、路径设置、参数规范化
- `ConfigValidationError` 在缺少必要字段时快速失败

### 4. 领域抽象层
- `RecommenderBase`：所有模型必须实现 `forward()`、`calculate_loss()`、`full_sort_predict()`
- `TrainerBase`：共享的优化器设置、早停逻辑、最优结果追踪
- `create_loaders()`：统一的数据加载器创建接口（`core/data/pipeline.py`）

### 5. 范式扩展层
- `core/federated/`：联邦学习专属逻辑（客户端采样、聚合、个性化参数）
- `core/sequential/`：序列推荐专属逻辑（位置编码、leave-one-out 评估）
- `models/`：按范式拆分的模型实现

### 6. 评估与导出层
- `TopKEvaluator` / `SequentialEvaluator`：统一产出 Recall、NDCG、Precision 等指标
- `output.export`：仅在 final test evaluation 写出按用户聚合的推荐列表
- `export_contract.py`：在配置阶段校验格式、top-k、split、新旧导出互斥关系

`output.export` 不在 HPO trial、validation、训练中 test evaluation 写出。若希望“调参后的最优配置”产出推荐文件，应开启 `optimization.final_train.enabled`，让 final_train 作为正式训练阶段触发 final test 导出。

---

## 五、模型发现约定

`get_model("ModelName")` 不再扫描多个目录，而是先读取 `configs/models/ModelName.yaml`
中的三个位：
1. `is_federated`
2. `is_multimodal_model`
3. `is_sequential`

然后路由到唯一规范包：
1. `models/centralized/id/`
2. `models/centralized/multimodal/`
3. `models/federated/id/`
4. `models/federated/multimodal/`
5. `models/sequential/id/`
6. `models/sequential/multimodal/`

随后在该范式根目录下递归定位 `modelname.py`，通常位于 `graph/`、`factorization/`、`diffusion/`、`flow/` 或 `contrastive/` 等家族子包中。

文件中的类名必须与 `--model` 参数**大小写完全一致**。

---

## 六、配置系统

参数来源优先级（高覆盖低）：
```
overall.yaml < models/{Model}.yaml < datasets/{Dataset}.yaml < dataset model_overrides.{Model} < CLI --param_overrides
```

所有默认值只在 YAML 中定义一次。模型代码通过 `config["param"]` 直接访问（KeyError = 配置缺失 = 快速失败）。

---

## 七、输出目录

```
outputs/
├── logs/{Model}/{Dataset}/          训练日志
├── checkpoints/{Model}/{Dataset}/   标准训练检查点；HPO 最优 trial 在 hpo/{strategy}/{type}/{comment}/ 下
├── results/{Model}/{Dataset}/{type}/ 指标 CSV；serial grid-HPO trial CSV 也写在这里
├── hyper_search/{Model}/{Dataset}/  Bayesian/TPE/random/parallel HPO 搜索历史
└── figures/{Model}/{Dataset}/       可视化图表
```

`optimization.final_train` 默认将正式重训结果写到 `outputs/results/{Model}/{Dataset}/final_train/`，并在 `save_model=true` 时写出标准 `best_model.pth`。推荐导出默认写到本次 run 的 `paths.save` 目录；如果 `output.export.path` 非空，则写到该目录。导出文件名包含 model、dataset、`type.comment.seed.idx0.top50.recommendations` 这类 top-k 标记；默认 JSON 是一组按用户聚合的 `items` 记录，JSONL 是一用户一行，CSV/TSV 是 `user_id,rank,item_id,score` 长表。每个数据文件旁边都有一个 `.metadata.json`，记录 NexusRec internal zero-based id space、user/item counts、row grain、ordering/score semantics、metrics、provenance 和 HPO lineage。

当 `resume_training=true` 时，训练器还会按 `checkpoint_every_n_epochs` 写出完整恢复状态 `resume_state.pth`。它保存 model、optimizer、scheduler、随机数状态、dataloader 位置和 best-tracking 状态；它不是评估用的 `best_model.pth`。

评估层有三条需要保持稳定的契约：`EvalDataLoader` 不允许 shuffle；多模态 `.npy` 特征行数必须等于 item catalog size，维度必须匹配 `features.visual_dim/text_dim`；test split 的重购目标不会被 history mask 掩掉，`Novelty` 和 item-bucket popularity 使用纯 train split 统计。

公开调用接口、模型实现契约、DataLoader 批格式、评估返回值和 benchmark/significance 入口见 [interfaces.md](interfaces.md)。
