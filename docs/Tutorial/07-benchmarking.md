# Benchmarking And Significance

This tutorial covers repeated experiment sweeps, benchmark ledgers, summary reports, and paired significance testing.

---

## Structured Benchmark Manifests

Use `configs/examples/benchmark.yaml` when you want SCOPE-Bench to expand model × dataset × seed runs, execute them, resume completed runs, and summarize metrics.

---

## Benchmark Manifest Shape

Minimal train sweep:

```yaml
experiments:
  - name: "centralized-seed-sweep"
    models: ["BPR", "LightGCN"]
    datasets: ["Beauty", "Baby"]
    seeds: [2024, 2025]
    mode: "train"
    type: "benchmark"
    comment: "seed_sweep"
    overrides:
      training:
        max_epochs: 20
```

Minimal HPO sweep:

```yaml
experiments:
  - name: "vbpr-hpo-smoke"
    model: "VBPR"
    dataset: "Beauty"
    seeds: [2024]
    mode: "hpo"
    type: "benchmark"
    comment: "bayesian_smoke"
    overrides:
      training:
        max_epochs: 20
    hpo:
      strategy: "bayesian"
      budget: 5
      resume: false
      verbose: true
```

Supported experiment fields:

| Field | Description |
|---|---|
| `name` | Human-readable experiment group name. |
| `model` / `models` | One model or a list of models. |
| `dataset` / `datasets` | One dataset or a list of datasets. |
| `seeds` | Non-empty list of seeds. Must be provided explicitly. |
| `mode` | `train` or `hpo`. |
| `type` | Output type tag, usually `benchmark`. |
| `comment` | Base comment tag. The runner appends seed and run-id suffixes. |
| `overrides` | Deep-merged config overrides. |
| `hpo` | HPO settings; used only when `mode: hpo`. Recognized fields are `strategy`, `budget`, `resume`, and `verbose`; summary support expects `bayesian`, `tpe`, or `random`. |

Command-based entries are rejected by the benchmark planner. Benchmark HPO manifests with `strategy: grid` are also rejected at plan time because grid trial histories are written through the normal save path rather than `hyper_search_dir`, which keeps summary matching unambiguous.

---

## Dry Run First

```bash
python scripts/run_benchmark.py \
  --spec configs/examples/benchmark.yaml \
  --dry-run
```

Dry-run expands the manifest without training. It writes:

```text
outputs/benchmarks/{manifest_name}/{manifest_hash12}/
  ledger.jsonl
  ledger.csv
  plan.json
```

Each planned row records the model, dataset, seed, mode, output paths, command, result file path, log directory, checkpoint directory, HPO directory, JSON-encoded overrides, and JSON-encoded HPO settings. `plan.json` and `ledger.jsonl` retain the full `hpo_json`; `ledger.csv` is for compact inspection.

The manifest hash is computed from `experiments:` only. Changing the optional `reporting:` block does not change run IDs.

---

## Execute And Resume

```bash
python scripts/run_benchmark.py --spec configs/examples/benchmark.yaml
```

Execution appends ledger rows as runs move through `running`, `completed`, or `failed`. By default, completed runs are skipped on a later invocation. Use `--no-resume` to force reruns:

```bash
python scripts/run_benchmark.py --spec configs/examples/benchmark.yaml --no-resume
```

A nonzero subprocess return code is recorded as a failed run and surfaces immediately.

---

## Summaries

```bash
python scripts/run_benchmark.py \
  --spec configs/examples/benchmark.yaml \
  --summarize
```

The CLI executes or resumes the benchmark plan first, then writes the summary artifacts. Use this command after execution has completed, or when you intentionally want pending jobs to run before summary generation.

Summary includes every planned run. Completed runs include metric values; failed, skipped, or still-incomplete runs remain visible in `summary_runs.csv` and in the failed/incomplete count in `summary.md`.

For completed runs:

- `mode: train`: loads the one-row result CSV recorded in the ledger.
- `mode: hpo`: loads the matching HPO CSV from `outputs/hyper_search`, selects the best completed trial by `target_score`, and reports that trial's `test_metrics`. HPO evaluates the validation-best state on test by default (`optimization.eval_final_test: true`), while still selecting hyperparameters by validation score.

If an HPO run enables `optimization.final_train`, the final-train result CSV, checkpoint, and optional `output.export` recommendation files are formal artifacts for downstream use. Benchmark summary still reads the HPO trial-history CSV for `mode: hpo`; it does not replace summary metrics with the nested final-train result.

Use `bayesian`, `tpe`, or `random` for HPO manifests that you plan to summarize. Serial grid search currently writes its trial CSV through the save path rather than `hyper_search_dir`.

Outputs:

```text
outputs/benchmarks/{manifest_name}/{manifest_hash12}/
  summary_runs.csv
  summary_groups.csv
  summary.md
  summary_significance.csv   # only when significance is requested
```

The optional manifest-level `reporting:` block keeps summary defaults with the experiment matrix:

```yaml
reporting:
  metrics: ["Recall@10", "NDCG@10"]
  significance_baseline: "BPR"
  significance_test: "wilcoxon"
  significance_pair_field: "seed"
```

Allowed benchmark significance pair fields are `seed` and `comment`. Prefer `seed` for multi-seed manifests. Use `comment` only when it is unique per paired run; a shared base comment such as `seed_sweep` is not enough to identify pairs. (`output_comment` and `run_id` encode the model and therefore can never pair a baseline run against a candidate run, so they are not offered.) Allowed tests are `wilcoxon` and `paired_t`.

Benchmark significance compares models only within the same `(experiment_name, dataset, mode, type)` group. Groups without the requested baseline are skipped; if the baseline is absent from all completed groups, summary generation fails.

CLI arguments override `reporting:` defaults:

```bash
python scripts/run_benchmark.py \
  --spec configs/examples/benchmark.yaml \
  --summarize \
  --significance-baseline BPR \
  --metrics NDCG@10 Recall@10 \
  --test wilcoxon \
  --significance-pair-field seed
```

---

## Standalone Significance

For repeated runs that already produced result CSVs, use:

```bash
python scripts/significance_test.py \
  --baseline "outputs/results/BPR/Beauty/stability/*.csv" \
  --candidate "outputs/results/NCF/Beauty/stability/*.csv" \
  --metrics NDCG@10 Recall@10 \
  --pair-field comment \
  --test wilcoxon
```

Standalone `--pair-field` choices are `comment` and `stem`. Reusing the same `comment` values across baseline and candidate runs is the safest pairing strategy.

The significance command is strict by design. It stops if:

- a result CSV is empty or has multiple rows
- a metric column is missing
- baseline and candidate pair keys do not match
- there are fewer than two paired runs

---

## Practical Workflow

1. Create or copy a structured benchmark YAML.
2. Run `--dry-run` and inspect `plan.json` plus `ledger.csv`; use `plan.json` for full HPO settings.
3. Execute without `--dry-run`.
4. Re-run the same command to resume skipped completed runs.
5. Run `--summarize`.
6. Add `reporting:` or CLI significance flags once paired outputs exist.
