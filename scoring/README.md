# Qwen Cognitive Depth Score annotation

This directory contains the complete code path used to annotate the full video catalog with Qwen. The rubric in `prompts.py` assigns a **Cognitive Depth Score (CDS)** from 0 to 6; `null` is reserved for items with insufficient evidence.

The seven paper-defined levels are **Affect, Point, Concept, Procedure, Mechanism, Judgment,** and **Model**.

## Run

```bash
export OPENROUTER_API_KEY="your-key"
bash scoring/run_qwen_full.sh
```

The default run uses `qwen/qwen3.7-max`, temperature `0.3`, seed `42`, a 4,096-token output budget, prompt caching, and resumable JSONL output. Override any setting through environment variables, for example:

```bash
CONCURRENCY=8 LIMIT=100 bash scoring/run_qwen_full.sh
```

`evaluate_videos.py` never persists the API key. Existing valid output rows are skipped when a run is resumed.

## Output contract

Each JSONL record contains the video id, resolved caption/category/ASR inputs, score, level name, concise reason, evidence, confidence, provider metadata, token usage, and raw model response. `scores_jsonl_to_csv.py` produces an analysis-friendly CSV using the same records.

Full outputs live in `scoring/results/` and are intentionally ignored by Git. A compact distribution summary is tracked under `benchmark_results/`.
