# List-wise Cognitive Depth Score Evaluation

**List-wise Cognitive Depth Score (LCDS)** evaluates the content depth of a
recommendation list using a Cognitive Depth Score (CDS) label for each
recommended video. It is reported alongside Recall, NDCG, and Precision because
it measures content depth rather than behavioral relevance.

For the default setting, the per-item gain is:

```text
g(d_i) = d_i / 6
```

where `d_i` is the CDS score in `{0, 1, 2, 3, 4, 5, 6}`. If the CDS score is
`null` because the video content is insufficient for reliable assessment, the
gain is conservatively set to `0`.

The unweighted list metric is:

```text
A-LCDS@K = (1 / K) * sum_{i=1..K} g(d_i)
```

The exposure-weighted metric follows the NDCG discount:

```text
E-LCDS@K =
  sum_{i=1..K} g(d_i) / log2(i + 1)
  -----------------------------------
  sum_{i=1..K} 1 / log2(i + 1)
```

Both metrics are in `[0, 1]`, where larger values indicate recommendations with
greater cognitive depth.

## Automatic evaluation

ShortVideoSampled and ShortVideoFull enable LCDS by default. Normal validation and final-test evaluation append `A-LCDS@K` and `E-LCDS@K` to the same result dictionary, log line, result CSV, HPO metrics, and recommendation metadata as Recall/NDCG/Precision. The cutoffs are always taken from `evaluation.topk`.

The evaluator loads `scoring/results/Qwen3_7_Max_CDS_scores.jsonl` once per process and caches the item gain table across evaluator instances. It fails early if the score file or item-mapping artifacts are missing.

## Export Test Recommendations

Use `scripts/export_test_recommendations.py` to load `best_model.pth`, run only
the test split, and store the ranked item ids. This does not retrain the model.

```bash
python scripts/export_test_recommendations.py \
  --dataset ShortVideoFull \
  --models BM3 BPR FlowCF GRCN LightGCN NCF \
  --checkpoint-root outputs/checkpoints \
  --topk 10 20 50 \
  --export-dir outputs/recommendations/ShortVideoFull/test_export
```

By default, this writes one `.json` recommendation file per model. Each artifact
uses NexusRec internal zero-based item ids. The sibling metadata JSON records the
id-space contract and the Recall/NDCG/Precision values computed during test
inference.

## Compute Recall, NDCG, And LCDS From Saved Lists

After recommendation files exist, compute the behavioral metrics and LCDS
without running any model:

```bash
python scripts/compute_lcds_from_recommendations.py \
  --recommendations outputs/recommendations/ShortVideoFull/test_export/*/*.json \
  --dataset-dir datasets/ShortVideoFull \
  --cds-jsonl scoring/results/Qwen3_7_Max_CDS_scores.jsonl \
  --topk 10 20 50 \
  --output outputs/lcds/ShortVideoFull/lcds_summary.csv
```

For ShortVideoFull, internal item ids are mapped to CDS rows through:

```text
internal item_id -> source_pid -> video_id -> CDS score
```

Do not use `item_id + 1` as the CDS `video_id`; the filtered recommendation
catalog has been reindexed.

## Output layout

The local export and LCDS commands above write:

```text
outputs/recommendations/ShortVideoFull/test_export/<MODEL>/
outputs/lcds/ShortVideoFull/test_export/<MODEL>.lcds.csv
```
