# Scoring artifacts

Place the downloaded or locally generated full Qwen artifacts here:

- `Qwen3_7_Max_CDS_scores.jsonl`
- `Qwen3_7_Max_CDS_scores.csv`

These files are excluded from Git because of their size. The public download URL will be added before release.

The JSONL file is required by the default ShortVideo evaluator. Without it, a run fails early with a missing LCDS artifact error instead of silently omitting A-LCDS/E-LCDS.
