# Prepared datasets

The repository expects one or both prepared bundles below:

```text
datasets/
├── ShortVideoSampled/
│   ├── inter.csv
│   ├── image_features.npy
│   ├── text_features.npy
│   ├── id_mappings.json
│   ├── items.json
│   ├── items_final_fixed.json
│   └── metadata.json
└── ShortVideoFull/
    └── ... same contract ...
```

| Bundle | Users | Items | Interactions | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShortVideoSampled | 4,450 | 25,534 | 74,869 | 57,383 | 8,041 | 9,445 |
| ShortVideoFull | 9,324 | 132,166 | 626,747 | 497,175 | 62,762 | 66,810 |

The split is chronological per user (80/10/10) after click-positive filtering and a minimum of four positive interactions. Visual features have 256 dimensions and SentenceTransformer text features have 384 dimensions.

Dataset files are excluded from Git. Google Drive and Hugging Face download links will be added before public release. To rebuild from the raw WWW2025 files, follow [`docs/dataset_repair.md`](../docs/dataset_repair.md).

Validate one downloaded bundle with `python scripts/validate_short_video_bundle.py --datasets ShortVideoFull`, or omit `--datasets` to validate both.

Default evaluation also requires the full Qwen CDS JSONL under `scoring/results/`. The evaluator maps each internal item through `source_pid -> video_id`, then appends A-LCDS and E-LCDS at the same `topk` cutoffs as the ranking metrics.
