# Repairing and preparing the WWW2025 short-video data

The raw release contains a title/source/interaction id sequence and a raw-video/ASR sequence that are not globally aligned. Duplicated, missing, and extra raw videos cause the offset to change across the catalog. A single global `+1` correction is therefore incorrect.

This repository keeps the complete piecewise map in `scripts/data_repair/VIDEO_ID_ALIGNMENT_NOTE.md` and the executable repair stages in `scripts/data_repair/`.

## Inputs

The default raw root is `raw/Short-Video-dataset-WWW2025/`. The repair requires:

- `interaction.csv` and `interaction_sampled.csv`;
- `fix_ShortVideo/items.json`;
- the Step-1/Step-2 alignment partitions under `fix_ShortVideo/final_hymt_step1_step2_alignment/`;
- `fix_ShortVideo/items_raw_mp4_durations.json`;
- per-video arrays under `visual_feature_fixed/`.

Canonical ids and all source/category/title fields remain tied to the interaction-side item. Only `asr_text`, `asr_text_cn`, `raw_video_id`, and `raw_file_mp4` are read from the corrected raw-side item.

## 1. Apply the piecewise ASR alignment

Run from the repository root:

```bash
python scripts/data_repair/fix_items_asr_raw_offset.py \
  --items raw/Short-Video-dataset-WWW2025/fix_ShortVideo/items.json \
  --note scripts/data_repair/VIDEO_ID_ALIGNMENT_NOTE.md \
  --output /tmp/items_fixed.json
```

The current map covers 153,561 canonical items. Explicit missing rows keep their canonical metadata and receive null ASR fields.

## 2. Build the Step-2 draft

```bash
python scripts/data_repair/build_items_final_fixed.py \
  --items raw/Short-Video-dataset-WWW2025/fix_ShortVideo/items.json \
  --match raw/Short-Video-dataset-WWW2025/fix_ShortVideo/final_hymt_step1_step2_alignment/items_match_hymt_step1_step2.json \
  --no-match raw/Short-Video-dataset-WWW2025/fix_ShortVideo/final_hymt_step1_step2_alignment/items_no_match_hymt_step1_step2.json \
  --output /tmp/items_step2_draft.json \
  --include-raw-file
```

The original alignment used HY-MT token overlap between Chinese title text and Chinese ASR plus an interaction/raw-duration agreement threshold of 0.5 seconds. A second pass accepted duration-only candidates within 0.3 seconds. The resulting automatic partition contained 131,653 matches and 21,908 unresolved items before local recovery.

## 3. Recover local no-match candidates

```bash
python scripts/data_repair/local_no_match_candidate_pass.py \
  --items-final /tmp/items_step2_draft.json \
  --items-raw raw/Short-Video-dataset-WWW2025/fix_ShortVideo/items.json \
  --durations raw/Short-Video-dataset-WWW2025/fix_ShortVideo/items_raw_mp4_durations.json \
  --interactions raw/Short-Video-dataset-WWW2025/interaction.csv \
  --manual-log raw/Short-Video-dataset-WWW2025/fix_ShortVideo/Video_ID_Alignment_Manual_log.md \
  --output-dir /tmp/item_local_recovery \
  --radius 5 \
  --duration-threshold 0.3
```

Apply the recovered rows while preserving the canonical schema:

```bash
python scripts/data_repair/apply_local_recovered_items_to_final.py \
  --items-final /tmp/items_step2_draft.json \
  --recovered /tmp/item_local_recovery/local_no_match_candidate_pass_pm5_dur0p3_recovered_items.json \
  --output raw/Short-Video-dataset-WWW2025/fix_ShortVideo/items_final_fixed.json
```

Optionally export the remaining null-ASR cohort:

```bash
python scripts/data_repair/export_item_final_no_match.py \
  --items-final raw/Short-Video-dataset-WWW2025/fix_ShortVideo/items_final_fixed.json \
  --output raw/Short-Video-dataset-WWW2025/fix_ShortVideo/item_final_no_match.json
```

## 4. Build recommendation bundles

The convenience entry rebuilds both sampled and full bundles with fixed metadata, cover-image features, SentenceTransformer title features, click-positive filtering, and temporal 80/10/10 splits:

```bash
SHORT_VIDEO_DATA_ROOT="$PWD/raw/Short-Video-dataset-WWW2025" \
python scripts/rebuild_short_video_fixed.py
```

Validate the generated artifacts before training:

```bash
python scripts/validate_short_video_bundle.py
```

The validator checks required files, interaction schema and row counts, feature shapes, finite feature values recorded by the preparation contract, and item-count consistency.

