# Video ID Alignment Note: piecewise raw/asr alignment map

## Source of truth

All current observations and manual examples in this note should be interpreted
against the local `items.json` file unless another source is explicitly named.



## Correction policy

For future corrected item metadata, keep `video_id` as the
title/source/interaction-side canonical id. Therefore these fields should stay
tied to `video_id=i`:

- `caption`
- `category`
- `first_level_category`
- `second_level_category`
- `third_level_category`
- `category_cn`
- `first_level_category_cn`
- `second_level_category_cn`
- `third_level_category_cn`
- `source_pid`
- `source_title_cn`
- `source_match_title_cn`

Only raw/asr-side fields should be read from the corrected raw/asr id. The
current full-dataset correction is the piecewise segment table below plus the
explicit canonical-side null-ASR list. Do not derive corrected item metadata
from any single local duplicate or any single global offset rule.

## Important caveat

This note records the current working map. It should not be collapsed into one
permanent offset.

## the offset is piecewise, not a single `+1`

A full-range coarse scan (`title_en[i]` vs `asr_en[i+shift]`) shows the title->asr
offset changes several times across the dataset and is **non-monotonic** (the
raw/asr side has both duplicated videos `+1` and dropped/missing videos `-1`).
Manual checks have pinned the current boundaries. The current working segment
table is:

```text
video_id  1      - 13275     offset +0
video_id  13276  - 32248     offset +1
video_id  32249              ASR = null
video_id  32250              offset +0
video_id  32251  - 32268     offset +1
video_id  32269              ASR = null
video_id  32270  - 40523     offset +0
video_id  40524  - 46000     offset +1
video_id  46001              ASR = null
video_id  46002              offset +0
video_id  46003              ASR = null
video_id  46004  - 46640     offset +1
video_id  46641              ASR = null
video_id  46642  - 47092     offset +0
video_id  47093  - 60960     offset +1
video_id  60961              ASR = null
video_id  60962  - 67875     offset +0
video_id  67876  - 67999     offset +1
video_id  68000              ASR = null
video_id  68001  - 71122     offset +1
video_id  71123              ASR = null
video_id  71124              offset +0
video_id  71125  - 71145     offset +2
video_id  71146              ASR = null
video_id  71147              offset +1
video_id  71148              offset +2
video_id  71149              ASR = null
video_id  71150              ASR = null
video_id  71151  - 71877     offset +0
video_id  71878  - 73921     offset +1
video_id  73922              ASR = null
video_id  73923  - 73998     offset +2
video_id  73999              ASR = null
video_id  74000              offset +1
video_id  74001  - 75999     offset +2
video_id  76000              ASR = null
video_id  76001              offset +1
video_id  76002              offset +2
video_id  76003              ASR = null
video_id  76004              offset +1
video_id  76005  - 80595     offset +2
video_id  80596              ASR = null
video_id  80597              offset +1
video_id  80598              offset +2
video_id  80599              ASR = null
video_id  80600  - 80601     offset +1
video_id  80602  - 87272     offset +2
video_id  87273  - 87274     offset +3
video_id  87275              offset +0
video_id  87276  - 87277     offset +2
video_id  87278  - 87288     offset +3
video_id  87289              ASR = null
video_id  87290  - 87293     offset +2
video_id  87294  - 89999     offset +3
video_id  90000              ASR = null
video_id  90001  - 90002     offset +2
video_id  90003  - 97572     offset +3
video_id  97573              ASR = null
video_id  97574  - 97575     offset +2
video_id  97576  - 97580     offset +3
video_id  97581              ASR = null
video_id  97582  - 97583     offset +2
video_id  97584  - 97586     ASR = null
video_id  97587  - 100990    offset +0
video_id  100991 - 153559    offset +1
video_id  153560 - 153561    ASR = null
```

In this table, `offset` maps a canonical title/source/interaction id to the
current raw/asr/raw_file id as `raw_id = canonical_id + offset`. Single
`ASR = null` rows are canonical items kept for `interaction.csv` completeness
but with no raw/asr counterpart. The raw/asr skip/delete ids that create some of
these offset changes are listed in the Status section below.

So a single `13276:+1` rule is incomplete. Any regenerated corrected item file
must use the piecewise mapping in this table plus explicit null-ASR handling for
dropped canonical videos.

### Status

- `fix_items_asr_raw_offset.py` still defaults to the old single `13276:+1`
  example rule. Do not use its defaults for the full correction.
- Next: regenerate corrected items from the segment table above, then verify
  that canonical fields still match `items.json`, ASR fields are read from the
  mapped raw/asr ids, and the explicit canonical-side null-ASR list is applied
  (currently: `video_id 32249`,
  `video_id 32269`, `video_id 46001`, `video_id 46003`, `video_id 46641`,
  `video_id 60961`, `video_id 68000`, `video_id 71123`, `video_id 71146`,
  `video_id 71149`,
  `video_id 71150`, `video_id 73922`, `video_id 73999`, `video_id 76000`,
  `video_id 76003`, `video_id 80596`, `video_id 80599`, `video_id 87289`,
  `video_id 90000`, `video_id 97573`, `video_id 97581`, `video_id 97584`,
  `video_id 97585`, `video_id 97586`, `video_id 153560`,
  `video_id 153561`).
- Note: the null-ASR list is canonical-side, while the skip/delete list below is
  raw/asr-side. The same numeric id may appear in both lists with different
  meanings.
- Confirmed raw/asr skip/delete list currently:
  - `video_id 13276` (duplicate of raw/asr 13275)
  - `video_id 32251` (duplicate of raw/asr 32250)
  - `video_id 40524` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 46003` (duplicate of raw/asr 46002)
  - `video_id 46004` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 47093` (duplicate of raw/asr 47092)
  - `video_id 67876` (duplicate of raw/asr 67875)
  - `video_id 68001` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 71125` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 71126` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 71149` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 71878` (duplicate of raw/asr 71877)
  - `video_id 73923` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 73924` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 74002` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 76003` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 76006` (empty/extra raw/asr item not in canonical interaction sequence)
  - `video_id 80599` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 80603` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 87280` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 87296` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 90005` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 97578` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 97586` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 100991` (extra raw/asr item not in canonical interaction sequence)
  - `video_id 153561` (extra tail raw/asr item not used by any canonical interaction record)
