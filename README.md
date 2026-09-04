# DOSE/Data Selection Tools

This folder intentionally keeps only the current implementation needed for
DOSE-style data selection.

## Files

- `clip_score.py`: compute CLIP image-text scores and append `clip_score`.
- `dense_guide_sampling.py`: single-dimension dense-guide WRS and
  two-dimension dense-guide WRS intersection expansion.
- `dense_guide_sampling_visualization.png`: toy visualization of the sampling
  behavior.
- `dense_guide_sampling_visualization.summary.json`: numeric summary for the
  visualization.

## CLIP Score

The released `clip_score_up.py` / `clip_score_down.py` concatenate every
`value` in `ori_conversations` and score it against the sample image with
OpenCLIP `ViT-B-32`, pretrained on `laion2b_s34b_b79k`.

Example:

```bash
python clip_score.py \
  --input input.json \
  --output output_with_clip.json \
  --image-root /path/to/images \
  --text-source ori_conversations \
  --text-mode all_values \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k
```

## Sampling

Single-dimension dense-guide WRS:

```bash
python dense_guide_sampling.py \
  --input scored.json \
  --output selected_text_20p.json \
  --method dense1d \
  --score-key yes_target_logprob_7B_NImg \
  --ratio 0.2 \
  --seed 42 \
  --meta-output selected_text_20p.meta.json
```

Two-dimension dense-guide WRS with expanding intersection:

```bash
python dense_guide_sampling.py \
  --input scored_with_clip.json \
  --output selected_20p_wrs2d.json \
  --method wrs2d \
  --score-key yes_target_logprob_7B_NImg \
  --clip-key clip_score \
  --ratio 0.2 \
  --expand-ratios 0.20,0.24,0.27,0.30,0.35,0.40,0.50,0.60,0.80,1.00 \
  --seed 42 \
  --meta-output selected_20p_wrs2d.meta.json
```

This does not use raw top scores for a single dimension. It first computes a
dense-guide weight for each dimension independently, then performs weighted
random sampling without replacement. For two dimensions, each dimension gets a
stable weighted random ordering; the merge uses the expanding top-ratio
intersection rule.

The sampling script does not compute LLM logits or CLIP scores. It consumes
existing score fields.

## Visualization Result

The generated visualization is included in this folder:

```text
dense_guide_sampling_visualization.png
dense_guide_sampling_visualization.summary.json
```

It uses 6,000 toy scores sampled from a clipped normal distribution in `[0, 1]`
and selects a 20% budget, i.e. 1,200 samples.

Left panel:

- Gray shows the original normal score distribution.
- Dark outline shows raw top-20% selection, which hard-cuts the right tail and
  produces a narrow high-score-only subset.
- Orange shows dense-guide WRS-20%, which shifts the selected distribution
  toward higher scores while retaining broader coverage than raw top-k.
- Dashed vertical lines show the KDE peak of the original distribution and the
  dense-guide target mean.

Right panel:

- Gray points are all samples in the two-dimensional text-score / CLIP-score
  space.
- Dark points show a simple raw top baseline by `text_score + clip_score`.
- Red points show the final 2D dense-guide WRS samples.
- In this demo, the 20% final budget required expanding the per-dimension
  candidate ratio to 40%:

```text
0.20 -> 493
0.24 -> 636
0.27 -> 773
0.30 -> 904
0.35 -> 1159
0.40 -> 1429
```

After the 40% candidate intersection produced enough samples, the final subset
was truncated to the requested 1,200 budget.
