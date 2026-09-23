# MSP-Podcast-only speech emotion recognition (in-domain reference)

Trains `wavlmasp` (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on MSP-Podcast alone — the same IS2025 challenge target domain —
and scores it on the same 12 test sets `egs3/multicorpus_core7/wavlmasp` and
`egs3/multicorpus_full20/wavlmasp` use: MSP-Podcast Test1/Test2 plus the ten
corpora those recipes train on (read here as evaluation-only). This is the
in-domain reference point the two cross-corpus recipes are compared against:
how much does training on MSP-Podcast alone win on MSP-Podcast itself, and
how much does it lose everywhere else?

Labels are the same **core7** set `multicorpus_core7` uses (`angry`, `sad`,
`happy`, `surprise`, `fear`, `disgust`, `neutral`; `contempt` dropped), so
macro-F1 is directly comparable across all three recipes.

## How it differs from `egs3/msppodcast/esp2_cls`

`esp2_cls` freezes WavLM and trains a small head on top; this recipe
fine-tunes the full WavLM Large encoder, matching the architecture the
`multicorpus_core7`/`multicorpus_full20`/`wavlmasp` family uses. The two are
not meant to be compared with each other; this one exists to be compared
with the multicorpus recipes.

## Quick start

```bash
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml
python run.py --stages collect_stats train \
    --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`MSPPODCAST_OUTPUT` must already point at MSP-Podcast's own prepared data
(same as `egs3/msppodcast/esp2_cls`); `MSPPODCAST_WAVLMASP_OUTPUT` controls
where this recipe's own filtered manifests and label list are written.
