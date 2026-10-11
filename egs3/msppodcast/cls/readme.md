# MSP-Podcast-only speech emotion recognition (in-domain reference)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on MSP-Podcast alone -- the same IS2025 challenge target domain --
and scores it on the same 12 test sets `egs3/multicorpus_core7/cls` and
`egs3/multicorpus_full20/cls` use: MSP-Podcast Test1/Test2 plus the ten
corpora those recipes train on, each read through that corpus's own
`cls` recipe as evaluation-only. This is the in-domain reference point
the two cross-corpus recipes are compared against: how much does training on
MSP-Podcast alone win on MSP-Podcast itself, and how much does it lose
everywhere else?

Labels are the same **core7** set `multicorpus_core7` uses (`angry`, `sad`,
`happy`, `surprise`, `fear`, `disgust`, `neutral`; `contempt` dropped), so
macro-F1 is directly comparable across all three recipes.

[MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
release 2.0 is released under a licence agreement with UT Dallas, so this
recipe never downloads it. Request it from the lab's page and point
`MSPPODCAST` at a directory holding `Audios/MSP-PODCAST_*.wav` and
`Labels/Labels/labels_consensus.csv`; `create_dataset` says as much if it
cannot find them.

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

`MSPPODCAST_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
