# ASED speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on ASED alone and scores it on ASED's own test split, classifying each
utterance into one of five emotions.

[ASED](https://github.com/wgwangang/ASED_V1) (Amharic Speech Emotional Dataset,
v1, created 2021-02-12 at Northwest University) is acted Amharic speech
recorded by 65 participants (25 female, 40 male) at a time and place of their
own choosing. Eight judges reviewed every recording and kept it only if at
least five accepted it, which left 2,474 of the 4,550 expected recordings:
`neutral`, `fear`, `happy`, `sad` and `angry`, 469 to 522 each, 2.1
hours in total, each 2 to 3.9 seconds long. There is no `disgust` or
`surprise`.

The label comes from the file name,
`<emotion><folder>-<sentence>-<repeat>-<gender>-<speaker>.wav` (e.g.
`n1-01-01-01-01.wav`); the emotion codes are n, f, h, s and a. Each file also
sits in a folder numbered after its emotion, and `create_dataset` fails if the
two disagree. One file in the release is capitalised (`S4-04-02-01-29.wav`),
so names are matched ignoring case. The recordings are already 16 kHz mono;
`create_dataset` still passes them through ffmpeg so every recipe writes the
same format.

53 pairs of recordings have the same sound (always one speaker, one sentence and one
emotion in repetitions 1 and 2, so the second "repetition" is a copy of the first). With
`exclude_duplicates: true` in `dataset/config.yaml` (the default) the second file of
each pair is left out and 2,421 recordings are used; with `false` all 2,474 are used. For sentences 03 to
07, 37 speakers have all 50 files and 28 speakers have none, although the README says
each recording was judged separately.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `ASED` — the extracted repository archive, or where to download it to
- `ASED_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the repository archive from GitHub, pinned to the
latest commit. The repository's README states no license or terms of use. The licence is to be asked of the
authors (not yet done); until then use the data only for local experiments.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 1732 |
| valid | 248 |
| test | 494 |

## Quick start

Run the three commands separately. With several GPUs, `train` starts one
process per GPU by re-running its own command, so any other stage listed in the
same command would run once per GPU as well.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels \
    collect_stats --training_config conf/training.yaml
python run.py --stages train \
    --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
