# MESD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MESD alone and scores it on MESD's own test split, classifying each
utterance into one of six emotions.

[MESD](https://data.mendeley.com/datasets/cy34mh68j9/5) (Mexican Emotional
Speech Database) is Mexican Spanish single-word speech from non-professional
actors (3 female, 2 male and 6 child voices, per the dataset description),
with words drawn from two word corpora
(A and B). This recipe uses the 862 human recordings in six emotions:
`angry`, `disgust`, `fear`, `happy`, `sad` (144 each) and `neutral` (143),
with `angry` and `neutral` one child recording short of 144 each in the
release.

The label comes from the file name, `<Emotion>_<voice>_<word corpus>_<word>.wav`
(e.g. `Fear_C_B_oscuridad.wav`, voice `F`/`M`/`C`); no annotation file is read.
The file name does not identify the speaker, only the voice type. The release
also holds 288 naturalness-reduced copies of female corpus-B recordings
(`<Emotion>_F_B_L<level>_<word>.wav`, edited from the human recordings); these
are skipped. The recordings are 48 kHz mono 24-bit, so `create_dataset`
resamples them to 16 kHz with ffmpeg.

The utterances are single words, from 0.20 s to 2.54 s long, and 164 of the
862 are shorter than 0.5 s. `remove_long_short` therefore applies no minimum
duration here (`min_wav_duration: 0`), unlike the 0.5 s used elsewhere, and
keeps every utterance.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MESD` — an extracted copy of the release, or where to download it to
- `MESD_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads version 5 of the dataset as one zip (101 MB) from
[Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5). The corpus is
CC BY 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 604 |
| valid | 84 |
| test | 174 |

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
