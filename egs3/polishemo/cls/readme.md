# Polish Speech Emotions speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on this corpus alone and scores it on its own test split, classifying each
utterance into one of three emotions.

[Emotions in Polish speech recordings](https://doi.org/10.34808/h46c-hb44)
(Marzena Mięsikowska; published on MOST Wiedzy)
is acted Polish speech by five men aged 21 to 23. Each says six words (`nie`,
`oddaj`, `podaj`, `stop`, `tak`, `trzymaj`) five times in each of three emotions,
`angry`, `fear` and `neutral`: 450 recordings, 150 per emotion, 0.11 hours. Each
is 0.48 to 1.63 seconds long. There is no `disgust`, `happy`, `sad` or
`surprise`.

The label comes from the file name,
`<speaker>.<emotion>_<word>_<take>.wav` (e.g. `m1.a_nie_1.wav`); the emotion
codes are a (anger), s (fear) and n (neutral), as listed in the record's
description. The recordings are 44.1 kHz mono; `create_dataset` resamples them
to 16 kHz.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `POLISHEMO` — the extracted archive, or where to download it to
- `POLISHEMO_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the archive (28 MB) from MOST Wiedzy Open Research
Data, licensed CC BY 4.0. The record publishes no checksum, so the download is
checked only by the number of recordings (450).

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 315 |
| valid | 45 |
| test | 90 |

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
